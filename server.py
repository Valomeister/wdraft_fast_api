import math
from io import BytesIO

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from PIL import Image
import io
import numpy as np
import json

from torch.serialization import MAP_SHARED

import GreedySearch
import static_data, MCTS
from screenshot_decomposition import decompose_screenshot

app = FastAPI()

def get_brawler_filename(brawler_label):
    label_lower = brawler_label.lower()
    result = (label_lower.replace("-", "_").replace(". ", "_")
              .replace(" ", "_").replace("&", "and").replace(".", "_"))
    result += '.webp'

    return result

def format_screenshot_info(screenshot_info):
    screenshot_map = screenshot_info['map']['name']
    mode = screenshot_info['mode']
    teamA = screenshot_info['picks']['team_blue']
    teamB = screenshot_info['picks']['team_red']

    if len(teamA) > len(teamB):
        teamA, teamB = teamB, teamA

    match = {
        "mode": mode,
        "map": screenshot_map,
        "teams": [
            teamA,
            teamB
        ]
    }

    bans_teamA = screenshot_info['bans']['team_blue']
    bans_teamB = screenshot_info['bans']['team_red']

    brawlers_set = set(bans_teamA + bans_teamB)
    bans_mask = np.zeros(96)
    for b in brawlers_set:
        if b in static_data.BRAWLERS:
            idx = static_data.BRAWLERS.index(b)
            bans_mask[idx] = 1

    return match, bans_mask

@app.post("/predict")
async def process_screenshot(file: UploadFile = File(...)):
    print("process_screenshot()")

    content = await file.read()
    screenshot_info, debug_img = decompose_screenshot(content)
    match, bans_mask = format_screenshot_info(screenshot_info)

    if len(match["teams"][0]) > len(match["teams"][1]):
        match["teams"][0], match["teams"][1] = match["teams"][1], match["teams"][0]

    def event_stream():
        # сначала отправляем распознавание
        yield json.dumps({"type": "screenshot_info", "data": screenshot_info}) + "\n"
        # потом постепенно MCTS
        for mcts_result in MCTS.get_mcts_results([match], 10_000, bans_mask):
            # сериализация каждого словаря
            yield json.dumps({"type": "mcts", "data": mcts_result}) + "\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

def get_turn_from_match(match):
    teamA = match["teams"][0]
    teamB = match["teams"][1]
    return 1 if len(teamA) <= len(teamB) else -1

@app.post("/predict_ios")
async def handle_image(file: UploadFile = File(...), detail: int | None = Form(None)):
    # читаем байты
    content = await file.read()

    screenshot_info, debug_img = decompose_screenshot(content, False)
    match, bans_mask = format_screenshot_info(screenshot_info)

    raw_results = GreedySearch.get_greedy_search_results(match, bans_mask)

    action_probs = raw_results["action_probs"]
    max_action_probs = raw_results["max_action_probs"]
    avg_value = raw_results["avg_value"]

    n = 10
    rows = 2
    cols = math.ceil(n / rows)
    icon_size = 100
    top_n = sorted(
        action_probs.items(),
        key=lambda x: x[1],
        reverse=True
    )[:n]

    canvas = Image.new("RGB", (cols * icon_size, rows * icon_size), "white")

    for i, (brawler, prob) in enumerate(top_n):
        row = i // cols
        col = i % cols
        icon_filename = get_brawler_filename(brawler)
        icon_path = f"brawlers_icons_medium/{icon_filename}"
        icon = Image.open(icon_path).convert("RGB")
        canvas.paste(icon, (col * icon_size, row * icon_size))


    # 3. Сохраняем в память
    buf = BytesIO()
    canvas.save(buf, format="JPEG", quality=100)
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="image/jpeg"
    )