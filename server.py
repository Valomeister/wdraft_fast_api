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

    screenshot_info, debug_img = decompose_screenshot(content, True)
    match, bans_mask = format_screenshot_info(screenshot_info)

    raw_results = GreedySearch.get_greedy_search_results(match, bans_mask)

    action_probs = raw_results["action_probs"]
    max_action_probs = raw_results["max_action_probs"]
    avg_value = raw_results["avg_value"]

    top_10 = sorted(
        action_probs.items(),
        key=lambda x: x[1],
        reverse=True
    )[:]

    lines = []

    if detail == 1:
        lines.append(', '.join([i[0] for i in top_10]))

    if detail >= 2:
        for i, (name, prob) in enumerate(top_10, start=1):
            index_part = str(i) + "." + ' ' * (4 - len(str(i)))
            brawler_part = name
            prob_part = f"   ({prob / max_action_probs * 100:.0f}%)"
            lines.append(index_part + brawler_part + prob_part)

    if detail >= 3:
        lines.append("")
        cur_turn = get_turn_from_match(match)
        print(match)
        scaled_val = avg_value + 1 / 2
        if cur_turn == 1:
            blue_val = scaled_val
        else:
            blue_val = 1 - scaled_val
        red_val = 1 - blue_val
        lines.append(f"VICTORY: {blue_val*100:.0f}% vs {red_val*100:.0f}%")

    if detail >= 4:
        lines.append("")
        debug_info = ''
        debug_info += f"{screenshot_info["mode"]}, {screenshot_info["map"]["name"]}\n"
        debug_info += f"{', '.join(screenshot_info["picks"]["team_blue"])}  vs  {', '.join(screenshot_info["picks"]["team_red"])}\n"
        debug_info += f"Bans: {', '.join(screenshot_info["bans"]["team_blue"])} + {', '.join(screenshot_info["bans"]["team_red"])}\n"
        lines.append(debug_info)

    result_str = "\n".join(lines)

    print(result_str)

    IMAGE_PATH = "sample_output.jpg"  # ← путь к фото на ПК

    # 1. Открываем изображение
    img = Image.open(IMAGE_PATH)
    img = img.convert("RGB")

    # 3. Сохраняем в память
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="image/jpeg"
    )