from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from PIL import Image
import io
import numpy as np
import json

from torch.serialization import MAP_SHARED

import static_data, MCTS
from screenshot_decomposition import decompose_screenshot

app = FastAPI()

def format_screenshot_info(screenshot_info):
    screenshot_map = screenshot_info['map']['name']
    mode = static_data.MODES_FOR_MAPS["Belles Rock"]
    teamA = screenshot_info['picks']['team_blue']
    teamB = screenshot_info['picks']['team_red']

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
