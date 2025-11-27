from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import io
import numpy as np

import static_data, MCTS
from screenshot_decomposition import decompose_screenshot

app = FastAPI()

def format_screenshot_info(screenshot_info):
    map = screenshot_info['map']['name']
    mode = static_data.MODES_FOR_MAPS[map]
    teamA = screenshot_info['picks']['team_blue']
    teamB = screenshot_info['picks']['team_red']

    match = {
        "mode": mode,
        "map": map,
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


def get_readable_mcts_prediction(match, searches, bans_mask, n):
    probs, val = MCTS.get_mcts_results([match], searches, bans_mask)
    top_n = MCTS.top_n_brawlers(probs, n)
    readable = "Топ драфтов:\n" + "\n".join([f"{pos}. {static_data.BRAWLERS[idx]}" for pos, idx in top_n])
    return readable


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # читаем файл
    content = await file.read()
    image = Image.open(io.BytesIO(content))

    # !!! здесь вызываешь свой MCTS + модель !!!
    # result = your_model.process(image)

    result = {"status": "ok", "dummy": 123}  # тестовый ответ

    return JSONResponse(result)
