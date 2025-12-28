import math
import time
from io import BytesIO

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from PIL import Image, ImageDraw, ImageFont
import io
import numpy as np
import json

from torch.serialization import MAP_SHARED

import GreedySearch
import static_data, MCTS
from screenshot_decomposition import decompose_screenshot

app = FastAPI()

font = ImageFont.truetype("arial.ttf", size=32)

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

    match = {
        "mode": mode,
        "map": screenshot_map,
        "teams": [
            teamA,
            teamB
        ]
    }

    if len(teamA) == len(teamB):
        turn = screenshot_info["turn"]
    elif len(teamA) > len(teamB):
        turn = -1
    else:
        turn = 1


    match_neutral = dict(match)
    if len(teamA) > len(teamB) or len(teamA) == len(teamB) and turn == -1:
        match_neutral["teams"] = [teamB, teamA]

    bans_teamA = screenshot_info['bans'][0]
    bans_teamB = screenshot_info['bans'][1]

    brawlers_set = set(bans_teamA + bans_teamB)
    bans_mask = np.zeros(96)
    for b in brawlers_set:
        if b in static_data.BRAWLERS:
            idx = static_data.BRAWLERS.index(b)
            bans_mask[idx] = 1

    return match, match_neutral, bans_mask, turn

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



def concat_vertical_pil(imgs):
    imgs_widths = [img.width for img in imgs]
    if imgs_widths.count(imgs_widths[0]) != len(imgs_widths):
        raise ValueError("Ширина изображений должна совпадать")

    total_height = sum([img.height for img in imgs])
    result = Image.new(
        imgs[0].mode,
        (imgs[0].width, total_height)
    )

    cur_h = 0
    for img in imgs:
        result.paste(img, (0, cur_h))
        cur_h += img.height

    return result

def generate_map_info_canvas(match, cols, language, theme, height, icon_size=100):
    bg_color = "white" if theme == "light" else "black"
    canvas = Image.new("RGB", (cols * icon_size, height), bg_color)
    draw = ImageDraw.Draw(canvas)

    if language == "ru":
        text = f"{static_data.MODES_RU[match["mode"]]}, {static_data.MAPS_RU[match["map"]]}"
    else:
        text = f"{static_data.MODES_EN[match["mode"]]}, {match["map"]}"

    text_color = "black" if theme == "light" else "white"
    draw.text(
        (cols * icon_size // 2, height // 2),
        text=text,
        fill=text_color,
        font=font,
        anchor="mm"
    )

    return canvas

def generate_state_canvas(match, bans, turn, cols, theme, icon_size=100):
    margin_vertical = 10
    bg_color = "white" if theme == "light" else "black"
    canvas = Image.new("RGB", (cols * icon_size, icon_size + 2 * margin_vertical), bg_color)
    draw = ImageDraw.Draw(canvas)

    teams_icons = [[], []]
    for team in range(2):
        for brawler in match["teams"][team]:
            icon_filename = get_brawler_filename(brawler)
            icon_path = f"img/brawlers_icons_medium/{icon_filename}"
            icon = Image.open(icon_path).convert("RGB")
            teams_icons[team].append(icon)

        if team == 1:
            teams_icons[team] = teams_icons[team][::-1]

        brawler_count = len(match["teams"][team])
        if brawler_count < 3:
            for i in range(3 - brawler_count):
                placeholder_folder = "img/placeholders"
                placeholder_filename = (f"placeholder_blue_{theme}.png", f"placeholder_red_{theme}.png")[team]
                icon = Image.open(f"{placeholder_folder}/{placeholder_filename}").convert("RGB")

                teams_icons[team].append(icon)


    turn_idx = 0 if turn == 1 else 1
    turn_brawler_count = len(match["teams"][turn_idx])
    print(match["teams"][turn_idx])
    active_placeholder_folder = "img/placeholders"
    active_placeholder_filename = (f"placeholder_blue_active_{theme}.png", f"placeholder_red_active_{theme}.png")[turn_idx]
    active_placeholder_icon = Image.open(f"{active_placeholder_folder}/{active_placeholder_filename}").convert("RGB")
    teams_icons[turn_idx][turn_brawler_count] = active_placeholder_icon

    teams_icons[1] = teams_icons[1][::-1]

    margin_left = (cols * icon_size - 6 * icon_size) // 2
    for team in range(2):
        for i in range(3):
            x_pos = margin_left + team * 3 * icon_size + i * icon_size
            icon = teams_icons[team][i]
            canvas.paste(icon, (x_pos, margin_vertical))

    for c in range(6 + 1):
        draw.line(
            (margin_left + c * icon_size, margin_vertical, margin_left + c * icon_size, margin_vertical + 1 * icon_size - 2),
            fill="black",
            width=3
        )

    for r in range(2):
        draw.line(
            (margin_left + 0, margin_vertical + r * icon_size, margin_left + 6 * icon_size, margin_vertical + r * icon_size),
            fill="black",
            width=3
        )

    bans_icons = [[], []]
    for team in range(2):
        for ban in bans[team]:
            icon_path = f"img/ban_icons/{ban}.webp"
            icon = Image.open(icon_path).convert("RGBA")
            icon = icon.resize(
                (33, 33),
            )
            bans_icons[team].append(icon)

    for team in range(2):
        pos_x = margin_left - 33 - 15 if team == 0 else margin_left + 6 * icon_size + 15
        for i in range(min(3, len(bans[team]))):
            icon = bans_icons[team][i]
            pos_y = margin_vertical + i * 33
            canvas.paste(
                icon,
                (pos_x, pos_y),
                icon  # маска
            )

    return canvas

def generate_suggestions_canvas(top_brawlers, n, rows, icon_size=100):
    top_n = top_brawlers[:n]
    cols = math.ceil(len(top_n) / rows)
    canvas = Image.new("RGB", (cols * icon_size, rows * icon_size), "black")
    draw = ImageDraw.Draw(canvas)

    for i, (brawler, prob) in enumerate(top_n):
        row = i // cols
        col = i % cols
        icon_filename = get_brawler_filename(brawler)
        icon_path = f"img/brawlers_icons_medium/{icon_filename}"
        icon = Image.open(icon_path).convert("RGB")
        canvas.paste(icon, (col * icon_size, row * icon_size))

        if i < 3:
            medal_path = f"img/medals/medal_{i + 1}.png"
            medal = Image.open(medal_path).convert("RGBA")
            medal_size = 60
            medal = medal.resize(
                (medal_size, medal_size),
            )
            canvas.paste(medal, ((col + 1) * icon_size - int(medal_size * 0.76), row * icon_size - int(medal_size * 0.1)), medal)

    for c in range(cols - 1):
        draw.line(
            ((c + 1) * icon_size, 0, (c + 1) * icon_size, rows * icon_size),
            fill="black",
            width=3
        )

    for r in range(rows - 1):
        draw.line(
            (0, (r + 1) * icon_size, cols * icon_size, (r + 1) * icon_size),
            fill="black",
            width=3
        )

    return canvas

def remove_illegal_from_probs(action_probs, match, bans_mask):
    processed_action_probs = {}
    combined_teams = match["teams"][0] + match["teams"][1]
    for brawler, prob in action_probs.items():
        brawler_idx = static_data.BRAWLERS_IDXS[brawler]
        if not (brawler in combined_teams or bans_mask[brawler_idx]):
            processed_action_probs[brawler] = prob

    return processed_action_probs


@app.post("/predict_ios")
async def handle_image(file: UploadFile = File(...), detail: int | None = Form(None)):
    # читаем байты
    content = await file.read()

    screenshot_info, debug_img = decompose_screenshot(content, False)
    match, match_neutral, bans_mask, turn = format_screenshot_info(screenshot_info)
    lang = screenshot_info["map"]["lang"]

    raw_results = GreedySearch.get_greedy_search_results(match_neutral, bans_mask)

    action_probs = raw_results["action_probs"]
    max_action_probs = raw_results["max_action_probs"]
    avg_value = raw_results["avg_value"]

    processed_action_probs = remove_illegal_from_probs(action_probs, match, bans_mask)
    top_brawlers = sorted(
        processed_action_probs.items(),
        key=lambda x: x[1],
        reverse=True
    )

    n = 10
    rows = 1
    cols = math.ceil(n / rows)
    map_info_h = 44

    theme = "dark"
    map_info_canvas = generate_map_info_canvas(match, language=lang, cols=cols, height=map_info_h, theme=theme)
    state_canvas = generate_state_canvas(match, screenshot_info["bans"], turn, cols=cols, theme=theme)
    suggestions_canvas = generate_suggestions_canvas(top_brawlers, n=n, rows=rows)

    result = concat_vertical_pil([map_info_canvas, suggestions_canvas, state_canvas])

    # 3. Сохраняем в память
    buf = BytesIO()
    result.save(buf, format="JPEG", quality=100)
    buf.seek(0)


    return StreamingResponse(
        buf,
        media_type="image/jpeg"
    )