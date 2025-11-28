import cv2
from pathlib import Path
import recognize_bans_cnn, recognize_picks_cnn, recognize_map_cnn
import numpy as np

# {'map': {'name': 'Dry Season', 'lang': 'ru'},
# 'picks': {'team_blue': ['GENE', 'MAX'], 'team_red': ['MR. P', 'GUS']},
# 'bans': {'team_blue': ['MINA', 'BELLE', 'GUS'], 'team_red': ['GUS', 'OLLIE', 'MINA']}}
def decompose_screenshot(file_bytes):
    # Преобразуем байты в numpy array
    nparr = np.frombuffer(file_bytes, np.uint8)
    # Декодируем изображение
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    h, w = img.shape[:2]

    bans = recognize_bans_cnn.screenshot_to_banned_brawlers(img, debug=True)
    picks = recognize_picks_cnn.screenshot_to_picked_brawlers(img, debug=True)
    map = recognize_map_cnn.screenshot_to_map(img, debug=True)

    if False:
        cv2.imshow("Эщкере", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    bans.sort(key=lambda x: (x[0][1], x[0][0]))
    picks.sort(key=lambda x: x[0][0])

    decomposition_info = {
        "map": {
            "name": "?",
            "lang": "?"
        },
        "picks": {
            "team_blue": [],
            "team_red": []
        },
        "bans": {
            "team_blue": [],
            "team_red": []
        }
    }

    if map:
        decomposition_info["map"] = {
            "name": map[1][:-3],
            "lang": map[1][-2:]
        }

    if picks:
        picks_blue = []
        picks_red = []
        for pick in picks:
            x1, y1, x2, y2 = pick[0]
            margin_left = x1
            margin_right = w - x2
            if margin_left < margin_right:
                picks_blue.append(pick[1])
            else:
                picks_red.append(pick[1])

        decomposition_info["picks"] = {
            "team_blue": picks_blue,
            "team_red": picks_red
        }

    if bans:
        bans_blue = [ban[1] for ban in bans[::2]]
        bans_red = [ban[1] for ban in bans[1::2]]

        decomposition_info["bans"] = {
            "team_blue": bans_blue,
            "team_red": bans_red
        }

    return decomposition_info, img


