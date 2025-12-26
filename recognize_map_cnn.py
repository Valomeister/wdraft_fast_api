import time
from PIL import Image
import cv2
import os
import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
from pathlib import Path
import recognition_utils

class SimpleCNN(nn.Module):
    def __init__(self, num_classes, input_width, input_height):
        super(SimpleCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * (input_height // 8) * (input_width // 8), 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

REGION_WIDTH = 400
REGION_HEIGHT = 100

modes = ['brawlball', 'knockout', 'heist', 'gems', 'bounty', 'hotzone']
modes_color_dict = {
    'brawlball': (215, 165, 145),
    'knockout': (48, 140, 245),
    'heist': (205, 100, 210),
    'gems': (233, 65, 155),
    'bounty': (243, 210, 25),
    'hotzone': (90, 65, 220),
}

modes_normed_color_dict = {
    'brawlball': (1.0, 0.7674418604651163, 0.6744186046511628),
    'knockout': (0.19591836734693877, 0.5714285714285714, 1.0),
    'heist': (0.9761904761904762, 0.47619047619047616, 1.0),
    'gems': (1.0, 0.27896995708154504, 0.6652360515021459),
    'bounty': (1.0, 0.8641975308641975, 0.102880658436214),
    'hotzone': (0.4090909090909091, 0.29545454545454547, 1.0)
}

tgA = 10.71

# ---- загрузка модели ----
checkpoint = torch.load("models/map_ideal_crop_w_bar.pth", map_location="cpu")
classes = checkpoint["classes"]
num_classes = len(classes)

model = SimpleCNN(num_classes, REGION_WIDTH, REGION_HEIGHT)
model.load_state_dict(checkpoint["model"])
model.eval()

device = torch.device("cpu")
model.to(device)

transform = T.Compose([
    T.ToPILImage(),
    T.Resize((REGION_HEIGHT, REGION_WIDTH)),
    T.ToTensor()
])

def get_best_match_flexible(img_slice):
    avg_color = img_slice.mean(axis=(0, 1))  # B, G, R средние
    avg_color_scaled = avg_color / max(1, avg_color.max())

    best_mode = None
    best_d = float('inf')

    ds = []
    for name, color in modes_normed_color_dict.items():
        d = np.linalg.norm(avg_color_scaled - np.array(color))
        ds.append((name, d))
        if d < best_d:
            best_d = d
            best_mode = name

    return best_mode


def get_mode(img, debug):
    start = time.time()

    x1, y1, x2, y2 = 0, 0, REGION_WIDTH, REGION_HEIGHT
    region = img[y1:y2, x1:x2]

    b, g, r = cv2.split(region)
    black_mask = (b < 30) & (g < 30) & (r < 30)
    black_mask = black_mask.astype(np.uint8) * 255

    best_bottom_error = 1e11
    best_bottom_border = None
    for border in range(2, black_mask.shape[0] // 2):
        top_space = black_mask[border - 2:border, :]
        bottom_space = black_mask[border + 1:border + 3, :]

        # bottom border
        bottom_error = (bottom_space == 255).sum() / bottom_space.size
        top_error = (top_space == 0).sum() / top_space.size
        total_error = bottom_error + top_error
        if total_error <= best_bottom_error:
            best_bottom_error = total_error
            best_bottom_border = border

    region_top_slice = region[best_bottom_border + 1:best_bottom_border + 5, :]
    best_mode = get_best_match_flexible(region_top_slice)

    if debug:
        end = time.time()
        print(f"get_mode() -> {end - start:.2f} сек")

    return best_mode


def get_pixels(x0, tg, ctg, limit_top, limit_bottom):
    pixels = []
    for y in range(limit_top, limit_bottom):
        x = max(0, int(-y * ctg + x0))
        pixels.append((x, y))

    return pixels


def calculate_horizontal_error(img, x0, tg, ctg, mode, limit_top, limit_bottom):
    pixels = get_pixels(x0, tg, ctg, limit_top, limit_bottom)
    mode_color_avg = modes_color_dict[mode]
    reach = 1
    total_error = 0
    for x, y in pixels:
        x_start = max(0, x - reach)
        left_pixels = img[y, x_start:x]
        if left_pixels.size:
            left_avg_color = np.mean(left_pixels, axis=0)
        else:
            left_avg_color = np.zeros(3)
        left_error = -np.linalg.norm(mode_color_avg - left_avg_color)

        right_pixels = img[y, x:x +reach]
        right_avg_color = np.mean(right_pixels, axis=0)
        right_error = np.linalg.norm(mode_color_avg - right_avg_color)

        total_error += left_error + right_error
    return total_error


def find_best_left_border(img, tg, ctg, mode, limit_top, limit_bottom):
    best_error = float('inf')
    best_x0 = None
    for x0 in range(0, int(REGION_WIDTH / 3)):
        error = calculate_horizontal_error(img, x0, tg, ctg, mode, limit_top, limit_bottom)
        if error < best_error:
            best_error = error
            best_x0 = x0

    return best_x0


def extract_map_box(img, mode, debug):
    start = time.time()

    x1, y1, x2, y2 = 0, 0, REGION_WIDTH, REGION_HEIGHT
    region = img[y1:y2, x1:x2]

    h, w = img.shape[:2]
    if h > 710:
        map_margin_top = 20
        map_height = 78
        map_width = 334
    else:
        map_margin_top = round(2 / 71 * h)
        map_height = round(39 / 355 * h)
        map_width = round(264 / 62 * map_height)

    best_left_border = find_best_left_border(img, tgA, 1 / tgA, mode,
                                             map_margin_top, map_margin_top + map_height)
    best_right_border = best_left_border + map_width
    best_top_border = map_margin_top
    best_bottom_border = map_margin_top + map_height

    map_x1 = best_left_border
    map_y1 = best_top_border
    map_x2 = int(-100 * 1 / tgA + best_right_border)
    map_y2 = best_bottom_border

    if debug:
        end = time.time()
        print(f"extract_map_box() -> {end - start:.2f} сек")

        cv2.line(img, (best_left_border - 1, 0), (int(-100 * 1 / tgA + best_left_border) - 1,
                REGION_HEIGHT), color=(0, 255, 0), thickness=1)
        cv2.line(img, (best_right_border + 1, 0), (int(-100 * 1 / tgA + best_right_border) + 1,
                REGION_HEIGHT), color=(0, 255, 0), thickness=1)
        cv2.line(img, (0, best_top_border - 1), (REGION_WIDTH, best_top_border - 1),
                 color=(0, 255, 0), thickness=1)
        cv2.line(img, (0, best_bottom_border + 1), (REGION_WIDTH, best_bottom_border + 1),
                 color=(0, 255, 0), thickness=1)

        cv2.line(img, (map_x1 - 1, map_y1), (map_x1 - 1, map_y2), color=(255, 255, 255), thickness=1)
        cv2.line(img, (map_x2 + 1, map_y1), (map_x2 + 1, map_y2), color=(255, 255, 255), thickness=1)
        cv2.line(img, (map_x1, map_y1 - 1), (map_x2, map_y1 - 1), color=(255, 255, 255), thickness=1)
        cv2.line(img, (map_x1, map_y2 + 1), (map_x2, map_y2 + 1), color=(255, 255, 255), thickness=1)

    return (map_x1, map_y1, map_x2, map_y2)


def recognize_map(img, map_box, debug):
    start = time.time()

    x1, y1, x2, y2 = map_box
    map_region = img[y1:y2, x1:x2]
    map_region = cv2.cvtColor(map_region, cv2.COLOR_BGR2RGB)
    # cv2.imshow("map_region", map_region)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    tensor = transform(map_region).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor)
        pred_idx = out.argmax(dim=1).item()
        pred_name = classes[pred_idx]
        confidence = torch.softmax(out, dim=1)[0, pred_idx].item()

    print(pred_name)
    if pred_name == "Belles Rock_en":
        pred_name = "Belle's Rock_en"
    elif pred_name == "Belles Rock_ru":
        pred_name = "Belle's Rock_ru"

    if debug:
        end = time.time()
        print(f"recognize_map() -> {end - start:.2f} сек")

        text = f"{pred_name} {confidence * 100:.0f}%"

        # cv2.putText(img, text, (5, REGION_HEIGHT + 15),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
        recognition_utils.put_text_with_background(img, text, (5, REGION_HEIGHT + 15))

        # cv2.imshow(screenshot_path, img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    return (map_box, pred_name, confidence)

def screenshot_to_map(img, debug):

    mode = get_mode(img, debug)

    map_box = extract_map_box(img, mode, debug)

    recognized_map = recognize_map(img, map_box, debug)

    return recognized_map

if __name__ == "__main__":

    screenshots_folder = Path("screenshots")

    screenshots = [str(f) for f in screenshots_folder.glob("*") if f.is_file()]

    for screenshot_path in screenshots:
        img = cv2.imread(screenshot_path)
        print(screenshot_to_map(img, True))


