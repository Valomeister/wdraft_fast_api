import time

import recognition_utils
from train_picks_net import SimpleCNN
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from pathlib import Path

# --- Настройки ---
IMG_SIZE = 125
REGION_SIZE = 180

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load("models/character_cnn.pth", map_location=device)
classes = checkpoint["classes"]
num_classes = len(classes)

model = SimpleCNN(num_classes)
model.load_state_dict(checkpoint["model"])
model.eval()

# --- Преобразование для PyTorch ---
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

tick_templates = ["templates/ticks/cropped_tick_team_blue.png", "templates/ticks/cropped_tick_team_red.png"]

# --- Настройки распознавания галочек ---
threshold = 0.9
nms_overlap_thresh = 0.1



def non_max_suppression(boxes, overlapThresh):
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes)
    pick = []
    x1 = boxes[:, 0];
    y1 = boxes[:, 1];
    x2 = boxes[:, 2];
    y2 = boxes[:, 3]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    while len(idxs) > 0:
        last = idxs[-1]
        pick.append(last)
        idxs = idxs[:-1]
        xx1 = np.maximum(x1[last], x1[idxs])
        yy1 = np.maximum(y1[last], y1[idxs])
        xx2 = np.minimum(x2[last], x2[idxs])
        yy2 = np.minimum(y2[last], y2[idxs])
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / area[idxs]
        idxs = idxs[overlap <= overlapThresh]
    return boxes[pick].astype(int)

def get_bottom_section_h(img):
    h, w = img.shape[:2]
    if h > 710:
        bottom_section_h = 227
    else:
        bottom_section_h = 0.3156 * h

    return bottom_section_h


def find_crude_icon_boxes(img, tick_boxes, debug):
    start = time.time()

    h, w = img.shape[:2]

    bottom_section_h = get_bottom_section_h(img)
    icon_size = 0.515 * bottom_section_h

    cv2.rectangle(img, (0, h - round(bottom_section_h)), (w, h - round(bottom_section_h)), (0, 255, 0), 2)

    icon_boxes = []
    for x1, y1, x2, y2 in tick_boxes:
        icon_x1 = max(0, x1 - round(icon_size * 1))
        icon_y1 = max(0, y2 - round(icon_size * 1)) - 10
        icon_x2 = min(w, x1)
        icon_y2 = min(h, y2) - 10

        offset = 10
        icon_box = (icon_x1 - offset, icon_y1 - offset, icon_x2 + offset, icon_y2 + offset)
        icon_boxes.append(icon_box)

        if debug:
            x1, y1, x2, y2 = icon_box
            cv2.rectangle(img, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (0, 255, 0), 2)

    if debug:
        end = time.time()
        print(f"find_crude_icon_boxes() -> {end - start:.2f} сек")



    return icon_boxes

def recognize_brawlers(img, icon_boxes, debug):
    start = time.time()

    recognized = []
    for (x1, y1, x2, y2) in icon_boxes:
        if x1 >= x2 or y1 >= y2:
            print("recognize_picks_cnn.recognize_brawlers - skip")
            continue
        region = img[y1:y2, x1:x2]
        region = cv2.cvtColor(region, cv2.COLOR_BGR2RGB)
        # cv2.imshow("region", region)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        region_tensor = transform(region).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(region_tensor)
            probs = F.softmax(out, dim=1)
            pred_idx = torch.argmax(probs, dim=1)
            pred_name = classes[pred_idx.item()]
            confidence = torch.softmax(out, dim=1)[0, pred_idx].item()

        recognized.append(((x1, y1, x2, y2), pred_name, confidence))

        if debug:
            text = f"{pred_name} {confidence * 100:.0f}%"
            # cv2.putText(img, text, (x1, y1 - 5),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            recognition_utils.put_text_with_background(img, text, (x1, y1 - 5))

    if debug:
        end = time.time()
        print(f"recognize_brawlers() -> {end - start:.2f} сек")

        # cv2.imshow(screenshot_path, img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    return recognized

def screenshot_to_picked_brawlers(img, debug):
    h, w = img.shape[:2]

    start_y = h - round(get_bottom_section_h(img))
    tick_boxes = recognition_utils.find_templates(img, tick_templates, threshold, start_y, debug=debug)

    # icon_boxes = get_icon_boxes(img, tick_boxes, debug=debug)
    icon_boxes = find_crude_icon_boxes(img, tick_boxes, debug=debug)

    recognized_brawlers = recognize_brawlers(img, icon_boxes, debug=debug)

    return recognized_brawlers

if __name__ == "__main__":
    screenshots_folder = Path("screenshots")

    screenshots = [str(f) for f in screenshots_folder.glob("*") if f.is_file()]

    for screenshot_path in screenshots:
        img = cv2.imread(screenshot_path)
        print(screenshot_to_picked_brawlers(img, debug=True))
