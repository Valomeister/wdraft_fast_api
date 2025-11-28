import time
import cv2
import recognition_utils
import torch
import torch.nn as nn
import torchvision.transforms as T
from pathlib import Path

class IconNet(nn.Module):
    def __init__(self, num_classes, dropout_prob=0.3):
        super().__init__()

        # Свёрточные блоки с BatchNorm
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),  # 55 → 55
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),                 # 55 → 27

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),                 # 27 → 13

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, padding=1)       # 13 → 7
        )

        # Полносвязные слои с Dropout
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 512),
            nn.ReLU(),
            # nn.Dropout(dropout_prob),
            nn.Linear(512, 256),
            nn.ReLU(),
            # nn.Dropout(dropout_prob),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

IMG_SIZE = 55

checkpoint = torch.load("models/icon_randomcrop_model.pth", map_location="cpu")
classes = checkpoint["classes"]
num_classes = len(classes)

model = IconNet(num_classes)
model.load_state_dict(checkpoint["model"])
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

transform = T.Compose([
    T.ToPILImage(),
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor()
])

ban_templates = ["templates/bans/ban_blue.png",
                  "templates/bans/ban_red.png",
                  "templates/bans/ban_blue_wide_screen.png",
                  "templates/bans/ban_red_wide_screen.png",
                  "templates/bans/ban_blue_misteffa.jpg",
                  "templates/bans/ban_red_misteffa.jpg"]

template_directory = "bans_jpg"

threshold = 0.988

def get_bottom_section_h(img):
    h, w = img.shape[:2]
    if h > 710:
        bottom_section_h = 227
    else:
        bottom_section_h = 0.3156 * h

    return bottom_section_h

def get_icon_boxes(img, ban_boxes, debug):
    start = time.time()

    h, w = img.shape[:2]

    bottom_section_h = get_bottom_section_h(img)
    ban_size = round(0.218 * bottom_section_h)
    delta_x = round(0.02643 * bottom_section_h)
    delta_y = round(0.039647 * bottom_section_h)

    icon_boxes = []
    for box in ban_boxes:
        x1, y1, x2, y2 = box
        rx1 = max(0, x1 - ban_size - delta_x)
        ry1 = max(0, y2 - ban_size - delta_y)
        rx2 = x1 - round(delta_x)
        ry2 = y2 - round(delta_y)
        icon_box = (rx1, ry1, rx2, ry2)
        icon_boxes.append(icon_box)

    if debug:
        end = time.time()
        print(f"get_icon_boxes() -> {end - start:.2f} сек")

    return icon_boxes



def recognize_brawlers(img, icon_boxes, debug):
    start = time.time()

    recognized = []
    for i, box in enumerate(icon_boxes):
        x1, y1, x2, y2 = box

        avatar = img[y1:y2, x1:x2]

        # приводим к размеру и формату
        avatar_rgb = cv2.cvtColor(avatar, cv2.COLOR_BGR2RGB)
        # cv2.imshow("Avatar", avatar_rgb)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        tensor = transform(avatar_rgb).unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(tensor)
            pred_idx = out.argmax(dim=1).item()
            pred_name = classes[pred_idx]
            confidence = torch.softmax(out, dim=1)[0, pred_idx].item()

        recognized.append((box, pred_name, confidence))


    if debug:
        end = time.time()
        print(f"recognize_brawlers() -> {end - start:.2f} сек")

        for box, name, score in recognized:
            x1, y1, x2, y2 = box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            text = f"{name} {score * 100:.0f}%"

            # cv2.putText(img, text, (x1, y1 - 5),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            recognition_utils.put_text_with_background(img, text, (x1, y1 - 5))

        # cv2.imshow(screenshot_path, img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    return recognized


def screenshot_to_banned_brawlers(img, debug):
    h, w = img.shape[:2]

    start_y = h - round(get_bottom_section_h(img))
    ban_boxes = recognition_utils.find_templates(img, ban_templates, threshold, start_y, debug=debug)

    icon_boxes = get_icon_boxes(img, ban_boxes, debug=debug)

    recognized_brawlers = recognize_brawlers(img, icon_boxes, debug)

    return recognized_brawlers


if __name__ == "__main__":
    screenshots_folder = Path("screenshots")

    screenshots = [str(f) for f in screenshots_folder.glob("*") if f.is_file()]

    for screenshot_path in screenshots:
        img = cv2.imread(screenshot_path)
        print(screenshot_to_banned_brawlers(img, debug=True))