import time

import cv2
import numpy as np

# Объединение близких boxes в один box
def non_max_suppression(boxes, overlapThresh):
    if len(boxes) == 0:
        print("non_max_suppression - skip")
        return np.array([])
    boxes = np.array(boxes)
    pick = []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
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


def find_templates(img, templates, threshold, start_y, debug):
    start = time.time()

    img = img[start_y:, :]
    # cv2.imshow("", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    all_boxes = []
    for template_path in templates:
        template = cv2.imread(template_path)
        h, w = template.shape[:2]

        res = cv2.matchTemplate(img, template, cv2.TM_CCORR_NORMED)
        loc = np.where(res >= threshold)

        for pt in zip(*loc[::-1]):
            # корректируем координаты относительно исходного изображения
            all_boxes.append([pt[0], pt[1] + start_y , pt[0] + w, pt[1] + h + start_y])

    nms_overlap_thresh = 0.1
    ban_boxes = non_max_suppression(all_boxes, nms_overlap_thresh)
    ban_boxes = ban_boxes.tolist()

    if debug:
        end = time.time()
        print(f"find_templates() -> {end - start:.2f} сек")

        for (x1, y1, x2, y2) in ban_boxes:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return ban_boxes


def put_text_with_background(img, text, position, font_face=cv2.FONT_HERSHEY_SIMPLEX,
                             font_scale=0.6, text_color=(255, 255, 255),
                             bg_color=(0, 0, 0), thickness=2, line_type=cv2.LINE_AA):

    x, y = position

    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(text, font_face, font_scale, thickness)

    # Calculate background rectangle coordinates
    bg_x1 = x
    bg_y1 = y - text_height - 5  # Add some padding
    bg_x2 = x + text_width
    bg_y2 = y + baseline

    # Draw background rectangle
    cv2.rectangle(img, (bg_x1, bg_y1), (bg_x2, bg_y2), bg_color, -1)  # -1 for filled rectangle

    # Draw text
    cv2.putText(img, text, (x, y), font_face, font_scale, text_color, thickness, line_type)

    return img

def get_bottom_section_h(img):
    h, w = img.shape[:2]
    if h > 710:
        bottom_section_h = 227
    else:
        bottom_section_h = 0.3156 * h

    return bottom_section_h
