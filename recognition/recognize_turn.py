"""
This module implements the identification of the current turn from a screenshot
"""
import cv2
import numpy as np

from recognition import recognition_utils

def get_distances(color, target_red, target_blue):
    distance_blue = np.linalg.norm(np.array(color) - np.array(target_red))
    distance_red = np.linalg.norm(np.array(color) - np.array(target_blue))

    return distance_blue, distance_red

def screenshot_to_turn(img, debug):
    h, w = img.shape[:2]
    bottom_section_start = h - round(recognition_utils.get_bottom_section_h(img))

    points = []
    for y in range(bottom_section_start - 10, bottom_section_start, 2):
        for x in range(0, w, w // 10):
            points.append((x, y))

    blue_turn_bgr = (180, 100, 4)
    red_turn_bgr = (38, 10, 160)

    blue_total_dist = 0
    red_total_dist = 0
    for p in points:
        x, y = p
        b, g, r = img[y, x]
        distance_blue, distance_red = get_distances((b, g, r), blue_turn_bgr, red_turn_bgr)
        blue_total_dist += distance_blue
        red_total_dist += distance_red

    if debug:
        print(points)
        for p in points:
            x, y = p
            b, g, r = img[y, x]
            distance_blue, distance_red = get_distances((b, g, r), blue_turn_bgr, red_turn_bgr)
            color = blue_turn_bgr if distance_blue < distance_red else red_turn_bgr

            size = 1

            top_left = (x - size, y - size)
            bottom_right = (x + size, y + size)

            cv2.rectangle(
                img,
                top_left,
                bottom_right,
                color=color,  # BGR!
                thickness=1
            )

        print(blue_total_dist)
        print(red_total_dist)

    return 1 if blue_total_dist < red_total_dist else -1