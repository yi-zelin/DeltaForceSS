import cv2
import numpy as np
import os
import shutil
import pytesseract
import pyautogui
import yaml
import re
import time
from thefuzz import fuzz

with open('config.yaml', 'r', encoding='utf-8') as fin:
    config = yaml.load(fin, Loader=yaml.FullLoader)
    
SCALE_FACTOR = 2
OUTPUT_DIR = './log'

def setup_output_directory(output_dir):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

def scale_coords(coords):
    if isinstance(coords, (list, tuple)):
        return [int(x * SCALE_FACTOR) for x in coords]
    elif isinstance(coords, dict):
        return {k: scale_coords(v) for k, v in coords.items()}
    return coords

def full_screenshot():
    screenshot = np.array(pyautogui.screenshot())
    gray = screenshot[:, :, 0]  # Blue channel
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    
    setup_output_directory(OUTPUT_DIR)
    cv2.imwrite(os.path.join(OUTPUT_DIR, 'full_screenshot.png'), binary)
    cv2.imwrite(os.path.join(OUTPUT_DIR, 'full_screenshot_nm.png'), screenshot)

def pick_region(point, size):
    screenshot_path = os.path.join(OUTPUT_DIR, 'full_screenshot.png')
    if not os.path.exists(screenshot_path):
        raise FileNotFoundError(f"File not found: {screenshot_path}")

    screenshot = cv2.imread(screenshot_path, cv2.IMREAD_COLOR)
    if screenshot is None:
        raise ValueError("Failed to read!")

    x, y = point
    w, h = size
    cropped = screenshot[y:y+h, x:x+w]

    cv2.imwrite(os.path.join(OUTPUT_DIR, f'cropped_{x}_{y}_{w}_{h}.png'), cropped)

    return cropped

def show_image(image):
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def click_position(position):
    pyautogui.moveTo(position[0], position[1], duration=0.1)
    pyautogui.click()
    
def get_names_form_list(image):
    return;

def scroll_down_x4(position):
    pyautogui.moveTo(position[0], position[1], duration=0.1)
    for _ in range(4):
        pyautogui.scroll(-1)
        pyautogui.sleep(0.1)

def best_match_item(str1, department):
    max_score = 0
    best_match = None
    for item in department:
        score = fuzz.partial_ratio(str1, item)
        # print(f'{item}: {score}')
        if score > max_score:
            max_score = score
            best_match = item
    return best_match, max_score

def OCR_remain_time(image):
    t_config = r'--psm 7 -c tessedit_char_whitelist=0123456789:'
    text = pytesseract.image_to_string(image, config=t_config)
    time_pattern = r'\d{2}:\d{2}:\d{2}'
    match = re.search(time_pattern, text)
    if match:
        return match.group()
    return None

def OCR_is_free(image):
    t_config = r'-l chi_sim --psm 7'
    text = pytesseract.image_to_string(image, config=t_config)
    _, match_score = best_match_item(text, ['设备处于空闲状态'])
    if match_score is None:
        return False
    return match_score > 50

''' 0 for in progress, 1 for done, 2 for not started '''
def department_status(dep_coords):
    center_img = pick_region(dep_coords['free'], dep_coords['free_size'])
    if OCR_is_free(center_img):
        return 2
    timmer_img = pick_region(dep_coords['timmer'], dep_coords['timmer_size'])
    remain_time = OCR_remain_time(timmer_img)
    if remain_time is None:
        return 1
    return 0

def main():
    # time.sleep(2)
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    config['departments_coords'] = {k: scale_coords(v) for k, v in config['departments_coords'].items()}
    # full_screenshot()
    for dep, coords in config['departments_coords']['main_page'].items():
        print(f'{dep}: {department_status(coords)}')

main()