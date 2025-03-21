import cv2
import numpy as np
import os
import shutil
import pytesseract
import pyautogui
import yaml
import re
import time
import keyboard
import winsound
from rapidfuzz import fuzz

with open('config.yaml', 'r', encoding='utf-8') as fin:
    config = yaml.load(fin, Loader=yaml.FullLoader)
    
with open('user_config.yaml', 'r', encoding='utf-8') as fin:
    user_config = yaml.load(fin, Loader=yaml.FullLoader)

SCALE_FACTOR = user_config['SCALE_FACTOR']
OUTPUT_DIR = './log'
LIST_ITEMS_DIR = f'{OUTPUT_DIR}/list_items'
DASH_PAGE_DIR = f'{OUTPUT_DIR}/dash_page'
TESSERACT_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

is_running = False


# Setup
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


# Mouse
def click_position(position):
    pyautogui.moveTo(position[0], position[1], duration=0.3)
    pyautogui.click()

def scroll_down_x4(position):
    pyautogui.moveTo(position[0], position[1], duration=0.3)
    for _ in range(4):
        pyautogui.scroll(-1)
        pyautogui.sleep(0.1)
    time.sleep(1)

def craft(coordination):
    build_position = config['departments_coords']['build_position']
    click_position(coordination)
    time.sleep(1)
    click_position(build_position)
    time.sleep(3)
    keyboard.send('esc')
    time.sleep(1)


# Image
def adjust_gamma(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def pick_region(point, size, prefix='cropped', mother_file=f"{OUTPUT_DIR}/full_screenshot.png", dir=OUTPUT_DIR):
    if not os.path.exists(mother_file):
        raise FileNotFoundError(f"File not found: {mother_file}")

    screenshot = cv2.imread(mother_file, cv2.COLOR_BGR2GRAY)
    if screenshot is None:
        raise ValueError("Failed to read screenshot!")

    x, y = point
    w, h = size
    cropped = screenshot[y:y+h, x:x+w]

    cv2.imwrite(os.path.join(dir, f'{prefix}_{x}_{y}_{w}_{h}.png'), cropped)
    return cropped

def cut_by_lines(list_img, horizontal_lines, min_area, prefix='cell'):
    cells = []
    height, width = list_img.shape
    horizontal_lines.append(height)
    horizontal_lines = sorted(horizontal_lines)

    prev_y = 0
    for y in horizontal_lines:
        if y > prev_y:
            cell = list_img[prev_y:y, 0:width]
            # area = # black pixel
            area = np.sum(cell == 0)
            if area > min_area:
                # center y coord
                center_y = prev_y + (y - prev_y) // 2
                cells.append((cell, center_y))
                cv2.imwrite(os.path.join(LIST_ITEMS_DIR, f'{prefix}_{prev_y}_{y}.png'), cell)
            prev_y = y
    return cells


# Screenshot
def full_screenshot(dir):
    # pyautogui.moveTo(0, 0, duration=0.3)
    screenshot = np.array(pyautogui.screenshot())
    screenshot_bgr = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
    screenshot = adjust_gamma(screenshot_bgr, gamma=0.8)

    gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=1.7, beta=0)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(gray, 50, 150)

    setup_output_directory(dir)
    cv2.imwrite(os.path.join(dir, 'full_screenshot.png'), screenshot)
    cv2.imwrite(os.path.join(dir, 'full_screenshot_gray.png'), gray)
    cv2.imwrite(os.path.join(dir, 'full_screenshot_binary.png'), binary)
    cv2.imwrite(os.path.join(dir, 'full_screenshot_edges.png'), edges)


# Debug
def show_image(image):
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def debug_visualize_lines(image, lines, output_path):
    image_with_lines = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)

    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image_with_lines, (x1, y1), (x2, y2), (0, 255, 0), 1) # green line thickness = 1 px
    cv2.imwrite(os.path.join(LIST_ITEMS_DIR, f'debug_lines.png'), image_with_lines)
    
    
# OCR
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

def OCR_item_name(image, dep):
    OCR_config = config['OCR_configs'][dep]
    text = pytesseract.image_to_string(image, config=OCR_config)
    
    # manual improvement
    text = text.replace("番", "盔")
    return text

def best_match_item(str1, reference):
    max_score = 0
    best_match = None
    for item in reference:
        score = fuzz.partial_ratio(str1, item)
        # print(f'{item}: {score}')
        if score > max_score:
            max_score = score
            best_match = item
    return best_match, max_score


# Other function
def beep():
    winsound.Beep(1000, 500)

def department_status(dep_coords):
    '''
    0: not started
    1: in progress
    2: done
    '''
    center_img = pick_region(dep_coords['free'], dep_coords['free_size'], 'dash_page', f"{DASH_PAGE_DIR}/full_screenshot.png", DASH_PAGE_DIR)
    if OCR_is_free(center_img):
        return 0
    timmer_img = pick_region(dep_coords['timmer'], dep_coords['timmer_size'], 'dash_page', f"{DASH_PAGE_DIR}/full_screenshot.png", DASH_PAGE_DIR)
    remain_time = OCR_remain_time(timmer_img)
    if remain_time is None:
        return 2
    return 1

def match_list_items():
    related_var = config['departments_coords']
    full_screenshot(LIST_ITEMS_DIR)
    list_edge_img = pick_region(related_var['list_point'], related_var['list_size'], 'full_list', f"{LIST_ITEMS_DIR}/full_screenshot_edges.png", LIST_ITEMS_DIR)
    list_OCR_img = pick_region(related_var['list_point'], related_var['list_size'], 'full_list', f"{LIST_ITEMS_DIR}/full_screenshot_binary.png", LIST_ITEMS_DIR)
    return list_cell_detector(list_edge_img, list_OCR_img)

def list_cell_detector(list_edge_img, list_OCR_img):
    list_size = config['departments_coords']['list_size']
    item_size = config['departments_coords']['item_size']
    minLength = list_size[0] * 0.8
    minArea = int(item_size[0] * item_size[1] * 0.8)

    lines = cv2.HoughLinesP(list_edge_img, 1, np.pi / 180, threshold=100, minLineLength=minLength, maxLineGap=10)
    debug_visualize_lines(list_edge_img, lines, LIST_ITEMS_DIR)
    if lines is None:
        raise ValueError("Error: lines is empty. Please check images.")
    horizontal_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(y2 - y1) < 5:  # k approx 0
            horizontal_lines.append(y1)
    if horizontal_lines is None:
        raise ValueError("Error: horizontal lines is empty. Please check images.")

    # cut image
    cells = cut_by_lines(list_OCR_img, horizontal_lines, minArea)

    if cells:
        return cells
    raise ValueError("Error: cells is empty. Please check images.")

def dash_page():
    full_screenshot(DASH_PAGE_DIR)
    status = []
    for dep, coords in config['departments_coords']['dash_page'].items():
        status.append((dep, department_status(coords)))
    print(f'dash page: {status}')

    for dep, state in status:
        if state == 2:
            click_position(config['departments_coords']['dash_page'][dep]['free'])
            time.sleep(3)
            keyboard.send('space')
            time.sleep(1)
            state = 0
        if state == 0:
            click_position(config['departments_coords']['dash_page'][dep]['free'])
            time.sleep(3)
            list_page(dep)
            
def list_page(department):
    category, target = user_config[department]  
    list_page_operation(department, category, target)

def list_page_operation(department, category, target):
    reference = config['departments'][department][category]
    list_size = config['departments_coords']['list_size']
    list_point = config['departments_coords']['list_point']
    x = list_point[0] + int(list_size[0] / 2)
    y_offset = list_point[1]
    last_top_item = None
    
    for _ in range(100):
        y1 = 20
        cells = match_list_items()
        img, y1 = cells[0]
        current_top_item = OCR_item_name(img, department).strip()
        _, score = best_match_item(current_top_item, [last_top_item])
        last_top_item = current_top_item
        if score >= 85:
            print(f'{department}.{category}.{target} not found')
            keyboard.send('esc')
            time.sleep(1)
            return
        
        for i in cells:
            img, y = i
            # show_image(img)
            text = OCR_item_name(img, department)
            match, score = best_match_item(text, reference)
            match = match.strip()
            print(f'{text} matches: {match}, {score}')
            if score > 70 and match == target:
                craft((x, y_offset + y))
                return
        scroll_down_x4((x, y_offset + y1))

def main():
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
    config['departments_coords'] = {k: scale_coords(v) for k, v in config['departments_coords'].items()}
    running = True
    while running:
        if keyboard.is_pressed('f6'):
            running = False
            print('stop')
        if keyboard.is_pressed('f7'):
            time.sleep(2)
            beep()
            dash_page()
            beep()
        time.sleep(0.1)

if __name__ == "__main__":
    main()