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
import win32gui, win32ui, win32con, win32api
from PIL import Image
from rapidfuzz import fuzz
from datetime import datetime

with open('config.yaml', 'r', encoding='utf-8') as fin:
    config = yaml.load(fin, Loader=yaml.FullLoader)

with open('user_config.yaml', 'r', encoding='utf-8') as fin:
    user_config = yaml.load(fin, Loader=yaml.FullLoader)

SCALE_FACTOR = user_config['SCALE_FACTOR']
MAIN_MONITOR = user_config['MAIN_MONITOR']
OUTPUT_DIR = './log'
LIST_ITEMS_DIR = f'{OUTPUT_DIR}/list_items'
DASH_PAGE_DIR = f'{OUTPUT_DIR}/dash_page'
BUY_DIR = f'{OUTPUT_DIR}/buy'
TESSERACT_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
departments_coords = None

# Setup
def setup_output_directory(output_dir):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

def scale_coords(coords):
    if isinstance(coords, (list, tuple)):
        if all(isinstance(item, (list, tuple)) for item in coords):
            return [scale_coords(item) for item in coords]
        else:
            return [int(x * SCALE_FACTOR) for x in coords]
    elif isinstance(coords, dict):
        return {k: scale_coords(v) for k, v in coords.items()}
    else:
        return coords


# Mouse
def click_position(position):
    pyautogui.moveTo(position[0], position[1], duration=0.3)
    pyautogui.click()

def scroll_down_x4(position):
    pyautogui.moveTo(position[0], position[1], duration=0.3)
    for _ in range(4):
        pyautogui.scroll(-120)
        pyautogui.sleep(0.1)
    time.sleep(1)

def craft(coordination):
    build_position = departments_coords['build_position']
    click_position(coordination)
    time.sleep(1)
    has_all_materials = initalize_preparation()
    if has_all_materials:
        click_position(build_position)
        time.sleep(3)
    keyboard.send('esc')
    time.sleep(1)


# Image
def adjust_gamma(image, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def cut_by_lines(list_img, horizontal_lines, min_area, prefix='cell'):
    '''
    return list of (image, y position) array
    '''
    cells = []
    height, width = list_img.shape
    horizontal_lines.append(height)
    horizontal_lines = sorted(horizontal_lines)

    prev_y = 0
    for y in horizontal_lines:
        if y > prev_y:
            cell = list_img[prev_y:y, 0:width]
            # area = # black pixel
            area = cell.size
            if area > min_area:
                # center y coord
                center_y = prev_y + (y - prev_y) // 2
                cells.append((cell, center_y))
                cv2.imwrite(os.path.join(LIST_ITEMS_DIR, f'{prefix}_{prev_y}_{y}.png'), cell)
            prev_y = y
    return cells

# Screenshot
def win32_screenshot():
    """Return NumPy format full screenshot in correct RGB format, with optional raw image display"""
    # Get screen dimensions
    width = win32api.GetSystemMetrics(0)
    height = win32api.GetSystemMetrics(1)

    # Capture screenshot
    hwnd = win32gui.GetDesktopWindow()
    hdc = win32gui.GetWindowDC(hwnd)
    mfc_dc = win32ui.CreateDCFromHandle(hdc)
    save_dc = mfc_dc.CreateCompatibleDC()
    save_bitmap = win32ui.CreateBitmap()
    save_bitmap.CreateCompatibleBitmap(mfc_dc, width, height)
    save_dc.SelectObject(save_bitmap)
    save_dc.BitBlt((0, 0), (width, height), mfc_dc, (0, 0), win32con.SRCCOPY)

    # Convert to NumPy array (RAW FORMAT - BGRA)
    bmp_str = save_bitmap.GetBitmapBits(True)
    img_raw = np.frombuffer(bmp_str, dtype=np.uint8)
    img_raw = img_raw.reshape((height, width, 4))  # BGRA format

    # Process colors: BGRA -> RGB
    img_rgb = img_raw[:, :, :3]

    # Cleanup resources
    win32gui.DeleteObject(save_bitmap.GetHandle())
    save_dc.DeleteDC()
    mfc_dc.DeleteDC()
    win32gui.ReleaseDC(hwnd, hdc)

    return img_rgb

def full_screenshot(output_dir, type='binary', binary_white_threshold=0.008, max_attemps=10):
    '''return valid full screenshot'''
    attemps = 0
    while attemps < max_attemps:
        screenshot = win32_screenshot()

        # save with unique name
        image_hash = hashlib.md5(screenshot.tobytes()).hexdigest()[:8]
        output_filename = f'screenshot_{image_hash}.png'
        t_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(t_path, screenshot)

        gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        edge = cv2.Canny(gray, 25, 90)

        # check if binary have engouh white pixels:
        total_pixels = gray.size
        white_pixels = np.sum(edge == 255)
        ratio = white_pixels / total_pixels
        if ratio >= binary_white_threshold:
            print(f'Success with white ratio: {ratio}')
            break
        attemps += 1
        print(f'Fail with white ratio: {ratio}')

    if attemps == max_attemps:
        raise Exception(f'❗ Max attempts reached. No valid screenshot captured ❗')

    # now screenshot, gray, binary are valid images
    cv2.imwrite(os.path.join(output_dir, f'full_screenshot.png'), screenshot)
    cv2.imwrite(os.path.join(output_dir, f'gray_screenshot.png'), gray)
    cv2.imwrite(os.path.join(output_dir, f'edge_screenshot.png'), edge)
    cv2.imwrite(os.path.join(output_dir, f'binary_screenshot.png'), binary)
    if type == 'binary':
        cv2.imwrite(os.path.join(output_dir, f'binary_full_screenshot.png'), binary)
        return binary
    elif type == 'gray':
        cv2.imwrite(os.path.join(output_dir, f'gray_full_screenshot.png'), gray)
        return gray
    elif type == 'original':
        return screenshot
    elif type == 'combined_binary':
        # remove coin icon in front of price
        red_channel = screenshot[:, :, 2]
        _, red_binary = cv2.threshold(red_channel, 128, 255, cv2.THRESH_BINARY)
        combined_binary = cv2.bitwise_xor(binary, red_binary)
        cv2.imwrite(os.path.join(output_dir, f'combined_binary_full_screenshot.png'), combined_binary)
        return combined_binary
    elif type == 'edge':
        edge = cv2.Canny(gray, 25, 90)
        cv2.imwrite(os.path.join(output_dir, f'edge_full_screenshot.png'), edge)
        return edge, binary
    else:
        raise ValueError(f'❗ {type} is not a valid input ❗')

def cropImage(image, output_dir, x, y, w, h):
    cropped = image[y:y+h, x:x+w]

    now = datetime.now()
    timestamp = now.strftime('%M_%S_%f')[:-3]
    filename = f'edge_full_screenshot_{timestamp}.png'
    filepath = os.path.join(output_dir, filename)
    cv2.imwrite(filepath, cropped)
    return cropped


# Debug
def show_image(image):
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

import hashlib
import os

def debug_visualize_lines(image, lines, output_path):
    # Create a copy of the image in RGB format
    image_with_lines = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)

    # Draw all lines on the image
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image_with_lines, (x1, y1), (x2, y2), (0, 255, 0), 1)  # green line thickness = 1 px
    
    # Generate unique hash from image data
    image_hash = hashlib.md5(image_with_lines.tobytes()).hexdigest()[:8]
    
    # Create output filename with hash
    output_filename = f'debug_lines_{image_hash}.png'
    full_output_path = os.path.join(output_path, output_filename)
    
    # Save the image
    cv2.imwrite(full_output_path, image_with_lines)
    
    return full_output_path  # Return the full path where image was saved

# OCR
def OCR_remain_time(image):
    if image is None:
        return None
    t_config = r'--psm 7 -c tessedit_char_whitelist=0123456789:'
    text = pytesseract.image_to_string(image, config=t_config)
    time_pattern = r'\d{2}:\d{2}:\d{2}'
    match = re.search(time_pattern, text)
    if match:
        return match.group()
    return None

def OCR_is_free(image):
    '''
    return match score > 50
    '''
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

def OCR_price(image):
    t_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist="0123456789,"'
    text = pytesseract.image_to_string(image, config=t_config)
    price = re.sub(r'[^\d]', '', text)
    if price == '':
        return None
    print(f'🪙 OCR price: {price} 🪙')
    return int(price)


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

def buy_material():
    # purchase page
    x, y = departments_coords['price_point']
    w, h = departments_coords['price_size']
    price = None

    trial = 11
    for i in range(trial):
        screenshot = full_screenshot(BUY_DIR, 'combined_binary', 0.004)
        image = cropImage(screenshot, BUY_DIR, x, y, w, h)
        price = OCR_price(image)
        if price is not None:
            if i == trial - 1:
                keyboard.send('esc')
                time.sleep(1)
                return -1
            # cleck price to buy
            click_position(departments_coords['price_position'])
            time.sleep(3)
        else:
            return price
    return -1

def find_buy_state():
    '''
    -1: not exist buy icon
    0:  three materials
    1:  four materials
    '''
    buy_points = departments_coords['buy_points']
    w, h = departments_coords['buy_size']
    screenshot = full_screenshot(BUY_DIR, 'binary', 0.01)
    for index, value in enumerate(buy_points):
        x, y = value
        binary_img = cropImage(screenshot, BUY_DIR, x, y, w, h)
        white_pix = np.sum(binary_img == 255)
        white_ratio = white_pix / binary_img.size
        print(f'buy position {index}: {white_ratio}')
        if white_ratio >= 0.05:
            return index
    return -1

def initalize_preparation():
    setup_output_directory(BUY_DIR)
    buy_state = find_buy_state()
    if buy_state == -1:
        return True

    # go to buy page
    click_position(departments_coords['buy_positions'][buy_state])
    time.sleep(3)
    price = buy_material()
    if price == -1:
        print(f'⚠️ Failed to buy material: maximum retry attempts reached ⚠️')
    elif price == 0:
        print(f'⚠️ There is trade in only item ⚠️')
        keyboard.send('esc')
        time.sleep(1)
    buy_state = find_buy_state()
    return buy_state == -1

def department_status(dep_coords):
    '''
    0: not started
    1: in progress
    2: done
    '''

    # check 设备处于空闲状态
    screenshot = full_screenshot(DASH_PAGE_DIR, 'binary', 0.025)
    x, y = dep_coords['free']
    w, h = dep_coords['free_size']
    center_img = cropImage(screenshot, DASH_PAGE_DIR, x, y, w, h)
    if OCR_is_free(center_img):
        return 0

    # read remain time: success -> in progress, fail -> done
    x, y = dep_coords['timmer']
    w, h = dep_coords['timmer_size']
    timmer_img = cropImage(screenshot, DASH_PAGE_DIR, x, y, w, h)
    remain_time = OCR_remain_time(timmer_img)
    if remain_time is None:
        return 2
    return 1

def match_list_items():
    gray = full_screenshot(LIST_ITEMS_DIR, 'gray', 0.01)
    edge = cv2.Canny(gray, 25, 90)
    x, y = departments_coords['list_point']
    w, h = departments_coords['list_size']
    list_edge_img = cropImage(edge, LIST_ITEMS_DIR, x, y, w, h)
    list_OCR_img = cropImage(gray, LIST_ITEMS_DIR, x, y, w, h)

    if list_edge_img is None or list_OCR_img is None:
        raise Exception('❗ Error: fail to capture list image ❗')

    list_size = departments_coords['list_size']
    item_size = departments_coords['item_size']
    minLength = list_size[0] * 0.8
    minArea = int(item_size[0] * item_size[1] * 0.8)

    # find split lines
    lines = cv2.HoughLinesP(list_edge_img, 1, np.pi / 180, threshold=100, minLineLength=minLength, maxLineGap=50)
    debug_visualize_lines(list_edge_img, lines, LIST_ITEMS_DIR)
    if lines is None:
        raise Exception('❗ Error: no line was found ❗')
    horizontal_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(y2 - y1) < 5:  # k approx 0
            horizontal_lines.append(y1)
    if horizontal_lines is None:
        raise Exception('❗ Error: no horizontal line was found, check debug image ❗')

    # cut image
    cells = cut_by_lines(list_OCR_img, horizontal_lines, minArea)

    if cells:
        return cells
    raise Exception('❗ Error: cells is empty. Please check images ❗')

def dash_page():
    setup_output_directory(DASH_PAGE_DIR)
    status = []

    for dep, coords in departments_coords['dash_page'].items():
        status.append((dep, department_status(coords)))
    print(f'dash page: {status}')

    for dep, state in status:
        if state == 2:
            click_position(departments_coords['dash_page'][dep]['free'])
            time.sleep(3)
            keyboard.send('space')
            time.sleep(3)
            state = 0
        if state == 0:
            click_position(departments_coords['dash_page'][dep]['free'])
            time.sleep(3)
            list_page(dep)

def list_page(department):
    category, target = user_config[department]
    list_page_operation(department, category, target)

def list_page_operation(department, category, target):
    reference = config['departments'][department][category]
    list_size = departments_coords['list_size']
    list_point = departments_coords['list_point']
    x = list_point[0] + int(list_size[0] / 2)
    y_offset = list_point[1]
    last_top_item = None
    setup_output_directory(LIST_ITEMS_DIR)

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
            if match is None:
                continue
            print(f'{text} matches: {match}, {score}')
            if score > 95 and match == target:
                craft((x, y_offset + y))
                return
        scroll_down_x4((x, y_offset + y1))

def main():
    global departments_coords
    departments_coords = {k: scale_coords(v) for k, v in config['departments_coords'].items()}
    while True:
        beep()
        time.sleep(3)
        dash_page()
        print('🎉 Finished!')
        beep()
        time.sleep(30*60)

def test_screenshot_dxcam():
    capture_one_screenshot()
    
def test_screenshot_win32():
    img = win32_screenshot()
    cv2.imshow("win32", img)
    cv2.waitKey(0)  # 按任意键关闭窗口

def test_list_page():
    global departments_coords
    departments_coords = {k: scale_coords(v) for k, v in config['departments_coords'].items()}
    setup_output_directory(LIST_ITEMS_DIR)
    time.sleep(2)
    match_list_items()

def test_buy_material():
    global departments_coords
    departments_coords = {k: scale_coords(v) for k, v in config['departments_coords'].items()}
    setup_output_directory(LIST_ITEMS_DIR)
    time.sleep(2)
    find_buy_state()
    

def capture_one_screenshot(save_path=None, region=None):
    """
    用dxcam截取单张屏幕图片并显示
    :param save_path: 可选保存路径（如 "screenshot.png"）
    :param region: 可选截图区域 (left, top, width, height)
    """
    import dxcam
    # 初始化dxcam（默认主显示器）
    camera = dxcam.create()
    
    # 单次截图
    if region:
        left, top, width, height = region
        frame = camera.grab(region=(left, top, left+width, top+height))
    else:
        frame = camera.grab()  # 全屏截图
    
    if frame is not None:
        # 转换颜色格式（dxcam返回RGB，OpenCV需要BGR）
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # 显示截图
        cv2.imshow("dxcam", frame_bgr)
        cv2.waitKey(0)  # 按任意键关闭窗口
        
        # 可选保存
        if save_path:
            cv2.imwrite(save_path, frame_bgr)
            print(f"截图已保存到: {save_path}")
    else:
        print("截图失败！")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    # main()
    # test_list_page()
    # test_buy_material()
    time.sleep(1)
    while True:
        test_screenshot_dxcam()
        time.sleep(1)
        test_screenshot_win32()
        time.sleep(1)