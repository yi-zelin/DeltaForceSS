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
import hashlib
import win32gui, win32con
import dxcam
from datetime import datetime, timedelta
from rapidfuzz import fuzz
from datetime import datetime
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedSeq

class IncorrectPageError(Exception):
    def __init__(self, message="未检测到特勤处建造界面"):
        self.message = message
        super().__init__(self.message)


with open('config.yaml', 'r', encoding='utf-8') as fin:
    config = yaml.load(fin, Loader=yaml.FullLoader)

with open('user_config.yaml', 'r', encoding='utf-8') as fin:
    user_config = yaml.load(fin, Loader=yaml.FullLoader)

SCALE_FACTOR = user_config['SCALE_FACTOR']
OUTPUT_DIR = './log'
TESSERACT_PATH = user_config['TESSERACT_PATH']

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
departments_coords = None
debug_mode = user_config['debug_mode']

wait_list = {
    # None -> skip this department
    # [dep, item_name]
        'tech': None,
        'work': None,
        'medical': None,
        'armor': None
    }

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
    
def update_wait_list():
    def find_match(dep):
        target_name = user_config[dep][0][0]
        for key, value in config['departments'][dep].items():
            for item_name in value:
                if item_name == target_name:
                    wait_list[dep] = [key, target_name]
                    return
        raise ValueError(f'Incorrect name: {target_name}')
    
    for dep in ['tech', 'work', 'medical', 'armor']:
        if not user_config[dep]:
            wait_list[dep] = None
            continue
        
        find_match(dep)
        
def write_user_config(department):
    # Configure YAML settings
    yaml = YAML()
    yaml.indent(mapping=2, sequence=4, offset=2)
    yaml.preserve_quotes = True
    yaml.width = 120
    
    # Load the existing config with comments
    with open('user_config.yaml', 'r', encoding='utf-8') as file:
        user_config = yaml.load(file)

    if not user_config.get(department):
        return

    first_item = user_config[department][0]
    print(first_item)
    _, quantity = first_item
    
    # Modify the quantity
    if quantity in (0, 1):
        user_config[department].pop(0)

    elif quantity > 1:
        first_item[1] -= 1
    
    # Post-processing to maintain perfect formatting
    for key in user_config:
        # Convert empty lists to None to prevent "[]" output
        if isinstance(user_config[key], list) and not user_config[key]:
            user_config[key] = None
        # Ensure flow style for all list items
        elif isinstance(user_config[key], CommentedSeq):
            for item in user_config[key]:
                if isinstance(item, list):
                    item.fa.set_flow_style()
    
    # Write back to file
    with open('user_config.yaml', 'w', encoding='utf-8') as file:
        yaml.dump(user_config, file)

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
            prev_y = y
    return cells

# Screenshot
def screenshot(type = 'binary', hint = 'placeholder', region = None):
    """
    region (x, y, w, h)
    """
    try:
        camera = dxcam.create()
        if region:
            x, y, w, h = region
            frame = camera.grab(region=(x, y, x+w, y+h))
        else:
            frame = camera.grab()
        
        if frame is not None:
            original_img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            if debug_mode:
                red_channel = original_img[:, :, 2]
                _, red_binary = cv2.threshold(red_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                combined_binary = cv2.bitwise_xor(binary, red_binary)
                save_image([(original_img, f'{hint}_original', None),
                            (gray, f'{hint}_gray', None),
                            (binary, f'{hint}_binary', None),
                            (combined_binary, f'{hint}_combinedBinary', None)])
            
            if type == 'binary':
                return binary
            elif type == 'original':
                return original_img
            elif type == 'gray':
                return gray
            elif type == 'combined_binary':
                # remove coin icon in front of price
                red_channel = original_img[:, :, 2]
                _, red_binary = cv2.threshold(red_channel, 128, 255, cv2.THRESH_BINARY)
                combined_binary = cv2.bitwise_xor(binary, red_binary)
                return combined_binary
            else:
                raise ValueError(f'! Error: unsupported image type !')
        else:
            raise Exception(f'! Faild: screenshot !')
    finally:
        camera.stop()
        del camera
    

def cropImage(image, region):
    x, y, w, h = region
    cropped = image[y:y+h, x:x+w]

    if debug_mode:
        save_image([(cropped, 'cropped', region)])
    return cropped


# Debug
def show_image(image):
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def save_image(image_hint_region_tuples):
    '''
    (image, hint, region)
    '''
    timestamp = datetime.now().strftime("%H_%M_%S_%f")[:-3]  # hh:mm:ss:sss
    for image, hint, region in image_hint_region_tuples:
        if region:
            x, y, w, h = region
            cv2.imwrite(os.path.join(OUTPUT_DIR, f'{timestamp}_{hint}_{x},{y},{w},{h}.png'), image)
        else:
            cv2.imwrite(os.path.join(OUTPUT_DIR, f'{timestamp}_{hint}.png'), image)

def debug_visualize_lines(image, lines):
    # Create a copy of the image in RGB format
    image_with_lines = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)

    # Draw all lines on the image
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image_with_lines, (x1, y1), (x2, y2), (0, 255, 0), 1)  # green line thickness = 1 px
    
    # Generate unique hash from image data
    image_hash = hashlib.md5(image_with_lines.tobytes()).hexdigest()[:8]
    
    save_image([(image_with_lines, 'list_lines', None)])

# OCR
def OCR_remain_time(image):
    t_config = r'--psm 7 -c tessedit_char_whitelist=0123456789:'
    text = pytesseract.image_to_string(image, config=t_config)
    if text != '':
        return text
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
    print(f'✅ OCR price: {price}')
    return int(price)

def is_main_page(image):
    keywords = ['工作台', '技术中心', '制药台', '防具台']
    text = pytesseract.image_to_string(image, lang='chi_sim')
    text = ''.join(text.split())
    return len([kw for kw in keywords if kw in text]) >= 2

def best_match_item(str1, reference):
    max_score = 0
    best_match = None
    for item in reference:
        score = fuzz.ratio(str1, item)
        if score > max_score:
            max_score = score
            best_match = item
    return best_match, max_score


# Other function
def high_beep():
    winsound.Beep(2000, 500)

def low_beep():
    winsound.Beep(500, 500)
    
def alt_tab():
    keyboard.press('alt')
    time.sleep(0.13)
    keyboard.press('tab')
    time.sleep(0.1)
    keyboard.release('tab')
    time.sleep(0.02)
    keyboard.release('alt')

def buy_material():
    # purchase page
    x, y = departments_coords['price_point']
    w, h = departments_coords['price_size']
    price = None

    trial = 11
    for i in range(trial):
        image = screenshot('combined_binary', 'materialPricePage', (x, y, w, h))
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
    full_page = screenshot('binary', 'buyState')
    for index, value in enumerate(buy_points):
        x, y = value
        binary_img = cropImage(full_page, (x, y, w, h))
        white_pix = np.sum(binary_img == 255)
        white_ratio = white_pix / binary_img.size
        if white_ratio >= 0.05:
            return index
    return -1

def initalize_preparation():
    buy_state = find_buy_state()
    if buy_state == -1:
        return True

    # go to buy page
    click_position(departments_coords['buy_positions'][buy_state])
    time.sleep(3)
    price = buy_material()
    if price == -1:
        print(f'! Failed to buy material: maximum retry attempts reached')
    elif price == 0:
        print(f'! There is trade in only item')
        keyboard.send('esc')
        time.sleep(1)
    buy_state = find_buy_state()
    return buy_state == -1

def department_status(dash_img, dep_coords):
    '''
    return remain time in sec
    -1: done
    -2: not started
    '''
    def time_to_seconds(time_str):
        if time_str is None:
            return None
        try:
            hh, mm, ss = map(int, time_str.split(':'))
            return hh * 3600 + mm * 60 + ss
        except:
            return 30*60
    
    # check 设备处于空闲状态
    x, y = dep_coords['free']
    w, h = dep_coords['free_size']
    center_img = cropImage(dash_img, (x, y, w, h))
    if OCR_is_free(center_img):
        return -2

    # read remain time: success -> in progress, fail -> done
    x, y = dep_coords['timmer']
    w, h = dep_coords['timmer_size']
    timmer_img = cropImage(dash_img, (x, y, w, h))
    remain_time = OCR_remain_time(timmer_img)
    remain_time = time_to_seconds(remain_time)
    
    if remain_time is None:
        return -1
    return remain_time

def match_list_items():
    x, y = departments_coords['list_point']
    w, h = departments_coords['list_size']
    list_OCR_img = screenshot('gray', 'list', (x, y, w, h))
    list_edge_img = cv2.Canny(list_OCR_img, 25, 90)

    if list_edge_img is None or list_OCR_img is None:
        raise Exception('! Error: fail to capture list image !')

    list_size = departments_coords['list_size']
    item_size = departments_coords['item_size']
    minLength = list_size[0] * 0.8
    minArea = int(item_size[0] * item_size[1] * 0.8)

    # find split lines
    lines = cv2.HoughLinesP(list_edge_img, 1, np.pi / 180, threshold=100, minLineLength=minLength, maxLineGap=50)
    
    if debug_mode:
        debug_visualize_lines(list_edge_img, lines)
        
    if lines is None:
        raise Exception('! Error: no line was found !')
    horizontal_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(y2 - y1) < 5:  # k approx 0
            horizontal_lines.append(y1)
    if horizontal_lines is None:
        raise Exception('! Error: no horizontal line was found, check debug image !')

    # cut image
    cells = cut_by_lines(list_OCR_img, horizontal_lines, minArea)

    if cells:
        return cells
    raise Exception('! Error: cells is empty. Please check images !')

def dash_page():
    def get_remain_times(dash_img):
        status = []
        for dep, coords in departments_coords['dash_page'].items():
            status.append((dep, department_status(dash_img, coords)))
        return status
    
    if debug_mode:
        setup_output_directory(OUTPUT_DIR)
        
    dash_img = screenshot('binary', 'department_status')

    if not is_main_page(dash_img):
        raise IncorrectPageError()

    status = get_remain_times(dash_img)
    processing_department = []
        
    for dep, state in status:
        if state == -1:
            click_position(departments_coords['dash_page'][dep]['free'])
            time.sleep(3)
            keyboard.send('space')
            time.sleep(3)
            state = -2
        if state == -2 and wait_list[dep]:
            processing_department.append(dep)
            click_position(departments_coords['dash_page'][dep]['free'])
            time.sleep(3)
            list_page(dep)
            
    dash_img = screenshot('binary', 'department_status')
    status = get_remain_times(dash_img)
    remain_times = []
    print(f'# dash page info:')
    for dep, state in status:
        if state == -2:
            remain_times.append(0)
            print(f'\t{dep}\t not started')
        elif state == -1:
            remain_times.append(0)
            print(f'\t{dep}\t completed!')
        else:
            remain_times.append(state)
            print(f'\t{dep}\t working:\t{state // 3600}:{(state % 3600) // 60:02d}:{state % 60:02d}')
            if dep in processing_department:
                write_user_config(dep)
                
    return remain_times

def list_page(department):
    category, target = wait_list[department]
    list_page_operation(department, category, target)

def list_page_operation(department, category, target):
    reference = config['departments'][department][category]
    list_size = departments_coords['list_size']
    list_point = departments_coords['list_point']
    x = list_point[0] + int(list_size[0] / 2)
    y_offset = list_point[1]
    last_top_item = None

    for k in range(100):
        y1 = 2
        cells = match_list_items()
        # (image, y position)
        img, y1 = cells[0]
        factor = config['OCR_factors'][department]

        # same top item -> reached bottom
        current_top_item = OCR_item_name(img, department)
        score = 0
        # TODO: debug line
        print(f'!!! round: {k}, last: {last_top_item}, current: {current_top_item}')
        if last_top_item:
            score = fuzz.ratio(last_top_item, current_top_item)
        last_top_item = current_top_item
        if score >= factor:
            print(f'! {department}.{category}.{target} not found')
            keyboard.send('esc')
            time.sleep(1)
            return

        # loop list
        for i in cells:
            img, y = i
            text = OCR_item_name(img, department)
            match, score = best_match_item(text, reference)
            if debug_mode:
                print(f'{text}, {match}, {score}')
            if match is None:
                continue
            if match == target and score >= factor :
                craft((x, y_offset + y))
                return
        scroll_down_x4((x, y_offset + y1))
        
def print_restart_info(remain_time):
    restart_time = datetime.now() + timedelta(seconds=remain_time)
    
    time_str = f"Restart after: {remain_time // 3600}:{(remain_time % 3600) // 60:02d}:{remain_time % 60:02d}"
    restart_str = f"Restart at: {restart_time.strftime('%H:%M:%S')}"
    max_length = max(len(time_str), len(restart_str)) + 8
    border = '#' * max_length
    
    output = (
        f"\n{border}\n"
        f"#{time_str.center(max_length - 2)}#\n"
        f"#{restart_str.center(max_length - 2)}#\n"
        f"{border}\n\n"
    )
    
    print(output)
    

def main():
    global departments_coords
    departments_coords = {k: scale_coords(v) for k, v in config['departments_coords'].items()}
    background_mode = user_config['background_mode']
    hwnd = win32gui.FindWindow('UnrealWindow', '三角洲行动  ')
    
    while True:
        try:
            high_beep()
            time.sleep(1)
            if background_mode:
                win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                time.sleep(3)  
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
            time.sleep(6)
            
            update_wait_list()
            remain_times = dash_page()
            
            time.sleep(3)
            
            remain_time = min(remain_times)
            remain_time += 1*60     # 1 min buffer
            
            if background_mode:
                alt_tab()
                    
            print_restart_info(remain_time)
            low_beep()
            time.sleep(remain_time)
        except IncorrectPageError as e:
            print(f'界面异常: {e}')
            input('回到特勤处制造界面后, 按 *回车* 键继续...')
        except Exception as e:
            input('程序异常, 按 *回车* 键退出')
            return
            

def list_OCR_test(department, categories):
    global departments_coords
    departments_coords = {k: scale_coords(v) for k, v in config['departments_coords'].items()}
    hwnd = win32gui.FindWindow('UnrealWindow', '三角洲行动  ')
    win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
    time.sleep(4)
    
    list_size = departments_coords['list_size']
    list_point = departments_coords['list_point']
    x = list_point[0] + int(list_size[0] / 2)
    y_offset = list_point[1]
    factor = config['OCR_factors'][department]

    for category in categories:
        reference = config['departments'][department][category]
        for target in reference:
            y1 = 2
            t = True
            print(f"#########################: {target}")
            while t:
                cells = match_list_items()
                # (image, y position)
                img, y1 = cells[0]
                
                # loop list
                for i in cells:
                    img, y = i
                    OCR_text = OCR_item_name(img, department)
                    match, score = best_match_item(OCR_text, reference)
                    if match is None:
                        continue
                    if match == target and score >= factor :
                        print(f'!!!! {OCR_text} match: {match} at: {score}')
                        t = False
                        pyautogui.scroll(5000)
                        pyautogui.sleep(0.5)
                        low_beep()
                        time.sleep(2)
                        break
                    else:
                        print(f'xxxx {OCR_text} match: {match} at: {score}')

                if t:
                    scroll_down_x4((x, y_offset + y1))
    high_beep()

if __name__ == "__main__":
    main()
    # list_OCR_test('tech', ['枪口'])