import cv2
import numpy as np
import os
import shutil
import pytesseract

# Tesseract OCR path
TESSERACT_PATH = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Define the output folder path
OUTPUT_DIR = './log'

def setup_output_directory(output_dir):
    # Create or clear the output directory.
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)  # Delete the folder and its contents
    os.makedirs(output_dir)  # Create the folder

def crop_image(image, top_ratio=0.37, bottom_ratio=0.85):
    # Crop the top and bottom of the image.
    height = image.shape[0]
    crop_top = int(height * top_ratio)  # Crop the top
    crop_bottom = int(height * bottom_ratio)  # Keep the bottom part
    return image[crop_top:crop_bottom, :]

def preprocess_image(image):
    # Convert the image to grayscale and perform edge detection.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return gray, edges

def detect_vertical_lines(edges, height, min_line_length_ratio=0.4):
    # Detect vertical lines using Hough Line Transform.
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
    vertical_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            line_length = abs(y2 - y1)  # Calculate line length
            if abs(x1 - x2) < 5 and line_length > min_line_length_ratio * height:  # Allow slight tilt and length greater than min_line_length_ratio
                vertical_lines.append(x1)
    return sorted(vertical_lines)  # Sort by x-coordinate

def crop_based_on_vertical_lines(image, vertical_lines, height, width, min_area_ratio=0.1):
    # Crop the image based on vertical lines and filter by area.
    cropped_images = []
    for i in range(len(vertical_lines) - 1):
        x1 = vertical_lines[i]
        x2 = vertical_lines[i + 1]
        
        # Check if the cropping area is valid
        if x1 >= x2 or x1 < 0 or x2 > width:
            print(f"Skipping invalid cropping area: x1={x1}, x2={x2}")
            continue
        
        cropped = image[:, x1:x2]  # Crop the image
        if cropped.size == 0:  # Check if the cropped image is empty
            print(f"Cropped image is empty: x1={x1}, x2={x2}")
            continue
        
        # Check if the cropped image area is greater than a threshold
        cropped_area = cropped.shape[0] * cropped.shape[1]  # Calculate the cropped image area
        if cropped_area < min_area_ratio * (height * width):  # Skip if the area is too small
            print(f"Cropped image area too small: x1={x1}, x2={x2}, area={cropped_area}")
            continue
        
        cropped_images.append(cropped)
    return cropped_images

def save_cropped_images(cropped_images, output_dir):
    # Save cropped images and perform OCR on each.
    for i, cropped in enumerate(cropped_images):
        # Convert to grayscale (keep only the blue channel)
        gray = cropped[:, :, 0]  # Blue channel
        
        # Binarization
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
        
        # Save binary image
        cv2.imwrite(os.path.join(output_dir, f'binary_{i + 1}.png'), binary)
        
        # OCR recognition
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
        text = pytesseract.image_to_string(binary, lang='chi_sim')  # Use the Simplified Chinese language pack

        # Save the recognition result
        with open(os.path.join(output_dir, f'OCR_{i + 1}.txt'), 'w', encoding='utf-8') as f:
            f.write(text)

def main():
    # Setup output directory
    setup_output_directory(OUTPUT_DIR)

    # Read the image
    image = cv2.imread('./.test/test3.png')
    if image is None:
        raise ValueError("Image file not found or cannot be read!")
    height, width, _ = image.shape

    # Crop the image (top 37% and bottom 15%)
    image = crop_image(image, top_ratio=0.37, bottom_ratio=0.85)

    # Preprocess the image (grayscale and edge detection)
    gray, edges = preprocess_image(image)

    # Detect vertical lines
    vertical_lines = detect_vertical_lines(edges, height, min_line_length_ratio=0.4)

    # Crop the image based on vertical lines
    cropped_images = crop_based_on_vertical_lines(image, vertical_lines, height, width, min_area_ratio=0.1)

    # Save cropped images and perform OCR
    save_cropped_images(cropped_images, OUTPUT_DIR)

    print(f"Processing complete! Results saved to {OUTPUT_DIR}.")

if __name__ == "__main__":
    main()