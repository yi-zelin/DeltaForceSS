import cv2
import numpy as np
import os
import shutil

# Define the output folder path
output_dir = './log'

# If the folder exists, clear it
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)  # Delete the folder and its contents

# Create the folder
os.makedirs(output_dir)

# Read the image
image = cv2.imread('./.test/test1.png')
if image is None:
    raise ValueError("Image file not found or cannot be read!")
height, width, _ = image.shape

# Crop the image
crop_top = int(height * 0.37)  # Crop the top
crop_bottom = int(height * 0.85)  # Keep the bottom part
image = image[crop_top:crop_bottom, :]

# 1. Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 2. Edge detection
edges = cv2.Canny(gray, 50, 150)

# 3. Detect lines
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

# 4. Filter vertical lines (length greater than 40% of the image height)
vertical_lines = []
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        line_length = abs(y2 - y1)  # Calculate line length
        if abs(x1 - x2) < 5 and line_length > 0.4 * height:  # Allow slight tilt and length greater than 40% of the image height
            vertical_lines.append(x1)

# 5. Crop the image based on vertical lines
vertical_lines = sorted(vertical_lines)  # Sort by x-coordinate
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
    
    # Check if the cropped image area is greater than a threshold (e.g., 10% of the total image area)
    cropped_area = cropped.shape[0] * cropped.shape[1]  # Calculate the cropped image area
    if cropped_area < 0.1 * (height * width):  # Skip if the area is smaller than 10% of the total image area
        print(f"Cropped image area too small: x1={x1}, x2={x2}, area={cropped_area}")
        continue
    
    cropped_images.append(cropped)
    cv2.imwrite(os.path.join(output_dir, f'cropped_{i + 1}.png'), cropped)  # Save the cropped image

print(f"Cropped images have been saved to the {output_dir} folder!")
