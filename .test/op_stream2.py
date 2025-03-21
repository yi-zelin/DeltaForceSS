import cv2
import os
import shutil
import pytesseract

# Define the output folder path
output_dir = '.test/output/op_stream2'

# If the folder exists, clear it
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)  # Delete the folder and its contents

# Create the folder
os.makedirs(output_dir)

print(f"Folder {output_dir} has been cleared and recreated!")

# Read the image
image = cv2.imread('./.test/test4.png')
cv2.imwrite(os.path.join(output_dir, '1_op_stream.png'), image)

# Convert to grayscale (keep only the blue channel)
gray = image[:, :, 0]  # Blue channel
cv2.imwrite(os.path.join(output_dir, '2_op_stream.png'), gray)

# Binarization
_, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
cv2.imwrite(os.path.join(output_dir, '3_op_stream.png'), binary)

# Crop the image
height, width = binary.shape
crop_top = int(height * 0.37)  # Crop the top part
crop_bottom = int(height * 0.85)  # Keep the bottom part
crop_left = int(width * 0.06)  # Crop the left part
crop_right = int(width * 0.94)  # Keep the right part
cropped = binary[crop_top:crop_bottom, crop_left:crop_right]
cv2.imwrite(os.path.join(output_dir, '4_cropped.png'), cropped)

# 7. OCR recognition
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
text = pytesseract.image_to_string(cropped, lang='chi_sim')  # Use the Simplified Chinese language pack

# Save the recognition result to OCR.txt
with open(os.path.join(output_dir, 'OCR.txt'), 'w', encoding='utf-8') as f:
    f.write(text)

print(f"Recognition result has been saved to {os.path.join(output_dir, 'OCR.txt')}!")

print(f"Images have been saved to the {output_dir} folder!")