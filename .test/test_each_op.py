import cv2
import os
import shutil
import numpy as np

# Create the output folder (if it doesn't exist)
output_dir = '.test/output/op_stream1'
# If the folder exists, clear it
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)  # Delete the folder and its contents

# Create the folder
os.makedirs(output_dir)

# Read the image
image = cv2.imread('./.test/test1.png')
cv2.imwrite(os.path.join(output_dir, '1_original.png'), image)  # Save the original image

# Split color channels
blue_channel = image[:, :, 0]  # Blue channel
green_channel = image[:, :, 1]  # Green channel
red_channel = image[:, :, 2]  # Red channel

# Save the blue channel image
cv2.imwrite(os.path.join(output_dir, 'blue_channel.png'), blue_channel)

# Save the green channel image
cv2.imwrite(os.path.join(output_dir, 'green_channel.png'), green_channel)

# Save the red channel image
cv2.imwrite(os.path.join(output_dir, 'red_channel.png'), red_channel)

print(f"Color channel images have been saved to the {output_dir} folder!")

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite(os.path.join(output_dir, '2_gray.png'), gray)  # Save the grayscale image

# Binarization
_, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
cv2.imwrite(os.path.join(output_dir, '3_binary.png'), binary)  # Save the binary image

# Denoising (using median blur)
denoised = cv2.medianBlur(gray, 5)
cv2.imwrite(os.path.join(output_dir, '4_denoised.png'), denoised)  # Save the denoised image

# Edge detection (using Canny algorithm)
edges = cv2.Canny(gray, 100, 200)
cv2.imwrite(os.path.join(output_dir, '5_edges.png'), edges)  # Save the edge detection result

# Image resizing
resized = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
cv2.imwrite(os.path.join(output_dir, '6_resized.png'), resized)  # Save the resized image

print(f"All images have been saved to the {output_dir} folder!")
