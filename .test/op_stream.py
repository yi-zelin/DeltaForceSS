import cv2
import os

# 创建 output 文件夹（如果不存在）
output_dir = '.test/output/op_stream'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 读取图像
image = cv2.imread('./.test/test1.png')
cv2.imwrite(os.path.join(output_dir, '1_op_stream.png'), image)

# 灰度化
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite(os.path.join(output_dir, '2_op_stream.png'), gray)

# 或者使用中值滤波
denoised = cv2.medianBlur(gray, 5)
cv2.imwrite(os.path.join(output_dir, '3.1_op_stream.png'), denoised)

# 二值化
_, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

cv2.imwrite(os.path.join(output_dir, '3.2_op_stream.png'), binary)
print(f"图像已保存到 {output_dir} 文件夹！")