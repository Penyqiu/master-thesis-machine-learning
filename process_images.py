import cv2
import os

def resize_image_to_80x80(image_path, output_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    resized_img = cv2.resize(img, (80, 80))
    cv2.imwrite(output_path, resized_img)

def process_images_from_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            resize_image_to_80x80(input_path, output_path)

input_folder_open = 'train/Open_Eyes'
input_folder_closed = 'train/Closed_Eyes'
output_folder_open = 'train/output/open_eyes'
output_folder_closed = 'train/output/closed_eyes'

process_images_from_folder(input_folder_open, output_folder_open)
process_images_from_folder(input_folder_closed, output_folder_closed)