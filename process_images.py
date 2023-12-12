import cv2
import os

def process_images_from_folder(folder_name, output_folder, size=(80, 80)):
    for filename in os.listdir(folder_name):
        if filename.endswith(".png"):
            img_path = os.path.join(folder_name, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            resized_img = cv2.resize(img, size)

            cv2.imwrite(os.path.join(output_folder, filename), resized_img)

input_folder_open = 'C:\\Users\\Razer\\Desktop\\magisterka\\master-thesis-machine-learning\\train\\Open_Eyes'
input_folder_closed = 'C:\\Users\\Razer\\Desktop\\magisterka\\master-thesis-machine-learning\\train\\Closed_Eyes'
output_folder_open = 'C:\\Users\\Razer\\Desktop\\magisterka\\master-thesis-machine-learning\\train\\output\\open_eyes'
output_folder_closed = 'C:\\Users\\Razer\\Desktop\\magisterka\\master-thesis-machine-learning\\train\\output\\closed_eyes'

os.makedirs(output_folder_open, exist_ok=True)
os.makedirs(output_folder_closed, exist_ok=True)

process_images_from_folder(input_folder_open, output_folder_open)
process_images_from_folder(input_folder_closed, output_folder_closed)