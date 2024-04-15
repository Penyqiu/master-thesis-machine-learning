import cv2
import os
import shutil

main_directory = '/home/kacper-penczynski/Pulpit/magisterka/master-thesis-machine-learning/uta-rldd/train'

subdirectories = ['awake', 'drowsy', 'low vigilant']

resized_directory = '/home/kacper-penczynski/Pulpit/magisterka/master-thesis-machine-learning/resized'
os.makedirs(resized_directory, exist_ok=True)

for subdir in subdirectories:
    subdir_path = os.path.join(main_directory, subdir)
    resized_subdir_path = os.path.join(resized_directory, subdir)
    if os.path.exists(subdir_path):
        os.makedirs(resized_subdir_path, exist_ok=True)
        for filename in os.listdir(subdir_path):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                image = cv2.imread(os.path.join(subdir_path, filename))
                resized_image = cv2.resize(image, (224, 224))
                cv2.imwrite(os.path.join(resized_subdir_path, filename), resized_image)
    else:
        print("Podkatalog '{}' nie istnieje.".format(subdir))

print("Zmiana rozmiaru obrazów zakończona.")
