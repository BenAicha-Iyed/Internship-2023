import os
import cv2


def labeling_with_name(name, directory_root):
    files = os.listdir(directory_root)
    for i, file in enumerate(files):
        file_path = os.path.join(directory_root, file)
        file_name = f"{name}_{i}.jpg"
        output_path = os.path.join("peoples", file_name)
        cv2.imwrite(output_path, cv2.imread(file_path))
        # print(type(file_name))


dir_root = "Actors"
for actor in os.listdir(dir_root):
    labeling_with_name(actor, os.path.join(dir_root, actor))
