import cv2
import os


def minimum_size(folder_path="peoples"):
    INF = 3000
    height, width = INF, INF
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        img = cv2.imread(file_path)
        height = min(height, img.shape[0])
        width = min(width, img.shape[1])

    return height, width


h, w = minimum_size()


def resize_all_references_images(folder_path="peoples", size=(w, h)):
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        img = cv2.imread(file_path)
        os.remove(file_path)
        cv2.imwrite(file_path, cv2.resize(img, size))


resize_all_references_images()
