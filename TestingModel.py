import os

import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import cv2
import numpy as np

from Main import FindOnlyPeopleInImage, InitGlobalVariables


def SelectOnePerson(persons):
    if len(persons) == 1:
        return persons[0]
    return None


def predict(image, tolerance=0.55):
    persons = FindOnlyPeopleInImage(image, tolerance=tolerance)
    return SelectOnePerson(list(persons.values()))


def PredictTestSet(folder_path="Test_Set"):
    db_name, db_path, reference_people, output_folder = InitGlobalVariables(folder_path, operation='test')
    y_pred, y_true = [], []
    for actor_name in reference_people:
        subdir_path = os.path.join(db_path, actor_name)
        print(f"Start predicting {actor_name}'s dataset ...")
        for image_name in os.listdir(subdir_path):
            image_path = os.path.join(subdir_path, image_name)
            image = cv2.imread(image_path)
            prediction = predict(image)
            if prediction is not None:
                y_pred.append(prediction)
                y_true.append(actor_name)
        print(f"{actor_name}'s dataset is predicted ")

    return np.array(y_true), np.array(y_pred)


Y_true, Y_pred = PredictTestSet()
encoder = LabelEncoder()
Y_pred = encoder.fit_transform(Y_pred)
Y_true = encoder.transform(Y_true)
display_labels = encoder.classes_
print("accuracy = ", accuracy_score(Y_true, Y_pred))
conf_matrix = confusion_matrix(Y_true, Y_pred, normalize=None)
conf_matrix = ConfusionMatrixDisplay(conf_matrix, display_labels=display_labels)
conf_matrix.plot(colorbar=False)
plt.xticks(rotation=-90)
plt.show()
