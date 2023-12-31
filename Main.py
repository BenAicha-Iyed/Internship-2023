import cv2
import os
from uuid import uuid1
import face_recognition as fr
import numpy as np
from numpy.linalg import norm
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import pandas as pd


def InitGlobalVariables(root_db="Data_Base", operation='train'):
    list_db = os.listdir(root_db)
    if len(list_db) == 1:
        DB_name = list_db[0]
    else:
        DB_name = input(
            f"Which database do you want to {operation} on ?\n"
            f"Select one from this list {list_db}"
        )
    DB_path = f"{root_db}\\{DB_name}"
    persons_in_database = os.listdir(DB_path)
    output = "Results"
    return DB_name, DB_path, persons_in_database, output


db_name, db_path, reference_people, output_folder = InitGlobalVariables()


def FileName(file_path=""):
    return file_path.split("\\")[-1]


def AddImageToDataBase(waiting_time=1000):
    name = input(
        f"{db_name} = {reference_people}\n"
        f"Enter the new_person's name that you want to add to your DataBase: "
    )
    if name not in reference_people:
        os.mkdir(f"{db_path}\\{name}")
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        key = RenderImage(frame, waiting_time)
        if key == ord("s"):
            cv2.imwrite(f"{db_path}\\{name}\\{uuid1()}.jpg", frame)
        if key == ord(" "):
            break

    cap.release()
    cv2.destroyAllWindows()


def EncodeFaces(folder_path=db_path, load=False):
    codes_path = f'people_encoded\\{db_name}\\people_codes.npy'
    names_path = f'people_encoded\\{db_name}\\reference_to_code.npy'
    try:
        codes = np.load(codes_path)
        names = np.load(names_path)
    except:
        codes = []
        names = []
        for person_name in reference_people:
            subdir_person_path = os.path.join(folder_path, person_name)
            for img_name in os.listdir(subdir_person_path):
                img_path = os.path.join(subdir_person_path, img_name)
                img = fr.load_image_file(img_path)
                img_encoded = fr.face_encodings(img)
                try:
                    codes.append(img_encoded[0])
                    names.append(person_name)
                except:
                    pass
        codes, names = np.array(codes), np.array(names)
        np.save(f'{codes_path}', codes)
        np.save(f'{names_path}', names)
    return codes, names


people_encoding = EncodeFaces()


def CreateFrame(target, location, label):
    Top, Right, Bottom, Left = location
    if label == "Unknown person":
        color = (0, 0, 255)
    else:
        color = (255, 0, 0)

    cv2.rectangle(target, (Left, Top), (Right, Bottom), color, 2)
    cv2.rectangle(target, (Left, Bottom + 20), (Right, Bottom), color, cv2.FILLED)
    cv2.putText(target, label, (Left + 3, Bottom + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


def RenderImage(image, waiting_time=1, image_title='Results'):
    cv2.imshow(image_title, image)
    key = cv2.waitKey(waiting_time)
    return key


# "MostCorrespondingPerson()" is a function that determines the person most corresponding
# to the person we want to identify.It takes as parameters a dictionary whose value corresponds to the distance
# between the key and the person who appears in the designated image, and a set of people who are already identified
# in this image. It returns "Unknown person" if all keys are already detected before in this image.
def MostCorrespondingPersonTo(distances={}):
    min_distance = min(distances.values())
    for person_name in distances:
        if distances[person_name] == min_distance:
            return person_name
    return "Unknown person"


# FindPeopleInImage is a function that returns a dictionary contains:
# key: location,
# value: dict {
#       key: predicted_person,
#       value: distance between predicted_person and the real person in the location
# }
def FindAllPeopleCanBeInImage(image, tolerance=0.525, INF=1e18):
    face_locations = fr.face_locations(image)
    image_encoded = fr.face_encodings(image, known_face_locations=face_locations)
    people_can_be_corresponding_in = {}
    for face_number, location in enumerate(face_locations):
        people_can_be_corresponding_in[location] = {}
        for encoded_face, person_name in zip(people_encoding[0], people_encoding[1]):
            distance = norm(encoded_face - image_encoded[face_number])
            if distance <= tolerance:
                people_can_be_corresponding_in[location][person_name] = distance
            else:
                people_can_be_corresponding_in[location]["Unknown person"] = INF
    return people_can_be_corresponding_in


def FindOnlyPeopleInImage(image, tolerance=0.525):
    people_can_be_corresponding_in = FindAllPeopleCanBeInImage(image, tolerance=tolerance)
    ans = {key: MostCorrespondingPersonTo(value) for (key, value) in people_can_be_corresponding_in.items()}
    return ans


def ShowPeopleInImage(image, labels=None, waiting_time=1, image_title=''):
    if labels is None:
        labels = FindOnlyPeopleInImage(image)
    for (loc, label) in labels.items():
        CreateFrame(image, loc, label)
    return RenderImage(image, waiting_time, image_title)


def RealTime(video_path, waiting_time=1):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if cv2.waitKey(waiting_time) == ord(" ") or not ret:
            break
        key = ShowPeopleInImage(frame)
        if key == ord(' '):
            break
    cap.release()
    cv2.destroyAllWindows()


def LoadingImage(waiting_time=8000):
    Tk().withdraw()
    load_image = askopenfilename()
    target_image = fr.load_image_file(load_image)
    rgb = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
    key = ShowPeopleInImage(rgb, waiting_time)
    if key == ord(' '):
        cv2.destroyAllWindows()
    if key == ord('s'):
        cv2.imwrite(f"{output_folder}\\result_1.jpg", rgb)


def IsNewScene(current_frame_id, last_frame_id, length_scene=8000):
    return current_frame_id - last_frame_id >= length_scene


def CreateLabels(person_in, stats):
    for location, person in person_in.items():
        if person != "Unknown person":
            person_in[location] += f" {stats[person]}"
    return person_in


def UpdateStats(frame, frame_id, stats, last_frame_detecting_):
    people_in_frame = FindOnlyPeopleInImage(frame)
    for location, person in people_in_frame.items():
        if person != "Unknown person":
            if IsNewScene(frame_id, last_frame_detecting_[person]):
                stats[person] += 1
                last_frame_detecting_[person] = frame_id
    return stats, last_frame_detecting_, people_in_frame


def InitStatisticsParameters(episode_path, INF=1e18):
    episode_name = FileName(episode_path)
    episode_name = episode_name[:len(episode_name) - 4]
    stats = {actor: 0 for actor in reference_people}
    stats["episode_num"] = episode_name
    last_frame_detecting_ = {actor: -INF for actor in reference_people}
    cap = cv2.VideoCapture(episode_path)
    frame_id = 0
    return cap, frame_id, stats, last_frame_detecting_


def CalculateStatsByEpisode(episode_path="", output_df=None, step=200, waiting_time=1):
    cap, frame_id, stats, last_frame_detecting_ = InitStatisticsParameters(episode_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id % step == 0:
            stats, last_frame_detecting_, person_in_location = UpdateStats(
                frame, frame_id, stats, last_frame_detecting_
            )
            labels = CreateLabels(person_in_location, stats)
            key = ShowPeopleInImage(frame, labels, image_title=stats["episode_num"], waiting_time=waiting_time)
            if key == ord('0'):
                break
            if key == ord(' '):
                cv2.destroyAllWindows()
                return output_df, True
        frame_id += 1
    cap.release()
    cv2.destroyAllWindows()
    new_row = pd.DataFrame(stats, index=[len(output_df)])
    output_df = pd.concat([output_df, new_row])
    return output_df, False


def CalculateStatsBySeason(season_path="", episodes=[]):
    season_name = FileName(season_path)
    if len(episodes) == 0:
        episodes = os.listdir(season_path)
    df_path = f"{output_folder}\\{season_name}.csv"
    try:
        season_df = pd.read_csv(df_path)
    except:
        season_df = pd.DataFrame(columns=["episode_num"] + reference_people)
    for episode in episodes:
        episode_path = f"{season_path}\\{episode}"
        season_df, stop = CalculateStatsByEpisode(episode_path, season_df)
        if stop:
            return
    season_df.to_csv(df_path, index=False)

# RealTime(0)
# AddImageToDataBase()
# LoadingImage()
# CalculateStatsBySeason("D:\\second year internship\\Summer_Internship_2023\\clean DB\\Season_2_(2006)")
# videos_db = "D:\\second year internship\\Summer_Internship_2023\\clean DB\\Season_2_(2006)"
# for season_name in os.listdir(videos_db):
#     season_path = f"{videos_db}\\{season_name}"
#     CalculateStatsBySeason(season_path)
