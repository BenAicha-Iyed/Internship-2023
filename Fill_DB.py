import os
import cv2
import pandas as pd
from test import persons_in_frame

actors = os.listdir("Actors")
empty_db = {actor: 0 for actor in actors}
INF = 1e18


def statistics_in_episode(episode_path, name):
    episode_stat = empty_db.copy()
    episode_stat["episode_num"] = name
    cap = cv2.VideoCapture(episode_path)
    frame_id = 0
    last_frame_detecting = {actor: -INF for actor in os.listdir("Actors")}
    while cap.isOpened():
        ret, frame = cap.read()
        if frame_id % 200 == 0:
            episode_stat, last_frame_detecting = persons_in_frame(frame, episode_stat, last_frame_detecting, frame_id)
            if cv2.waitKey(1) == ord(" "):
                break
        frame_id += 1
        if not ret:
            break

    cap.release()
    cv2.destroyAllWindows()
    return episode_stat


def create_data_base(season_path, episodes=[]):
    season_name = season_path.split('\\')[-1]
    df_path = f"Results\\Datasets\\{season_name}.csv"
    try:
        df = pd.read_csv(df_path)
    except:
        df = pd.DataFrame(columns=["episode_num"] + actors)
    if len(episodes) == 0:
        episodes = os.listdir(season_path)
    for file_name in episodes:
        episode_name = file_name.split('.')[0]
        episode_path = os.path.join(season_path, file_name)
        row = statistics_in_episode(episode_path, episode_name)
        print(row)
        df = df.append(row, ignore_index=True)
        # df = pd.concat([df, pd.DataFrame(row)])

    df.to_csv(df_path, index=False)


def create_DB_with_reading_params_from_console():
    root_DB = "clean DB"
    saison = input(f'Enter your the Saison name from this list {os.listdir(root_DB)}:\n')
    saison_path = os.path.join(root_DB, saison)
    Df_path = f"Results\\Datasets\\{saison}.csv"
    Df = pd.read_csv(Df_path)
    episode_number = input(f"Enter the episode number: from 1 -> {len(os.listdir(saison_path))}: ")
    episode_name = f"episode_{episode_number}"
    episode_path = os.path.join(saison_path, f"{episode_name}.mp4")
    new_row = statistics_in_episode(episode_path, episode_name)
    print(new_row)
    Df = Df.append(new_row, ignore_index=True)
    Df.to_csv(Df_path)


# create_DB_with_reading_params_from_console()
# create_data_base("clean DB\\Saison2 (2006)")
create_data_base("D:\\second year internship\\Summer_Internship_2023\\Short_videos_to_test_project")
# create_DB_with_reading_params_from_console()
