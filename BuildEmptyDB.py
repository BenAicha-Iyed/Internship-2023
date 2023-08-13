import pandas as pd
import os


df = pd.DataFrame(columns=['episode_num']+os.listdir("Actors"))
root_DB = "clean DB"
datasets_path = "Results\\Datasets"
while True:
    Df_name = input(f"Enter your dataset title from this list: {os.listdir(root_DB)}")
    if Df_name in os.listdir(root_DB):
        Df_name += ".csv"
        break
    print(Df_name, "not in database, \ntry again")

output_path = f"{datasets_path}\\{Df_name}"
df.to_csv(output_path, index=False)

