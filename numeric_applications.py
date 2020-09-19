
from download import download
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#data links
url1 = "https://github.com/MegDie/advanced_lm_introduction/blob/master/datasets/Donnees_Comptages_Velos_Totem_Albert_1er_verbose.csv"
url2 = "https://github.com/MegDie/advanced_lm_introduction/blob/master/datasets/crash_bikes.csv"
#data download
path_target1 = "datasets/Donnees_Comptages_Velos_Totem_Albert_1er_verbose.csv"
path_target2 = "datasets/crash_bikes.csv"
download(url1, path_target1, replace=False)
download(url2, path_target2, replace=False)

#clean data
df_bikes = pd.read_csv(path_target3 , na_values="", converters={'data': str, 'heure': str})
df_bikes.heure.unique()
df_bikes['heure']=df_bikes['heure'].replace('', np.nan)
df_bikes.dropna(subset=['heure'], inplace=True)

time_improved = pd.to_datetime(df_bikes['date'] +
                               ' ' + df_bikes['heure'] + ':00',
                               format='%Y-%m-%d %H:%M')
df_bikes['Time'] = time_improved
df_bikes.set_index('Time',inplace=True)
df_bikes['existence securite'] = df_bikes['existence securite'].replace(np.nan, "Inconnu")