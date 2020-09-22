from download import download
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#data links
url1 = "https://github.com/MegDie/advanced_lm_introduction/blob/master/datasets/Donnees_comptage.csv"
url2 = "https://github.com/MegDie/advanced_lm_introduction/blob/master/datasets/crash_bikes.csv"

#data download
path_target1 = "datasets/datasets/Donnees_comptage.csv"
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


#security device histogram
plt.figure()
df_bikes.groupby('existence securite').size().plot(kind='bar')
plt.xlabel("Security device")
plt.ylabel("Death's number")
plt.title("Breakdown of deaths by security's type")
plt.savefig('security_devices.pdf', bbox_inches="tight")


#gravity of the accidents
plt.figure()
df_bikes.groupby('gravite accident').size().plot(kind='bar')
plt.xlabel("Gravity of accidents")
plt.ylabel("Number of accidents")
plt.title("Breakdown of accidents by severity")
plt.savefig('accidents_gravity.pdf', bbox_inches="tight")


#daily profile of the number of accidents per month
import calendar
df_bikes['month'] = df_bikes.index.month
df_bikes['month'] = df_bikes['month'].apply(lambda x: calendar.month_abbr[x])
df_bikes.head()

sns.set_palette("Set2", n_colors=12)

df_bikes_month = df_bikes.groupby(['month', df_bikes.index.hour])[
    'age'].count().unstack(level=0)

fig, axes = plt.subplots(1, 1, figsize=(7, 7), sharex=True)

df_bikes_month.plot(ax=axes)
axes.set_ylabel("Number of accidents")
axes.set_xlabel("Hour")
axes.set_title(
    "Daily profile of the number of accidents per month")
axes.set_xticks(np.arange(0, 24))
axes.set_xticklabels(np.arange(0, 24), rotation=45)
axes.legend(labels=calendar.month_name[1:], loc='lower left', bbox_to_anchor=(1, 0.1))
plt.savefig('Daily_profile.pdf', bbox_inches="tight")
plt.tight_layout()


#gender
sns.set_palette("Set2", n_colors=12)
plt.figure()
df_bikes.groupby('sexe').size().plot(kind='bar')
plt.xlabel("Gender")
plt.ylabel("Number of accidents")
plt.title("Accidents per gender")
plt.savefig('Gender.pdf', bbox_inches="tight")

#death's gender
df_deads = df_bikes[df_bikes['gravite accident']=='3 - Tué']
df_deads.groupby('sexe').size().plot(kind='bar')
plt.xlabel("Gender")
plt.ylabel("Number of deaths")
plt.title("Deaths per gender")
plt.savefig('Gender_deaths.pdf', bbox_inches="tight")


#convert gravity into quantitative variable
df_bikes['grave_quanti'] = df_bikes['gravite accident']
df_bikes['grave_quanti'] = df_bikes['grave_quanti'].replace("0 - Indemne", 0)
df_bikes['grave_quanti'] = df_bikes['grave_quanti'].replace("1 - Blessé léger", 1)
df_bikes['grave_quanti'] = df_bikes['grave_quanti'].replace("2 - Blessé hospitalisé", 2)
df_bikes['grave_quanti'] = df_bikes['grave_quanti'].replace("3 - Tué", 3)
df_bikes['grave_quanti']


#Choice of the variable of interest: accident gravity, sexe, motif deplacement
#security exitence, type collision, type route
interest = ['gravite accident', 'grave_quanti', 'sexe', 
            'type collision', 'type route', 'existence securite', 'motif deplacement']
df_bikes_ols = df_bikes[interest]
df_bikes_ols = df_bikes_ols.dropna(how = 'any') #drop all Nan values
df_bikes_ols.rename(columns = {'gravite accident': 'grav_acc', 
                                  'type collision': 'collision_type', 'type route': 'road_type', 
                                  'existence securite': 'security', 
                                  'motif deplacement': 'exit_pattern'}, inplace=True)

#sampling
df_test = df_bikes_ols.iloc[0:1000]
df_sample = df_bikes_ols.iloc[1000:51947]

#Keep the most significant variables for the model
results = smf.ols('grave_quanti ~ collision_type + security + exit_pattern', data=df_sample).fit()

#predict on test data the gravity of the accident
Yhat = results.predict(df_test)
#transformation of the prediction to fit the variable
Yhat_proc = np.arange(1000)
for i in range(1000):
    if Yhat[i] <= 0.5:
        Yhat_proc[i] = 0
    if Yhat[i] <= 1.5 and Yhat[i] > 0.5:
        Yhat_proc[i] = 1
    if Yhat[i] <= 2.5 and Yhat[i] > 1.5:
        Yhat_proc[i] = 2
    if Yhat[i] > 2.5:
        Yhat_proc[i] = 3

#percentage of errors
res = df_test['grave_quanti'] == Yhat_proc
n=0
for i in range(1000):
    if res[i]==False:
        n=n+1
        
        








