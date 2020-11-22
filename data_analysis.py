from download import download
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import t
import calendar
import patsy

# Data links
url1 = "https://github.com/MegDie/advanced_lm_introduction/blob/master/datasets/Donnees_comptage.ods"
url2 = "https://github.com/MegDie/advanced_lm_introduction/blob/master/datasets/crash_bikes.csv"

# Data download
path_target1 = "datasets/datasets/Donnees_comptage.ods"
path_target2 = "datasets/crash_bikes.csv"
download(url1, path_target1, replace=False)
download(url2, path_target2, replace=False)

# Clean data
df_bikes = pd.read_csv(path_target2, na_values="", converters={'data': str, 'heure': str})
df_bikes.heure.unique()
df_bikes['heure'] = df_bikes['heure'].replace('', np.nan)
df_bikes.dropna(subset=['heure'], inplace=True)

time_improved = pd.to_datetime(df_bikes['date'] +
                               ' ' + df_bikes['heure'] + ':00',
                               format='%Y-%m-%d %H:%M')
df_bikes['Time'] = time_improved
df_bikes.set_index('Time', inplace=True)
df_bikes['existence securite'] = df_bikes['existence securite'].replace(np.nan, "Inconnu")
del df_bikes['identifiant vehicule']
del df_bikes['type autres vehicules']
del df_bikes['circulation']
del df_bikes['nb voies']
del df_bikes['profil long route']
del df_bikes['trace plan route']
del df_bikes['largeur TPC']
del df_bikes['situation']
del df_bikes['usage securite']
del df_bikes['obstacle fixe heurte']
del df_bikes['manoeuvre autres vehicules']
del df_bikes['categorie usager']
del df_bikes['age']


# Security device histogram
plt.figure()
df_bikes.groupby('existence securite').size().plot(kind='bar')
plt.xlabel("Security device")
plt.ylabel("Number of accidents")
plt.title("Breakdown of accidents by security's type")
plt.savefig('security_devices.pdf', bbox_inches="tight")

# Gender
sns.set_palette("Set2", n_colors=12)
plt.figure()
df_bikes.groupby('sexe').size().plot(kind='bar')
plt.xlabel("Gender")
plt.ylabel("Number of accidents")
plt.title("Accidents per gender")
plt.savefig('Gender.pdf', bbox_inches="tight")

# Death and gender
df_deads = df_bikes[df_bikes['gravite accident']=='3 - Tué']
df_deads.groupby('sexe').size().plot(kind='bar')
plt.xlabel("Gender")
plt.ylabel("Number of deaths")
plt.title("Deaths per gender")
plt.savefig('Gender_deaths.pdf', bbox_inches="tight")

# We will need this later
df_bikes['month'] = df_bikes.index.month
df_bikes['hour'] = df_bikes.index.hour
df_bikes['year'] = df_bikes.index.year

# Convert gravity and sex into quantitative variable
df_bikes['grave_quanti'] = df_bikes['gravite accident']
df_bikes['grave_quanti'] = df_bikes['grave_quanti'].replace("0 - Indemne", 0)
df_bikes['grave_quanti'] = df_bikes['grave_quanti'].replace("1 - Blessé léger", 1)
df_bikes['grave_quanti'] = df_bikes['grave_quanti'].replace("2 - Blessé hospitalisé", 2)
df_bikes['grave_quanti'] = df_bikes['grave_quanti'].replace("3 - Tué", 3)

df_bikes['sex_quanti'] = df_bikes['sexe']
df_bikes['sex_quanti'] = df_bikes['sex_quanti'].replace("M", 0)
df_bikes['sex_quanti'] = df_bikes['sex_quanti'].replace("F", 1)

# Try to predict with qualitative variables into ordinal variables
# predict gravity of accident with sex (bad try)

interest = ['grave_quanti', 'sex_quanti']
df_bikes_ols = df_bikes[interest].dropna(how='any')  # Drop all Nan values
df_bikes_ols

# Sampling
df_test = df_bikes_ols.iloc[0:1000]
df_sample = df_bikes_ols.iloc[1000:len(df_bikes_ols)]

# Model with qualitative variables
results = smf.ols('grave_quanti ~ sex_quanti', data=df_sample).fit()

plt.figure()
plt.scatter(df_test['sex_quanti'], df_test['grave_quanti'], label='Données')
plt.title('Prediction of the severity with sex')
plt.xlabel('sex')
plt.ylabel('gravity of the accident')
plt.plot(df_test['sex_quanti'], results.predict(df_test), '--', label='OLS')
plt.xlim([-0.1, 1.1])
plt.legend()
plt.savefig('severitypredictionwithsex.pdf', bbox_inches="tight")

# Mew try on real quantitative variables
# Let's focus on the number of accidents

# Daily profile of the number of accidents per month

df_bikes['month'] = df_bikes.index.month
df_bikes['month'] = df_bikes['month'].apply(lambda x: calendar.month_abbr[x])

sns.set_palette("husl", n_colors=12)

df_bikes_month = df_bikes.groupby(['month', df_bikes.index.hour])[
    'mois'].count().unstack(level=0)

fig, axes = plt.subplots(1, 1, figsize=(7, 7), sharex=True)

df_bikes_month.plot(ax=axes)
axes.set_ylabel("Number of accidents")
axes.set_xlabel("Hour")
axes.set_title(
    "Daily profile of the number of accidents per month")
axes.set_xticks(np.arange(0, 24))
axes.set_xticklabels(np.arange(0, 24), rotation=45)
axes.legend(labels=calendar.month_name[1:], loc='lower left',
            bbox_to_anchor=(1, 0.1))
plt.savefig('Daily_profile.pdf', bbox_inches="tight")
plt.tight_layout()

# New study, reset dataset
df_bikes = pd.read_csv(path_target2, na_values="",
                       converters={'data': str, 'heure': str})
df_bikes["date"] = pd.to_datetime(df_bikes["date"])
df_bikes = df_bikes.sort_values(["date"])

df_bikes_date = pd.DataFrame(df_bikes.groupby(
    "date").count()["identifiant accident"])

df_bikes_date['date'] = df_bikes_date.index
df_bikes_date.rename(columns={'identifiant accident': 'accidents'},
                     inplace=True)

# Features
df_bikes_date['month'] = df_bikes_date.index.month
df_bikes_date['year'] = df_bikes_date.index.year
df_bikes_date['day'] = df_bikes_date.index.day
df_bikes_date['periodic_day'] = np.cos(df_bikes_date['day'])
df_bikes_date['periodic_month'] = np.cos(df_bikes_date['month'])

# Sampling
df_sample = df_bikes_date.iloc[355:5100]
df_test = df_bikes_date.iloc[0:355]

results = smf.ols('accidents ~ day + month + year + periodic_day + periodic_month', data=df_sample).fit()

plt.figure()
plt.scatter(df_test['date'], df_test['accidents'], label='Data')
plt.xlabel('date')
plt.ylabel('Number of accidents')
plt.title('Prediction of the number of accidents')
plt.plot(df_test['date'],
         results.predict(df_test), '--', color='red', label='OLS')
plt.legend()
plt.savefig('accidentprediction.pdf', bbox_inches="tight")


# New dataset
df_comptage = pd.read_excel(path_target1, engine="odf")  # dependence on odf
df_comptage.columns.unique()
variables = ['Date de passage', 'Heure de passage',
             'Nb total de vélos passés depuis la mise en circulation',
             'Nb de vélos passés depuis le début de la journée']
df_comptage = df_comptage[variables]
df_comptage.rename(columns={'Nb de vélos depuis lancement': 'Total',
                            'Nb de vélos depuis le début de la journée': 'Day_total'}, inplace=True)

df_comptage.Day_total[505] = 0  # it was a str

# Create features
df_comptage['date'] = pd.to_datetime(df_comptage["Date de passage"], format='%Y-%m-%d')
df_comptage_ag = df_comptage.groupby('date').aggregate({'Total': 'max', 'Day_total': 'max'})

df_comptage_ag['date'] = df_comptage_ag.index
df_comptage_ag['month'] = df_comptage_ag.index.month
df_comptage_ag['day'] = df_comptage_ag.index.day
df_comptage_ag['num'] = range(182)
df_comptage_ag['sinus_day'] = np.sin(df_comptage_ag['day'])
df_comptage_ag['sinus_num'] = np.sin(df_comptage_ag['num'])

# Sampling
df_sample = df_comptage_ag.iloc[0:150]
df_test = df_comptage_ag.iloc[150:182]

results2 = smf.ols('Day_total ~ num + Total + month + day + sinus_day + sinus_num', data=df_sample).fit()

plt.figure()
plt.scatter(df_test['num'], df_test['Day_total'], label='Data')
plt.xlabel('date')
plt.ylabel('Number of accidents')
plt.title('Prediction of the number of accidents')
plt.plot(df_test['num'], results2.predict(df_test), '--', color='red', label='OLS')
plt.legend()
plt.savefig('accidentpredictionalbert1.pdf', bbox_inches="tight")
