# IMPORTS =====================================================================
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
import json
import time

# --- Load team-level data ---
df = pd.read_csv("full_data_test.csv")
print(f"\nOriginal shape: {df.shape}")

# --- Split into home and away teams ---
home = df[df["homeAway"] == "home"].copy()
away = df[df["homeAway"] == "away"].copy()

# --- Merge into one row per game ---
games = pd.merge(
    home,
    away,
    on=["game_id", 'year', 'week'],
    suffixes=("_home", "_away")
)

print(f"\nMerged shape: {games.shape}")
# print(games[["game_id", "team_home", "team_away", "points_home", "points_away"]].head())

# --- Create target variables ---
games["home_win"] = (games["points_home"] > games["points_away"]).astype(int)
games["one_score"] = (abs(games["points_home"] - games["points_away"]) <= 8).astype(int)
print(f"\nAdded 2 binary features: home_win & one_score.")

# --- Dropping Non-Interesting Columns ---
cols_to_drop = [
    'totalPenaltiesYards_away', 
    'totalPenaltiesYards_home', 
    'kickReturns_away', 
    'kickReturnYards_away', 
    'kickReturnTDs_away', 
    'kickReturnYards_home',
    'kickReturnTDs_home',
    'kickReturns_home',
    'interceptionYards_away',
    'interceptionTDs_away',
    'interceptionTDs_home',
    'interceptionYards_home',
    'puntReturnYards_away',
    'puntReturns_away',
    'puntReturnTDs_away',
    'puntReturnYards_home',
    'puntReturns_home',
    'puntReturnTDs_home',
    'defensiveTDs_away',
    'defensiveTDs_home',
    'passesIntercepted_home', # interceptions is better
    'passesIntercepted_away',
    'totalFumbles_home', # fumblesLost is better
    'totalFumbles_away'
    ]
games = games.drop(cols_to_drop, axis = 1)
print(f"\nDropped {len(cols_to_drop)} features.")

# --- Getting efficiency metrics ---

temp = games["thirdDownEff_home"].str.split("-", expand=True)
temp = temp.apply(lambda col: pd.to_numeric(col, errors="coerce"))
games["thirdDown%_home"] = temp[0] / temp[1]
games.loc[temp[1] == 0, "thirdDown%_home"] = 0

temp = games["fourthDownEff_home"].str.split("-", expand=True)
temp = temp.apply(lambda col: pd.to_numeric(col, errors="coerce"))
games["fourthDown%_home"] = temp[0] / temp[1]
games.loc[temp[1] == 0, "fourthDown%_home"] = 0

temp = games["thirdDownEff_away"].str.split("-", expand=True)
temp = temp.apply(lambda col: pd.to_numeric(col, errors="coerce"))
games["thirdDown%_away"] = temp[0] / temp[1]
games.loc[temp[1] == 0, "thirdDown%_away"] = 0

# temp = games["fourthDownEff_away"].str.split("-", expand=True)
# temp = temp.apply(lambda col: pd.to_numeric(col, errors="coerce"))
# games["fourthDown%_away"] = temp[0] / temp[1]
# games.loc[temp[1] == 0, "fourthDown%_away"] = 0

print("\nAdded 3 third/fourth down efficiency features.")
# print(games[["thirdDown%_home", "fourthDown%_home", "thirdDown%_away", "fourthDown%_away"]].head())


# --- Identify numeric columns to create difference features ---
num_cols_names = ["time", "yards", "turnovers", 'fumbleslost', 'turnovers', 
                  'sacks', 'passesdef', 'hurries', 'tackles', 'interceptions', 'downs', 'down%']
num_cols = [col for col in games.columns if any(stat.lower() in col.lower() for stat in num_cols_names)]
# print(num_cols)

# --- Convert numeric columns safely ---
for col in num_cols:
    if 'time' in col.lower():
        games[col] = pd.to_timedelta("00:" + games[col])
    else:
        games[col] = pd.to_numeric(games[col], errors="coerce")  # non-numeric -> NaN

# --- Create difference features: home - away ---
for col in num_cols:
    if col.endswith("_home") and col.replace("_home", "_away") in games.columns:
        away_col = col.replace("_home", "_away")
        diff_col = col.replace("_home", "_diff")
        games[diff_col] = games[col] - games[away_col]
print(f"\nDropped {int(len(num_cols) / 2)} numerical features, then")
print(f"Created {len(num_cols)} difference features.")



# --- Remove Nulls and Save the Combined Dataset ---
games = games[games["one_score"] == 1]
print(f"\nShape of One-Score Games: {games.shape}")

pd.set_option('display.max_rows', None)
null_counts = games.isnull().sum().sort_values(ascending=False)
# print(null_counts)

no_nulls = games.dropna().copy()
print(f"\nShape After Dropping Nulls: {no_nulls.shape}")
# print(no_nulls.columns)

no_nulls['possessionTime_diff'] = no_nulls['possessionTime_diff'].dt.total_seconds().astype(int)

no_nulls.to_csv("final_data.csv", index=False)
print(f"\nSaved {no_nulls.shape[0]} rows × {no_nulls.shape[1]} columns → final_data.csv")