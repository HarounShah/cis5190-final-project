# IMPORTS =====================================================================
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import requests
import os
from bs4 import BeautifulSoup
import json
import time
import cfbd
import ast

API_KEY = os.environ.get("CFBD_API_KEY")
if not API_KEY:
    raise RuntimeError("CFBD_API_KEY not set")

HEADERS = {"Authorization": f"Bearer {API_KEY}", "accept": "application/json"}
BASE_URL = "https://api.collegefootballdata.com/games/teams"

year = 2025  # inclusive 2014–2024
MAX_WEEKS = 18  # most seasons have up to 15 weeks (including bowls)
SLEEP_TIME = 0.5  # seconds between requests to avoid rate limits

all_records = []

print(f"\n=== {year} ===")
for week in range(1, MAX_WEEKS + 1):
    params = {
        "year": year,
        "week": week,
    }

    resp = requests.get(BASE_URL, headers=HEADERS, params=params)
    if resp.status_code != 200:
        print(f"⚠️ {year} week {week}: HTTP {resp.status_code}")
        continue

    data = resp.json()
    if not data:
        print(f"⏭️ No data for {year} week {week}")
        continue

    # Flatten team stats
    for game in data:
        game_id = game.get("id")
        for team_entry in game.get("teams", []):
            record = {
                "year": year,
                "week": week,
                "game_id": game_id,
                "teamId": team_entry.get("teamId"),
                "team": team_entry.get("team"),
                "conference": team_entry.get("conference"),
                "homeAway": team_entry.get("homeAway"),
                "points": team_entry.get("points")
            }
            for stat_entry in team_entry.get("stats", []):
                cat = stat_entry.get("category")
                val = stat_entry.get("stat")
                record[cat] = val
            all_records.append(record)

    print(f"✅ {year} week {week}: {len(data)} games ({len(data)*2} team rows)")
    time.sleep(SLEEP_TIME)

# Convert to DataFrame
df = pd.DataFrame(all_records)

# Convert numeric columns where possible
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="ignore")

# Save full dataset
df.to_csv("full_data_test.csv", index=False)
print(f"\n💾 Saved {df.shape[0]} rows × {df.shape[1]} columns to full_data_test.csv")