import requests
import json 
import pandas as pd
import nfl_data_py as nfl

def nflGames2024():
    return nfl.import_schedules([2024])[["game_id", "gameday"]]    

def TeamWL2023():
    teamDict = {'ARI', 'ATL', 'BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE', 'DAL', 'DEN', 'DET', 'GB', 'HOU', 'IND', 'JAC', 'KC', 'LV', 'LAC', 'LAR', 'MIA', 'MIN', 'NE', 'NO', 'NYG', 'NYJ', 'PHI', 'PIT', 'SF', 'SEA', 'TB', 'TEN', 'WAS'}
    teamdf = pd.DataFrame()
    return nfl.import_schedules([2023])[['game_id', 'home_score', 'away_score']]

def QbStats():
    Qbs = nfl.import_depth_charts([2023])
    Qbs = Qbs[Qbs["depth_position"] == 'QB']['full_name'] 
    Qbs = Qbs.drop_duplicates()
    QbsDf = pd.DataFrame(Qbs)
    QbsDf = QbsDf.rename(columns = {'full_name': 'player'})
    passingStats = nfl.import_seasonal_pfr("pass", [2023])
    rushingStats = nfl.import_seasonal_pfr("rush", [2023])
    mergedStats = pd.merge(passingStats, rushingStats)
    return pd.merge(mergedStats, QbsDf)

print(TeamWL2023())
#def stats

#print(nfl.import_depth_charts([2023])[['full_name', 'depth_position']])

#print(nfl.import_seasonal_pfr("pass", [2023]))