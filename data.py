import requests
import json 
import pandas as pd
import nfl_data_py as nfl

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

  

def expectedScore(opp_team_rating, team_rating):
    return (1 / (1 + 10**((opp_team_rating - team_rating) / 400)))

def newRating(team_rating, observed_score, expected_score, k_factor):
    return(team_rating + k_factor * (observed_score - expected_score))

def games(years):
    return nfl.import_schedules([years])[['game_id', 'home_score', 'away_score']]

def teamStats(year, week):
    pointsDict = {'ARI': [0, 0, 0], 'ATL': [0, 0, 0], 'BAL': [0, 0, 0], 'BUF': [0, 0, 0], 'CAR': [0, 0, 0], 'CHI': [0, 0, 0], 'CIN': [0, 0, 0], 'CLE': [0, 0, 0], 'DAL': [0, 0, 0], 'DEN': [0, 0, 0], 'DET': [0, 0, 0], 'GB': [0, 0, 0], 'HOU': [0, 0, 0], 'IND': [0, 0, 0], 'JAX': [0, 0, 0], 'KC': [0, 0, 0], 'LV': [0, 0, 0], 'LAC': [0, 0, 0], 'LA': [0, 0, 0], 'MIA': [0, 0, 0], 'MIN': [0, 0, 0], 'NE': [0, 0, 0], 'NO': [0, 0, 0], 'NYG': [0, 0, 0], 'NYJ': [0, 0, 0], 'PHI': [0, 0, 0], 'PIT': [0, 0, 0], 'SF': [0, 0, 0], 'SEA': [0, 0, 0], 'TB': [0, 0, 0], 'TEN': [0, 0, 0], 'WAS': [0, 0, 0]}
    
    teamDict = {'ARI': [0, 0, 0, 1500, 0, 0, 0], 'ATL': [0, 0, 0, 1500, 0, 0, 0], 'BAL': [0, 0, 0, 1500, 0, 0, 0], 'BUF': [0, 0, 0, 1500, 0, 0, 0], 'CAR': [0, 0, 0, 1500, 0, 0, 0], 'CHI': [0, 0, 0, 1500, 0, 0, 0], 'CIN': [0, 0, 0, 1500, 0, 0, 0], 'CLE': [0, 0, 0, 1500, 0, 0, 0], 'DAL': [0, 0, 0, 1500, 0, 0, 0], 'DEN': [0, 0, 0, 1500, 0, 0, 0], 'DET': [0, 0, 0, 1500, 0, 0, 0], 'GB': [0, 0, 0, 1500, 0, 0, 0], 'HOU': [0, 0, 0, 1500, 0, 0, 0], 'IND': [0, 0, 0, 1500, 0, 0, 0], 'JAX': [0, 0, 0, 1500, 0, 0, 0], 'KC': [0, 0, 0, 1500, 0, 0, 0], 'LV': [0, 0, 0, 1500, 0, 0, 0], 'LAC': [0, 0, 0, 1500, 0, 0, 0], 'LA': [0, 0, 0, 1500, 0, 0, 0], 'MIA': [0, 0, 0, 1500, 0, 0, 0], 'MIN': [0, 0, 0, 1500, 0, 0, 0], 'NE': [0, 0, 0, 1500, 0, 0, 0], 'NO': [0, 0, 0, 1500, 0, 0, 0], 'NYG': [0, 0, 0, 1500, 0, 0, 0], 'NYJ': [0, 0, 0, 1500, 0, 0, 0], 'PHI': [0, 0, 0, 1500, 0, 0, 0], 'PIT': [0, 0, 0, 1500, 0, 0, 0], 'SF': [0, 0, 0, 1500, 0, 0, 0], 'SEA': [0, 0, 0, 1500, 0, 0, 0], 'TB': [0, 0, 0, 1500, 0, 0, 0], 'TEN': [0, 0, 0, 1500, 0, 0, 0], 'WAS': [0, 0, 0, 1500, 0, 0, 0]}

    for g in games(year).itertuples():
        matchUp = g[1][8:].split("_")
        teamDict[matchUp[0]][6] = week
        teamDict[matchUp[1]][6] = week
        if matchUp[0] == 'OAK':
            matchUp[0] = 'LV'
        if matchUp[1] == 'OAK':
            matchUp[1] = 'LV'
        #if int(g[1][5:7]) > week:
         #   break
            #when home team
        pointsDict[matchUp[1]][0] += g[2] #PF
        pointsDict[matchUp[1]][1] += g[3] #PA
        pointsDict[matchUp[1]][2] += (g[2] - g[3]) #D
            #when away team
        pointsDict[matchUp[0]][0] += g[3] #PF
        pointsDict[matchUp[0]][1] += g[2] #PA
        pointsDict[matchUp[0]][2] += (g[3] - g[2]) #D
        if g[2] > g[3]: #home team won
            if int(g[1][5:7]) < 19:
                teamDict[matchUp[1]][0] += 1
                teamDict[matchUp[0]][1] += 1

                teamDict[matchUp[1]][3]= newRating(teamDict[matchUp[1]][3], 1, expectedScore(teamDict[matchUp[0]][3], teamDict[matchUp[1]][3]), 20)
                teamDict[matchUp[0]][3] = newRating(teamDict[matchUp[0]][3], 0, expectedScore(teamDict[matchUp[1]][3], teamDict[matchUp[0]][3]), 20)
            if int(g[1][5:7]) >= 19:
                teamDict[matchUp[1]][3]= newRating(teamDict[matchUp[1]][3], 1, expectedScore(teamDict[matchUp[0]][3], teamDict[matchUp[1]][3]), 40)
                teamDict[matchUp[0]][3] = newRating(teamDict[matchUp[0]][3], 0, expectedScore(teamDict[matchUp[1]][3], teamDict[matchUp[0]][3]), 40)

                teamDict[matchUp[1]][4] += 1
                teamDict[matchUp[1]][5] += 1
                
                teamDict[matchUp[0]][4] += 1
        elif g[2] < g[3]: #away team won
            if int(g[1][5:7]) < 19:
                teamDict[matchUp[1]][1] += 1
                teamDict[matchUp[0]][0] += 1

                teamDict[matchUp[0]][3] = newRating(teamDict[matchUp[0]][3], 1, expectedScore(teamDict[matchUp[1]][3], teamDict[matchUp[0]][3]), 20)
                teamDict[matchUp[1]][3] = newRating(teamDict[matchUp[1]][3], 0, expectedScore(teamDict[matchUp[0]][3], teamDict[matchUp[1]][3]), 20)
            #playoffs
            if int(g[1][5:7]) >= 19:
                teamDict[matchUp[0]][3] = newRating(teamDict[matchUp[0]][3], 1, expectedScore(teamDict[matchUp[1]][3], teamDict[matchUp[0]][3]), 40)
                teamDict[matchUp[1]][3] = newRating(teamDict[matchUp[1]][3], 0, expectedScore(teamDict[matchUp[0]][3], teamDict[matchUp[1]][3]), 40)

                teamDict[matchUp[0]][4] += 1
                teamDict[matchUp[0]][5] += 1
                
                teamDict[matchUp[1]][4] += 1
        else:
            teamDict[matchUp[1]][2] += 1
            teamDict[matchUp[0]][2] += 1
            
            teamDict[matchUp[0]][3] = newRating(teamDict[matchUp[0]][3], .5, expectedScore(teamDict[matchUp[1]][3], teamDict[matchUp[0]][3]), 20)
            teamDict[matchUp[1]][3] = newRating(teamDict[matchUp[1]][3], .5, expectedScore(teamDict[matchUp[0]][3], teamDict[matchUp[1]][3]), 20)
    teamdf = pd.DataFrame(data = teamDict)
    teamdf = teamdf.transpose()
    teamdf = teamdf.rename(columns = {0: "Wins", 1: "Loses", 2: "Ties", 3: 'ELO', 4: 'Playoff Games', 5: 'Playoff Wins', 6: 'Week'})
    teamdf.index.name = 'Teams'

    pointsdf = pd.DataFrame(data=pointsDict)
    pointsdf = pointsdf.transpose()
    pointsdf = pointsdf.rename(columns = {0: "PF", 1: "PA", 2: "D"})
    pointsdf.index.name = 'Teams'
    return pd.merge(teamdf, pointsdf, left_index=True, right_index=True)
print(teamStats(2023, 15))
def statsForLinear(year, week, homeTeam, awayTeam):
    homeStats = pd.DataFrame(teamStats(year, week).loc[homeTeam])

    awayStats = pd.DataFrame(teamStats(year, week).loc[awayTeam])
    

    matchUpdf = pd.concat([homeStats, awayStats], axis=1).transpose()
    matchUpdf['WP'] = matchUpdf['Wins'] / 17
    matchUpdf = matchUpdf.drop(columns=['PF', 'PA','Wins', 'Loses', 'Ties', 'Playoff Games', 'Playoff Wins'])
    
    return matchUpdf
#print(statsForLinear(2023, 1, 'DET', 'KC'))

#print(statsForLinear(2022))
    #newGames = games(2024)['game_id']
    #week = 1
    #for g in newGames:
    #    if int(g[5:7]) != week:
     #       break
      #  matchUp = g[8:].split("_")
       # linearStats.loc[matchUp[1], 'opp'] = matchUp[0]
        #linearStats.loc[matchUp[0], 'opp'] = matchUp[1]


def linearModel():
    years = [2019, 2020, 2021, 2022, 2023]
    models = []
    train_accuracies = []
    test_accuracies = []

    for year in years:
        X_all = []
        y_all = []
        data = games(year)
        for g in data.itertuples():
            week = int(g[1][5:7])-1
            year = int(g[1][0:4])
            print(week+1, year)
            matchUp = g[1][8:].split("_")
            if matchUp[0] == 'OAK':
                matchUp[0] = 'LV'
            if matchUp[1] == 'OAK':
                matchUp[1] = 'LV'
            homeTeam = matchUp[0]
            awayTeam = matchUp[1]

            currentData = statsForLinear(year, week, homeTeam, awayTeam)
            currentData['outcome'] = g[2] > g[3]
            X_all.append(currentData[['ELO', 'D', 'WP']])
            y_all.append(currentData['outcome']) 

        X_all = pd.concat(X_all, axis=0)
        y_all = pd.concat(y_all, axis=0)

        X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Initialize logistic regression model
        model = LogisticRegression()
        
        # Train the model
        model.fit(X_train_scaled, y_train)
        
        # Evaluate the model on training data
        train_pred = model.predict(X_train_scaled)
        train_accuracy = accuracy_score(y_train, train_pred)
        train_accuracies.append(train_accuracy)
        
        # Evaluate the model on test data
        test_pred = model.predict(X_test_scaled)
        test_accuracy = accuracy_score(y_test, test_pred)
        test_accuracies.append(test_accuracy)
        
        # Store the trained model for each year
        models.append(model)


    return models, train_accuracies, test_accuracies

    #probabilities = model.predict_proba(X_test_scaled)[:, 1] 
    #odds = 1 / probabilities

#models, train_accuracies, test_accuracies = linearModel()

#print("Models:", models)
#print("Train Accuracies:", train_accuracies)
#print("Test Accuracies:", test_accuracies)


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


#def stats

#print(nfl.import_depth_charts([2023])[['full_name', 'depth_position']])

#print(nfl.import_seasonal_pfr("pass", [2023]))