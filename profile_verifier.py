import pandas as pd
from games import fetch_games_today
current_profiles = pd.read_csv('current_profiles.csv', index_col = 0)
averages = pd.read_csv('all_averages.csv', index_col = 0)

def yesterdays_teams():
    yesterdays_games = fetch_games_today(True)
    print(yesterdays_games)
    all_teams = []
    for game in yesterdays_games:
        teams = game.split('-')
        all_teams += teams
    print(all_teams)
    return all_teams




played = yesterdays_teams()
teams = averages.groupby('teamTricode')
for name, group in teams:
    if name in played:
        print(name)
        group = group.sort_values(by = 'date', ascending = False)
        row = group.iloc[0]
        profile = current_profiles[current_profiles['teamTricode'] == name]
        profile = profile[group.columns]
        profile = profile.iloc[0]
        profile = profile.drop(labels = ['gameId', 'date'])
        row = row.drop(labels = ['gameId', 'date'])
        
        import numpy as np

        # Find indices where the two series differ
        different_indices = np.where(row != profile)[0]

        # Map indices to column names
        different_columns = row.index[different_indices]
        #print(different_columns)
        
        for col in different_columns:
            if 'minutes_50' in col:
                words = col.split('_')
                print(f'{words[0]} {words[1]}')
                #print(row[different_columns])
                #print(profile[different_columns])
                #print(row[col])
                #print(profile[col])
        
        print(f"Accuracy for {name}: {(1 - len(different_columns) / len(row)) * 100}%")


