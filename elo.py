import pandas as pd
import math
from datetime import date

k_starting_value = 1.3644
k_decay = .9346
print("Calculating decay multipliers...")
decay_multipliers = [k_starting_value * (k_decay ** x) for x in range(120)]

def update_elo(home_team, away_team, home_score, away_score, home_win, elo_ratings, home_game_count, away_game_count, k=20):
    """
    Update Elo ratings based on the outcome of a game.
    
    Parameters:
    - home_team: str, name of the home team
    - away_team: str, name of the away team
    - home_win: bool, True if the home team won, False otherwise
    - elo_ratings: dict, current Elo ratings for all teams
    - K: int, K-factor, which determines how much the Elo rating is adjusted
    
    Returns:
    - new_home_elo: float, updated Elo rating for the home team
    - new_away_elo: float, updated Elo rating for the away team
    """
    
    # Get current Elo ratings
    home_elo = elo_ratings[home_team]
    away_elo = elo_ratings[away_team]
    
    # Calculate expected outcome
    expected_home = 1 / (1 + 10 ** ((away_elo - (home_elo + 76)) / 400))
    expected_away = 1 - expected_home
    
    # Determine actual outcome
    actual_home = 1 if home_win else 0
    actual_away = 1 - actual_home

    home_multiplier = decay_multipliers[home_game_count]
    away_multiplier = decay_multipliers[away_game_count]

    if home_win:
        mov = math.log(abs(home_score - away_score)+1) * (2.2/((home_elo-away_elo)*.001+2.2)) 
    else:
        mov = math.log(abs(home_score - away_score)+1) * (2.2/((away_elo-home_elo)*.001+2.2)) 
    
    # Update Elo ratings with the K-factor modified by the margin of victory
    new_home_elo = home_elo + k * (actual_home - expected_home) * mov * home_multiplier
    new_away_elo = away_elo + k * (actual_away - expected_away) * mov * away_multiplier
    elo_ratings[home_team] = new_home_elo
    elo_ratings[away_team] = new_away_elo




seasons = ['2022-23', '2021-22', '2020-21', '2019-20', '2018-19', '2017-18', '2016-17']

games = pd.read_csv('all_games.csv', index_col = 0)
averages = pd.read_csv('all_averages.csv', index_col = 0)



# Identify gaps in game dates that are larger than a certain threshold (e.g., 90 days)
# Assuming games and averages DataFrames are already defined
games['date'] = games['GAME_DATE'].apply(lambda x: date(*map(int, x.split('-'))))
games['date_diff'] = games['date'].diff().dt.days.fillna(0)
season_end_indices = games[games['date_diff'] > 90].index

elo_ratings = {team: 1500 for team in averages['teamTricode'].unique()}
elo = pd.DataFrame(columns=['gameId', 'teamTricode', 'elo'])
elo = elo.loc[:, ~elo.columns.str.contains('^Unnamed')]

# Sort the games DataFrame by date in ascending order
games.sort_values(by='date', ascending=True, inplace=True)

print("Calculating elos...")
for i, row in games.iterrows():
    home_team = row['HOME_TEAM_ABBREVIATION']
    away_team = row['AWAY_TEAM_ABBREVIATION']
    gameId = row['gameId']
    
    game_averages = averages[averages['gameId'] == gameId]
    try:
    
        home_game_count = int(game_averages[game_averages['teamTricode'] == home_team]['game_count'].iloc[0])
        

    
        away_game_count = int(game_averages[game_averages['teamTricode'] == away_team]['game_count'].iloc[0])

        home_win = row['HOME_TEAM_PTS'] > row['AWAY_TEAM_PTS']

        new_rows = [
            {'gameId' : gameId, 'teamTricode' : home_team, 'elo' : elo_ratings[home_team]},
            {'gameId' : gameId, 'teamTricode' : away_team, 'elo' : elo_ratings[away_team]}
        ]

        elo = elo.append(new_rows, ignore_index = True)



        update_elo(home_team, away_team, row['HOME_TEAM_PTS'], row['AWAY_TEAM_PTS'], home_win, elo_ratings, home_game_count, away_game_count)

        if i in season_end_indices:
            for (key, value) in elo_ratings.items():
                elo_ratings[key] = value * .75 + 1505 * .25
    except:
        continue


print(elo_ratings)
averages = pd.merge(averages, elo, on = ['gameId', 'teamTricode'], how = 'left')
print("Saving...")

df_current_profiles = pd.read_csv('current_profiles.csv', index_col = 0)
elo = pd.DataFrame(columns=['teamTricode', 'elo'])
print(dict(sorted(elo_ratings.items(), key=lambda item: item[1], reverse=True)))
for team in elo_ratings:
    row = [{'teamTricode' : team, 'elo' : elo_ratings[team]}]
    elo = elo.append(row, ignore_index = True)
print(elo)
df_current_profiles = pd.merge(df_current_profiles, elo, on=['teamTricode'], how = 'left')
df_current_profiles.to_csv('current_profiles.csv')

averages.to_csv('all_averages.csv')
