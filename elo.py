import pandas as pd
import math
from datetime import date

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

    #home_multiplier = decay_multipliers[home_game_count]
    #away_multiplier = decay_multipliers[away_game_count]

    if home_win:
        mov = math.log(abs(home_score - away_score)+1) * (2.2/((home_elo-away_elo)*.001+2.2)) 
    else:
        mov = math.log(abs(home_score - away_score)+1) * (2.2/((away_elo-home_elo)*.001+2.2)) 
    
    # Update Elo ratings with the K-factor modified by the margin of victory
    new_home_elo = home_elo + k * (actual_home - expected_home) * mov #* home_multiplier
    new_away_elo = away_elo + k * (actual_away - expected_away) * mov #* away_multiplier
    elo_ratings[home_team] = new_home_elo
    elo_ratings[away_team] = new_away_elo




seasons = ['2022-23', '2021-22', '2020-21', '2019-20', '2018-19', '2017-18', '2016-17']


def elo(games, averages, current):

    # Identify gaps in game dates that are larger than a certain threshold (e.g., 90 days)
    # Assuming games and averages DataFrames are already defined

    games['date_diff'] = games['GAME_DATE'].diff().dt.days.fillna(0)
    season_end_indices = games[games['date_diff'] > 90].index

    elo_ratings = {team: 1500 for team in averages['teamTricode'].unique()}
    elo = pd.DataFrame(columns=['gameId', 'teamTricode', 'elo'])
    elo = elo.loc[:, ~elo.columns.str.contains('^Unnamed')]

    # Sort the games DataFrame by date in ascending order
    games.sort_values(by='GAME_DATE', ascending=True, inplace=True)

    print("Calculating elos...")
    for i, row in games.iterrows():
        home_team = row['HOME_TEAM_ABBREVIATION']
        away_team = row['AWAY_TEAM_ABBREVIATION']
        gameId = row['gameId']
        
        game_averages = averages[averages['gameId'] == gameId]
        
        if (game_averages.shape[0] == 0):
            print(gameId)

        home_game_count = int(game_averages[game_averages['teamTricode'] == home_team]['game_count'].iloc[0])

        away_game_count = int(game_averages[game_averages['teamTricode'] == away_team]['game_count'].iloc[0])

        home_win = row['HOME_TEAM_PTS'] > row['AWAY_TEAM_PTS']

        new_rows = [
            {'gameId' : gameId, 'teamTricode' : home_team, 'elo' : elo_ratings[home_team]},
            {'gameId' : gameId, 'teamTricode' : away_team, 'elo' : elo_ratings[away_team]}
        ]

        new_rows = pd.DataFrame(new_rows)
        elo = pd.concat([elo, new_rows])



        update_elo(home_team, away_team, row['HOME_TEAM_PTS'], row['AWAY_TEAM_PTS'], home_win, elo_ratings, home_game_count, away_game_count)

        if i in season_end_indices:
            for (key, value) in elo_ratings.items():
                elo_ratings[key] = value * .75 + 1505 * .25
    
        


    print(elo_ratings)
    print(elo.tail(50))
    print(averages.shape)
    # elo_ratings is your dictionary where keys are teamTricodes and values are Elo ratings
    elo_df = pd.DataFrame.from_dict(elo_ratings, orient='index', columns=['elo'])

    # Reset the index to turn the indices into a column
    elo_df = elo_df.reset_index()

    # Rename the columns appropriately
    elo_df.columns = ['teamTricode', 'elo']

    if not current:
        averages = pd.merge(averages, elo, on = ['gameId', 'teamTricode'], how = 'inner')
    return averages, elo_df
    print(averages.shape)
    print("Saving...")


