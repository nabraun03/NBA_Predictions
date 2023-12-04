import pandas as pd
import numpy as np
from datetime import date, datetime
from injuries import fetch_injured_players

injured_players = fetch_injured_players()
seasons = ['2023-24', '2022-23', '2021-22', '2020-21', '2019-20', '2018-19', '2017-18', '2016-17']

def calculate_days_between(d1, d2):
    return (date(d1[0], d1[1], d1[2]) - date(d2[0], d2[1], d2[2])).days


all_games = pd.DataFrame()
all_averages = pd.DataFrame()
df_player_stats = pd.DataFrame()


for season in seasons:
    player_logs = pd.read_csv(f'{season}_player_logs.csv', index_col=0)
    df_player_stats = pd.concat([df_player_stats, player_logs])

df_player_stats['GAME_DATE'] = pd.to_datetime(df_player_stats['GAME_DATE'])
df_player_stats['shifted_available_flag'] = df_player_stats.groupby('PLAYER_ID')['AVAILABLE_FLAG'].shift(-1)

count = 0




def get_current_roster(df_player_stats, team_id, injured_players):
    # Define a recent activity window (e.g., last 15 days)
    recent_date_threshold = pd.to_datetime('today') - pd.Timedelta(days=15)

    # Filter players with recent activity
    recent_active_players = df_player_stats[pd.to_datetime(df_player_stats['GAME_DATE']) >= recent_date_threshold]

    # Group by player and get the most recent entry for each player
    most_recent_games = recent_active_players.groupby('PLAYER_ID').apply(lambda x: x.sort_values(by='GAME_DATE', ascending=False).head(1)).reset_index(drop=True)

    # Filter out players whose most recent game was not with the specified team
    print(team_id)
    most_recent_team_players = most_recent_games[most_recent_games['TEAM_ABBREVIATION'] == team_id]

    # Remove injured players
    active_roster = most_recent_team_players[~most_recent_team_players['PLAYER_NAME'].isin(injured_players)]

    # Sort by relevant criteria (e.g., minutes played)
    sorted_roster = active_roster.sort_values(by='running_avg_PTS', ascending=False)

    return sorted_roster.head(12)

def get_top_12_players(df_games, df_player_stats, game_id, team_id):
    game_date = df_games.loc[df_games['gameId'] == game_id, 'GAME_DATE'].iloc[0]



    # Pre-filter the players based on the team and active period
    game_players = df_player_stats[(df_player_stats['GAME_ID'] == game_id) & 
                                (df_player_stats['TEAM_ABBREVIATION'] == team_id) & 
                                (df_player_stats['shifted_available_flag'] == 1)]


    # Calculate average minutes and get top 10
    top_players = game_players.head(12).sort_values(by='running_avg_PTS', ascending=False)
    return top_players


# Assuming 'df_player_stats' contains player stats with a column 'date' for game dates
span = 50  # Number of games to consider for the running average

df_player_stats = df_player_stats.sort_values(by = ['GAME_DATE'], ascending = True)
player_stats_columns = ['MIN','FGM','FGA','FG3M','FG3A','FTM','FTA','OREB','DREB','REB','AST','TOV','STL','BLK','BLKA','PF','PFD','PTS','PLUS_MINUS','NBA_FANTASY_PTS','DD2','TD3','WNBA_FANTASY_PTS']
for stat in player_stats_columns:  # Skipping gameId, playerId, teamId
    df_player_stats[f'running_avg_{stat}'] = df_player_stats.groupby('PLAYER_ID')[stat].transform(lambda x: x.ewm(span=span).mean().shift(1))
ranks = ['GP_RANK','W_RANK','L_RANK','W_PCT_RANK','MIN_RANK','FGM_RANK','FGA_RANK','FG_PCT_RANK','FG3M_RANK','FG3A_RANK','FG3_PCT_RANK','FTM_RANK','FTA_RANK','FT_PCT_RANK','OREB_RANK','DREB_RANK','REB_RANK','AST_RANK','TOV_RANK','STL_RANK','BLK_RANK','BLKA_RANK','PF_RANK','PFD_RANK','PTS_RANK','PLUS_MINUS_RANK','NBA_FANTASY_PTS_RANK','DD2_RANK','TD3_RANK','WNBA_FANTASY_PTS_RANK']
for rank in ranks:
    df_player_stats[f'running_avg_{rank}'] = df_player_stats.groupby('PLAYER_ID')[rank].transform(lambda x: x.ewm(span=span).mean().shift(1))
df_player_stats = df_player_stats.drop(columns = player_stats_columns)
df_player_stats = df_player_stats.drop(columns = ranks)

print("Generating most recent teams")

# Sort the dataframe by 'PLAYER_ID' and 'GAME_DATE' first
df_player_stats = df_player_stats.sort_values(by=['PLAYER_ID', 'GAME_DATE'])

# Compute the most recent team for each player

print('done')
df_player_stats.to_csv('all_player_logs.csv')

df_player_stats = pd.read_csv('all_player_logs.csv')

for season in seasons:

    df_games = pd.read_csv(f'{season}_all_games.csv')
    df_advanced = pd.read_csv(f'{season}_advanced_stats.csv')[['gameId', 'teamTricode', 'estimatedOffensiveRating', 'offensiveRating', 'estimatedDefensiveRating', 'defensiveRating', 'estimatedNetRating', 'netRating', 'estimatedPace', 'pace', 'pacePer40', 'possessions', 'PIE']]
    df_basic = pd.read_csv(f'{season}_traditional_stats.csv')[['gameId', 'teamTricode', 'fieldGoalsMade', 'fieldGoalsAttempted', 'threePointersMade', 'threePointersAttempted', 'freeThrowsMade', 'freeThrowsAttempted', 'reboundsOffensive', 'reboundsDefensive', 'reboundsTotal', 'assists', 'steals', 'blocks', 'turnovers', 'foulsPersonal', 'points', 'plusMinusPoints']]
    df_hustle = pd.read_csv(f'{season}_hustle_stats.csv')[['gameId', 'teamTricode', 'contestedShots', 'contestedShots2pt', 'contestedShots3pt', 'deflections', 'chargesDrawn', 'screenAssists', 'screenAssistPoints', 'looseBallsRecoveredOffensive', 'looseBallsRecoveredDefensive', 'looseBallsRecoveredTotal', 'offensiveBoxOuts', 'defensiveBoxOuts', 'boxOutPlayerTeamRebounds', 'boxOutPlayerRebounds', 'boxOuts']]
    df_misc = pd.read_csv(f'{season}_misc_stats.csv')[['gameId', 'teamTricode', 'pointsOffTurnovers', 'pointsSecondChance', 'pointsFastBreak', 'pointsPaint', 'oppPointsOffTurnovers', 'oppPointsSecondChance', 'oppPointsFastBreak', 'oppPointsPaint', 'blocksAgainst', 'foulsDrawn']]
    df_tracking = pd.read_csv(f'{season}_track_stats.csv')[['gameId', 'teamTricode', 'distance', 'reboundChancesOffensive', 'reboundChancesDefensive', 'reboundChancesTotal', 'touches', 'secondaryAssists', 'freeThrowAssists', 'passes', 'contestedFieldGoalsMade', 'contestedFieldGoalsAttempted', 'uncontestedFieldGoalsMade', 'uncontestedFieldGoalsAttempted', 'defendedAtRimFieldGoalsMade', 'defendedAtRimFieldGoalsAttempted']]

    

    dfs_to_merge = [df_basic, df_hustle, df_misc, df_tracking]
    df_merged = df_advanced
    for df in dfs_to_merge:

        df_merged = pd.merge(df_merged, df, on=['gameId', 'teamTricode'], how = 'inner')


    # Step 1: Merge the date column into df_merged
    df_merged = pd.merge(df_merged, df_games[['gameId', 'GAME_DATE']], on='gameId', how='left')
    df_merged['GAME_DATE'] = df_merged['GAME_DATE'].astype(str)
    # Step 2: Convert the date column to datetime format

    df_merged['date'] = df_merged['GAME_DATE'].apply(lambda x: date(*map(int, x.split('-'))))

    # ... (your code to read CSV files and merge DataFrames)

    # Initialize a new DataFrame to store running averages

    # Step 1: Calculate the winner for each game in df_games
    df_games['winner'] = np.where(df_games['HOME_TEAM_PTS'] > df_games['AWAY_TEAM_PTS'], df_games['HOME_TEAM_ABBREVIATION'], df_games['AWAY_TEAM_ABBREVIATION'])

    # Step 2: Merge this winner information into df_running_avgs
    df_merged = pd.merge(df_merged, df_games[['gameId', 'winner']], on='gameId', how='left')

    # Step 3: Calculate the winning percentage
    grouped = df_merged.groupby('teamTricode')

    # Initialize a new DataFrame to store running averages and winning percentages
    df_running_avgs = pd.DataFrame()
    if season == '2023-24':
        current_profiles = pd.DataFrame()

    for name, group in grouped:
        group = group.sort_values('date')
        

        # Count the number of games for each team up to each point
        group['game_count'] = group['gameId'].expanding().count()
        
        # Calculate if the team won or lost
        group['win'] = (group['teamTricode'] == group['winner']).astype(int)

        group['winning_percentage'] = group['win'].ewm(span=90, adjust=False).mean().shift(1)
        

        group['game_count'] = group['gameId'].expanding().count()
        group['time_between_games'] = group['date'].diff().dt.days
        group['playoff'] = group['game_count'] > 82
        
        # Identify unique columns for which you want to calculate running averages
        unique_columns = [col for col in group.columns if col not in ['teamTricode', 'gameId', 'HOME_TEAM', 'date', 'time_between_games', 'winning_percentage', 'win', 'winner', 'game_count', 'GAME_DATE', 'playoff']]
        
        if season == '2023-24':
            temp = group.copy()

            for col in unique_columns:
                running_avg_col_name = f'running_avg_{col}'
                temp[running_avg_col_name] = temp[col].ewm(span=90, min_periods=1).mean()
            # Sort by date
            temp.sort_values(by = ['date'], ascending = False)

            current_profiles = pd.concat([current_profiles, temp.iloc[-1].to_frame().T])

        for col in unique_columns:
            running_avg_col_name = f'running_avg_{col}'
            group[running_avg_col_name] = group[col].ewm(span=90, min_periods=1).mean().shift(1)

    
        cols_to_keep = ['teamTricode', 'gameId', 'time_between_games', 'game_count', 'winning_percentage', 'playoff', 'date'] + [f'running_avg_{col}' for col in unique_columns]
        group = group[cols_to_keep]

        df_running_avgs = pd.concat([df_running_avgs, group])

    if season == '2023-24':
        cols_to_order = ['gameId', 'playoff', 'teamTricode', 'time_between_games', 'game_count', 'date']
        new_columns = cols_to_order + [col for col in current_profiles.columns if col not in cols_to_order]
        current_profiles = current_profiles[new_columns]

        current_profiles['time_between_games'] = (date.today() - current_profiles['date']).dt.days
        current_profiles['game_count'] = current_profiles['game_count'] + 1
        cols_to_keep = ['teamTricode', 'gameId', 'time_between_games', 'game_count', 'winning_percentage', 'playoff', 'date'] + [f'running_avg_{col}' for col in unique_columns]
        current_profiles = current_profiles[cols_to_keep]
        

    #df_running_avgs = df_running_avgs[df_running_avgs['game_count'] >= 10]
    #df_running_avgs.drop(columns = ['game_count'])



        # Create a DataFrame to store team profiles for each game
    new_profiles = []
    
    final_profiles = {}
    final_rosters = {}
    df_games = df_games.sort_values(by = ['GAME_DATE'], ascending = True)
    print("Generating rosters")
    
    for index, game_id in enumerate(df_games['gameId'].unique()):
        #print(game_id)
        teams = df_player_stats[df_player_stats['GAME_ID'] == game_id]['TEAM_ABBREVIATION'].unique()
        invalid = False
        rosters = []
        for team in teams:
            roster = get_top_12_players(df_games, df_player_stats, game_id, team)
            if roster.shape[0] == 0:
                invalid = True
            else:
                rosters.append(roster)
                
        if invalid:
            continue
                
        for i, team_id in enumerate(teams):

            game_date = df_games[df_games['gameId'] == game_id]['GAME_DATE'].iloc[0]
            top_players = rosters[i]
            #print(game_id, team_id)
            #print(top_players)
            tricode = team_id

            profile = {'gameId' : game_id, 'teamTricode' : tricode, 'date' : game_date}
            # Sort players within each team by stats
            p_index = 0
            for index, player in top_players.iterrows():
                p_index += 1
                for column in player.index.to_list():
                    if column in ['MIN_RANK','FGM_RANK','FGA_RANK','FG_PCT_RANK','FG3M_RANK','FG3A_RANK','FG3_PCT_RANK','FTM_RANK','FTA_RANK','FT_PCT_RANK','OREB_RANK','DREB_RANK','REB_RANK','AST_RANK','TOV_RANK','STL_RANK','BLK_RANK','BLKA_RANK','PF_RANK','PFD_RANK','PTS_RANK','PLUS_MINUS_RANK','NBA_FANTASY_PTS_RANK','DD2_RANK','TD3_RANK','WNBA_FANTASY_PTS_RANK','AVAILABLE_FLAG','running_avg_MIN','running_avg_FGM','running_avg_FGA','running_avg_FG3M','running_avg_FG3A','running_avg_FTM','running_avg_FTA','running_avg_OREB','running_avg_DREB','running_avg_REB','running_avg_AST','running_avg_TOV','running_avg_STL','running_avg_BLK','running_avg_BLKA','running_avg_PF','running_avg_PFD','running_avg_PTS','running_avg_PLUS_MINUS','running_avg_NBA_FANTASY_PTS','running_avg_DD2','running_avg_TD3','running_avg_WNBA_FANTASY_PTS']:

                        profile[f'player_{p_index}_{column}'] = player[column]
            for i in range(12-top_players.shape[0]):
                for column in player.index.to_list():
                    if column in ['MIN_RANK','FGM_RANK','FGA_RANK','FG_PCT_RANK','FG3M_RANK','FG3A_RANK','FG3_PCT_RANK','FTM_RANK','FTA_RANK','FT_PCT_RANK','OREB_RANK','DREB_RANK','REB_RANK','AST_RANK','TOV_RANK','STL_RANK','BLK_RANK','BLKA_RANK','PF_RANK','PFD_RANK','PTS_RANK','PLUS_MINUS_RANK','NBA_FANTASY_PTS_RANK','DD2_RANK','TD3_RANK','WNBA_FANTASY_PTS_RANK','AVAILABLE_FLAG','running_avg_MIN','running_avg_FGM','running_avg_FGA','running_avg_FG3M','running_avg_FG3A','running_avg_FTM','running_avg_FTA','running_avg_OREB','running_avg_DREB','running_avg_REB','running_avg_AST','running_avg_TOV','running_avg_STL','running_avg_BLK','running_avg_BLKA','running_avg_PF','running_avg_PFD','running_avg_PTS','running_avg_PLUS_MINUS','running_avg_NBA_FANTASY_PTS','running_avg_DD2','running_avg_TD3','running_avg_WNBA_FANTASY_PTS']:
                        profile[f'player_{i + top_players.shape[0] + 1}_{column}'] = 0


            # Flatten the sorted stats into a single vector
            #print(sorted_top_players)  # Include necessary stats
            new_profiles.append(profile)
    new_profiles = pd.DataFrame(new_profiles)
    if new_profiles.size == 0: continue
    new_profiles['gameId'] = new_profiles['gameId'].astype(int)
    new_profiles['date'] = pd.to_datetime(new_profiles['date'])

    if season == '2023-24':
        print("Generating current rosters")
        final_profiles = []
        teams = new_profiles.groupby('teamTricode')
        for i, team in teams:
            game_id = current_profiles[current_profiles['teamTricode'] == i].iloc[0]['gameId']
            print(game_id)

            profile = {'gameId' : game_id, 'teamTricode' : i}
            roster = get_current_roster(df_player_stats, i, injured_players)
            print(roster)
            p_i = 0
            for p_index, player in roster.iterrows():
                for column in player.index.to_list():
                    if column in ['MIN_RANK','FGM_RANK','FGA_RANK','FG_PCT_RANK','FG3M_RANK','FG3A_RANK','FG3_PCT_RANK','FTM_RANK','FTA_RANK','FT_PCT_RANK','OREB_RANK','DREB_RANK','REB_RANK','AST_RANK','TOV_RANK','STL_RANK','BLK_RANK','BLKA_RANK','PF_RANK','PFD_RANK','PTS_RANK','PLUS_MINUS_RANK','NBA_FANTASY_PTS_RANK','DD2_RANK','TD3_RANK','WNBA_FANTASY_PTS_RANK','AVAILABLE_FLAG','running_avg_MIN','running_avg_FGM','running_avg_FGA','running_avg_FG3M','running_avg_FG3A','running_avg_FTM','running_avg_FTA','running_avg_OREB','running_avg_DREB','running_avg_REB','running_avg_AST','running_avg_TOV','running_avg_STL','running_avg_BLK','running_avg_BLKA','running_avg_PF','running_avg_PFD','running_avg_PTS','running_avg_PLUS_MINUS','running_avg_NBA_FANTASY_PTS','running_avg_DD2','running_avg_TD3','running_avg_WNBA_FANTASY_PTS']:
                        profile[f'player_{p_i + 1}_{column}'] = player[column]
                p_i += 1
                print(p_i)
            for p_i in range(12-roster.shape[0]):
                for column in player.index.to_list():
                    if column in ['MIN_RANK','FGM_RANK','FGA_RANK','FG_PCT_RANK','FG3M_RANK','FG3A_RANK','FG3_PCT_RANK','FTM_RANK','FTA_RANK','FT_PCT_RANK','OREB_RANK','DREB_RANK','REB_RANK','AST_RANK','TOV_RANK','STL_RANK','BLK_RANK','BLKA_RANK','PF_RANK','PFD_RANK','PTS_RANK','PLUS_MINUS_RANK','NBA_FANTASY_PTS_RANK','DD2_RANK','TD3_RANK','WNBA_FANTASY_PTS_RANK','AVAILABLE_FLAG','running_avg_MIN','running_avg_FGM','running_avg_FGA','running_avg_FG3M','running_avg_FG3A','running_avg_FTM','running_avg_FTA','running_avg_OREB','running_avg_DREB','running_avg_REB','running_avg_AST','running_avg_TOV','running_avg_STL','running_avg_BLK','running_avg_BLKA','running_avg_PF','running_avg_PFD','running_avg_PTS','running_avg_PLUS_MINUS','running_avg_NBA_FANTASY_PTS','running_avg_DD2','running_avg_TD3','running_avg_WNBA_FANTASY_PTS']:
                        profile[f'player_{p_i + roster.shape[0] + 1}_{column}'] = 0

            final_profiles.append(profile)
        final_profiles = pd.DataFrame(final_profiles)
        print(final_profiles)
        current_profiles = pd.merge(current_profiles, final_profiles, on = ['gameId', 'teamTricode'])
        
        current_profiles.to_csv('current_profiles.csv') 
    new_profiles = new_profiles.drop(columns = ['date'])
    # Rearrange columns to have 'team', 'gameid', 'home', and 'time_between_games' at the beginning
    cols_to_order = ['gameId', 'playoff', 'teamTricode', 'time_between_games', 'game_count', 'date']
    new_columns = cols_to_order + [col for col in df_running_avgs.columns if col not in cols_to_order]
    df_running_avgs = df_running_avgs[new_columns]
    new_profiles = new_profiles.rename(columns = {'GAME_ID' : 'gameId', 'TEAM_ABBREVIATION' : 'teamTricode'})
    


    df_running_avgs = pd.merge(df_running_avgs, new_profiles, on=['gameId', 'teamTricode'], how='left')

    print("Concatenating")
    all_games = pd.concat([all_games, df_games]) 
    all_averages = pd.concat([all_averages, df_running_avgs], ignore_index=True)
    # Merging the team profiles with existing game data


# Continue with any additional preprocessing as in your existing code...


    # Save to CSV
    df_running_avgs.to_csv(f'{season}_averages.csv', index=False)
    print("Completed season: ", season)



all_games.to_csv('all_games.csv')
all_averages.to_csv('all_averages.csv')