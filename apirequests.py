import csv
import pandas as pd
from nba_api.stats.endpoints import leaguegamelog, boxscoreadvancedv3, boxscoretraditionalv3, boxscorehustlev2, boxscoremiscv3, boxscoreplayertrackv3, PlayerGameLogs
from nba_api.stats.static import teams
from datetime import date

team_dict = {team['id']: team['abbreviation'] for team in teams.get_teams()}

def fetch_player_logs(season):
    logs = PlayerGameLogs(season_nullable=season)
    return logs.get_data_frames()[0]


def fetch_league_game_logs(season):

    game_log = leaguegamelog.LeagueGameLog(season=season, season_type_all_star='Regular Season')
    playoffs_log = leaguegamelog.LeagueGameLog(season=season, season_type_all_star="Playoffs")
    return pd.concat([game_log.get_data_frames()[0], playoffs_log.get_data_frames()[0]])


def fetch_advanced_box_score(game_id):
    params = {
        'end_period': 1,
        'end_range': 0,
        'game_id': game_id,
        'range_type': 0,
        'start_period': 1,
        'start_range': 0
    }
    box_score = boxscoreadvancedv3.BoxScoreAdvancedV3(**params)
    team_stats = box_score.team_stats.get_data_frame()
    return team_stats

def fetch_traditional_box_score(game_id):
    params = {
        'end_period': 1,
        'end_range': 0,
        'game_id': game_id,
        'range_type': 0,
        'start_period': 1,
        'start_range': 0
    }
    box_score = boxscoretraditionalv3.BoxScoreTraditionalV3(**params)
    team_stats = box_score.team_stats.get_data_frame()
    return team_stats

def fetch_hustle_box_score(game_id):
    box_score = boxscorehustlev2.BoxScoreHustleV2(game_id=game_id)
    team_stats = box_score.team_stats.get_data_frame()
    return team_stats

def fetch_misc_box_score(game_id):
    params = {
        'end_period': 1,
        'end_range': 0,
        'game_id': game_id,
        'range_type': 0,
        'start_period': 1,
        'start_range': 0
    }
    box_score = boxscoremiscv3.BoxScoreMiscV3(**params)
    return box_score.team_stats.get_data_frame()

def fetch_track_box_score(game_id):
    box_score = boxscoreplayertrackv3.BoxScorePlayerTrackV3(game_id=game_id)
    team_stats = box_score.team_stats.get_data_frame()
    return team_stats

def process_and_merge(df):
    # Drop duplicates based on GAME_ID and TEAM_ID
    df = df.drop_duplicates(subset=['GAME_ID', 'TEAM_ID'])
    
    home_games = df[df['MATCHUP'].str.contains(' vs. ')].copy()
    away_games = df[df['MATCHUP'].str.contains(' @ ')].copy()
    
    home_games = home_games.rename(columns={
        'TEAM_ID': 'HOME_TEAM_ID',
        'TEAM_ABBREVIATION': 'HOME_TEAM_ABBREVIATION',
        'PTS': 'HOME_TEAM_PTS'
    })
    
    away_games = away_games.rename(columns={
        'TEAM_ID': 'AWAY_TEAM_ID',
        'TEAM_ABBREVIATION': 'AWAY_TEAM_ABBREVIATION',
        'PTS': 'AWAY_TEAM_PTS'
    })
    
    merged_df = pd.merge(home_games, away_games, on=['GAME_ID', 'GAME_DATE'], suffixes=('_home', '_away'))
    merged_df = merged_df[[
        'GAME_ID', 'GAME_DATE', 'HOME_TEAM_ABBREVIATION', 'HOME_TEAM_PTS',
        'AWAY_TEAM_ABBREVIATION', 'AWAY_TEAM_PTS'
    ]]
    merged_df = merged_df.rename(columns={'GAME_ID' : 'gameId'})
    print(date.today())
    
    merged_df['GAME_DATE'] = merged_df['GAME_DATE'].astype(str)
    merged_df['GAME_DATE'] = merged_df['GAME_DATE'].apply(lambda x: date(*map(int, x.split('-'))))

    print(merged_df.dtypes)
    print(merged_df['GAME_DATE'])
    print(merged_df[merged_df['GAME_DATE'] == date.today()])
    merged_df = merged_df[merged_df['GAME_DATE'] != date.today()] 
    return merged_df


def write_to_csv(df, season):
    df.to_csv(f"{season}_all_games.csv", index=False)



seasons = ['2023-24']
for season in seasons:

    league_game_log = fetch_league_game_logs(season)
    league_game_log['GAME_DATE'] = league_game_log['GAME_DATE'].apply(lambda x: date(*map(int, x.split('-'))))
    league_game_log = league_game_log[league_game_log['GAME_DATE'] != date.today()]
    game_ids = league_game_log['GAME_ID'].unique()
    print(league_game_log.columns)
    season_log = process_and_merge(league_game_log)
    write_to_csv(season_log, season)

    


    # Initialize empty dataframes to store all player and team stats
    advanced_stats = pd.DataFrame()
    traditional_stats = pd.DataFrame()
    hustle_stats = pd.DataFrame()
    misc_stats = pd.DataFrame()
    track_stats = pd.DataFrame()

    advanced_stats_list = []
    traditional_stats_list = []
    hustle_stats_list = []
    misc_stats_list = []
    track_stats_list = []

    try:
        existing_games_df = pd.read_csv(f'{season}_all_games.csv')
        existing_advanced_stats = pd.read_csv(f'{season}_advanced_stats.csv')
        existing_hustle_stats = pd.read_csv(f'{season}_hustle_stats.csv')
        existing_misc_stats = pd.read_csv(f'{season}_misc_stats.csv')
        existing_track_stats = pd.read_csv(f'{season}_track_stats.csv')
        existing_traditional_stats = pd.read_csv(f'{season}_traditional_stats.csv')
    except:
        continue

    # Loop through each game ID and fetch its advanced box score stats
    for i, game_id in enumerate(game_ids):
        try:
            print(f'{season}: {i}/{len(game_ids)}')
            
            
            print(int(game_id))
            existing_game = existing_track_stats[existing_track_stats['gameId'] == int(game_id)]
            if existing_game.size > 1:
                print('skipped')
                continue
            else:
                advanced_game_stats = fetch_advanced_box_score(game_id)
                advanced_stats_list.append(advanced_game_stats)

                traditional_game_stats = fetch_traditional_box_score(game_id)
                traditional_stats_list.append(traditional_game_stats)

                hustle_game_stats = fetch_hustle_box_score(game_id)
                hustle_stats_list.append(hustle_game_stats)

                misc_game_stats = fetch_misc_box_score(game_id)
                misc_stats_list.append(misc_game_stats)

                track_game_stats = fetch_track_box_score(game_id)
                track_stats_list.append(track_game_stats)

        except:
            advanced_game_stats = fetch_advanced_box_score(game_id)
            advanced_stats_list.append(advanced_game_stats)

            traditional_game_stats = fetch_traditional_box_score(game_id)
            traditional_stats_list.append(traditional_game_stats)

            hustle_game_stats = fetch_hustle_box_score(game_id)
            hustle_stats_list.append(hustle_game_stats)

            misc_game_stats = fetch_misc_box_score(game_id)
            misc_stats_list.append(misc_game_stats)

            track_game_stats = fetch_track_box_score(game_id)
            track_stats_list.append(track_game_stats)

        advanced_stats = pd.concat(advanced_stats_list, ignore_index=True)
        traditional_stats = pd.concat(traditional_stats_list, ignore_index = True)
        hustle_stats = pd.concat(hustle_stats_list, ignore_index=True)
        misc_stats = pd.concat(misc_stats_list, ignore_index=True)
        track_stats = pd.concat(track_stats_list, ignore_index=True)


    # Save the collected stats to CSV files
    if len(advanced_stats_list) > 0:
        advanced_stats = pd.concat(advanced_stats_list, ignore_index=True)
        traditional_stats = pd.concat(traditional_stats_list, ignore_index = True)
        hustle_stats = pd.concat(hustle_stats_list, ignore_index=True)
        misc_stats = pd.concat(misc_stats_list, ignore_index=True)
        track_stats = pd.concat(track_stats_list, ignore_index=True)

    try:
        advanced_stats = pd.concat([existing_advanced_stats, advanced_stats], ignore_index=True)
        traditional_stats = pd.concat([existing_traditional_stats, traditional_stats], ignore_index = True)
        hustle_stats = pd.concat([existing_hustle_stats, hustle_stats], ignore_index=True)
        misc_stats = pd.concat([existing_misc_stats, misc_stats], ignore_index=True)
        track_stats = pd.concat([existing_track_stats, track_stats], ignore_index=True)
    except:
        continue
            

    
    advanced_stats.to_csv(f'{season}_advanced_stats.csv', index=False)
    traditional_stats.to_csv(f'{season}_traditional_stats.csv', index = False)
    hustle_stats.to_csv(f'{season}_hustle_stats.csv', index = False)
    misc_stats.to_csv(f'{season}_misc_stats.csv', index = False)
    track_stats.to_csv(f'{season}_track_stats.csv', index = True)

    player_logs = fetch_player_logs(season)
    player_logs.to_csv(f'{season}_player_logs.csv')
