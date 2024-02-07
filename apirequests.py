import pandas as pd
from nba_api.stats.endpoints import (leaguegamelog, boxscoreadvancedv3, 
                                     boxscoretraditionalv3, boxscorehustlev2, 
                                     boxscoremiscv3, boxscoreplayertrackv3, 
                                     PlayerGameLogs, commonteamroster)
from nba_api.stats.static import teams
from datetime import date
import time
from requests.exceptions import ReadTimeout
from json.decoder import JSONDecodeError

class NBADataFetcher:

    def __init__(self, season):
        self.season = season
        self.team_dict = {team['id']: team['abbreviation'] for team in teams.get_teams()}

        self.existing_games_df = pd.DataFrame()
        self.existing_advanced_stats = pd.DataFrame()
        self.existing_hustle_stats = pd.DataFrame()
        self.existing_misc_stats = pd.DataFrame()
        self.existing_track_stats = pd.DataFrame()
        self.existing_traditional_stats = pd.DataFrame()
        self.existing_advanced_player_stats = pd.DataFrame()
        self.existing_hustle_player_stats = pd.DataFrame()
        self.existing_misc_player_stats = pd.DataFrame()
        self.existing_track_player_stats = pd.DataFrame()
        self.existing_traditional_player_stats = pd.DataFrame()



    
    def fetch_player_logs(self):
        logs = PlayerGameLogs(season_nullable=self.season)
        return logs.get_data_frames()[0]


    def fetch_league_game_logs(self):

        game_log = leaguegamelog.LeagueGameLog(season=self.season, season_type_all_star='Regular Season')
        playoffs_log = leaguegamelog.LeagueGameLog(season=self.season, season_type_all_star="Playoffs")
        return pd.concat([game_log.get_data_frames()[0], playoffs_log.get_data_frames()[0]])

    def fetch_box_score(self, game_id, data_type):
        max_retries = 10
        wait_seconds = 0.1
        for attempt in range(max_retries):
            try:
                if data_type in ['advanced', 'traditional', 'misc']:
                    params = {
                        'end_period': 1,
                        'end_range': 0,
                        'game_id': game_id,
                        'range_type': 0,
                        'start_period': 1,
                        'start_range': 0
                    }

                    if data_type == 'advanced':
                        box_score = boxscoreadvancedv3.BoxScoreAdvancedV3(**params)
                    elif data_type == 'traditional':
                        box_score = boxscoretraditionalv3.BoxScoreTraditionalV3(**params)
                    else:
                        box_score = boxscoremiscv3.BoxScoreMiscV3(**params)
                    
                else:
                    if data_type == 'hustle':
                        box_score = boxscorehustlev2.BoxScoreHustleV2(game_id=game_id)
                    else:
                        box_score = box_score = boxscoreplayertrackv3.BoxScorePlayerTrackV3(game_id=game_id)
                
                team_stats = box_score.team_stats.get_data_frame()
                player_stats = box_score.player_stats.get_data_frame()
                return team_stats, player_stats
            except ReadTimeout as e:
                if attempt < max_retries - 1:
                    print(f"Timeout occurred. Retrying in {wait_seconds} seconds... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_seconds)
                else:
                    print("Maximum retries reached. Unable to fetch data.")
                    raise
            except JSONDecodeError as e:
                if attempt < max_retries - 1:
                    print(f"JSONDecodeError occurred. Retrying in {wait_seconds} seconds... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_seconds)
                else:
                    print("Fetching failed, skipping game")
                    raise
            
        

    def process_game_logs(self, game_logs):
        game_logs['GAME_DATE'] = game_logs['GAME_DATE'].apply(lambda x: date(*map(int, x.split('-'))))
        game_logs = game_logs[game_logs['GAME_DATE'] != date.today()]

        game_logs = game_logs.drop_duplicates(subset=['GAME_ID', 'TEAM_ID'])
    
        home_games = game_logs[game_logs['MATCHUP'].str.contains(' vs. ')].copy()
        away_games = game_logs[game_logs['MATCHUP'].str.contains(' @ ')].copy()
        
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

        return merged_df
    
    def find_existing_data(self):
        data_types = ['games', 'advanced', 'hustle', 'misc', 'track', 
                      'traditional', 'advanced_player', 'hustle_player', 
                      'misc_player', 'track_player', 'traditional_player']

        for data_type in data_types:
            try:
                existing_data = pd.read_csv(f'{self.season}_{data_type}_stats.csv')
                setattr(self, f'existing_{data_type}_stats', existing_data)
            except FileNotFoundError:
                setattr(self, f'existing_{data_type}_stats', pd.DataFrame())


    def data_exists(self, game_id, data_type, team_or_player):

        if team_or_player == 'team':
            existing_df = getattr(self, f'existing_{data_type}_stats')
        else:
            existing_df = getattr(self, f'existing_{data_type}_player_stats')

        return not existing_df.empty and int(game_id) in existing_df['gameId'].unique()
    
    
    def concatenate_and_save(self, data_type, new_data_list, team_or_player):
        # Determine the correct existing data based on data_type and team_or_player
        if team_or_player == 'team':
            existing_data = getattr(self, f'existing_{data_type}_stats')
        else:
            existing_data = getattr(self, f'existing_{data_type}_player_stats')
        

        # Concatenate with existing data if any
        if not existing_data.empty and len(new_data_list) != 0:
            combined_data = pd.concat([existing_data, pd.concat(new_data_list, ignore_index=True)], ignore_index=True)
        elif len(new_data_list) == 0:
            combined_data = existing_data
        else:
            combined_data = pd.concat(new_data_list, ignore_index=True)

        # Save data to CSV
        if team_or_player == 'team':
            combined_data.to_csv(f'{self.season}_{data_type}_stats.csv', index=False)
        else:
            combined_data.to_csv(f'{self.season}_{data_type}_player_stats.csv', index = False)



    
    def fetch_and_save_all_data(self):
        i = 0
        game_log = self.fetch_league_game_logs()
        game_log = self.process_game_logs(game_log)
        game_log.to_csv(f"{self.season}_all_games.csv", index=False)
        game_ids = game_log['gameId'].unique()

        self.find_existing_data()

        advanced_stats_list, hustle_stats_list, misc_stats_list, track_stats_list, traditional_stats_list = [], [], [], [], []
        advanced_player_stats_list, hustle_player_stats_list, misc_player_stats_list, track_player_stats_list, traditional_player_stats_list = [], [], [], [], []

        for gameId in game_ids:
            i += 1
            if i % 10 == 0: print(f'{i} / {len(game_ids)}')
            try:
                advanced_team_stats_exist, advanced_player_stats_exist = self.data_exists(gameId, 'advanced', 'team'), self.data_exists(gameId, 'advanced', 'player')
                if not (advanced_team_stats_exist and advanced_player_stats_exist):
                    advanced_box_score, advanced_player_box_score = self.fetch_box_score(gameId, 'advanced')
                    if not advanced_team_stats_exist:
                        advanced_stats_list.append(advanced_box_score)
                    if not advanced_player_stats_exist:
                        advanced_player_stats_list.append(advanced_player_box_score)

                hustle_team_stats_exist, hustle_player_stats_exist = self.data_exists(gameId, 'hustle', 'team'), self.data_exists(gameId, 'hustle', 'player')
                if not (hustle_team_stats_exist and hustle_player_stats_exist):
                    hustle_box_score, hustle_player_box_score = self.fetch_box_score(gameId, 'hustle')
                    if not hustle_team_stats_exist:
                        hustle_stats_list.append(hustle_box_score)
                    if not hustle_player_stats_exist:
                        hustle_player_stats_list.append(hustle_player_box_score)
                
                misc_team_stats_exist, misc_player_stats_exist = self.data_exists(gameId, 'misc', 'team'), self.data_exists(gameId, 'misc', 'player')
                if not (misc_team_stats_exist and misc_player_stats_exist):
                    misc_box_score, misc_player_box_score = self.fetch_box_score(gameId, 'misc')
                    if not misc_team_stats_exist:
                        misc_stats_list.append(misc_box_score)
                    if not misc_player_stats_exist:
                        misc_player_stats_list.append(misc_player_box_score)
                
                track_team_stats_exist, track_player_stats_exist = self.data_exists(gameId, 'track', 'team'), self.data_exists(gameId, 'track', 'player')
                if not (track_team_stats_exist and track_player_stats_exist):
                    track_box_score, track_player_box_score = self.fetch_box_score(gameId, 'track')
                    if not track_team_stats_exist:
                        track_stats_list.append(track_box_score)
                    if not track_player_stats_exist:
                        track_player_stats_list.append(track_player_box_score)

                traditional_team_stats_exist, traditional_player_stats_exist = self.data_exists(gameId, 'traditional', 'team'), self.data_exists(gameId, 'traditional', 'player')
                if not (traditional_team_stats_exist and traditional_player_stats_exist):
                    traditional_box_score, traditional_player_box_score = self.fetch_box_score(gameId, 'traditional')
                    if not traditional_team_stats_exist:
                        traditional_stats_list.append(traditional_box_score)
                    if not traditional_player_stats_exist:
                        traditional_player_stats_list.append(traditional_player_box_score)

            except KeyboardInterrupt:
                print("Caught KeyboardInterrupt, saving files before termination")
                self.concatenate_and_save('advanced', advanced_stats_list, 'team')
                self.concatenate_and_save('advanced', advanced_player_stats_list, 'player')
                self.concatenate_and_save('hustle', hustle_stats_list, 'team')
                self.concatenate_and_save('hustle', hustle_player_stats_list, 'player')
                self.concatenate_and_save('misc', misc_stats_list, 'team')
                self.concatenate_and_save('misc', misc_player_stats_list, 'player')
                self.concatenate_and_save('track', track_stats_list, 'team')
                self.concatenate_and_save('track', track_player_stats_list, 'player')
                self.concatenate_and_save('traditional', traditional_stats_list, 'team')
                self.concatenate_and_save('traditional', traditional_player_stats_list, 'player')
                raise

            
            except JSONDecodeError as e:
                print("Failed to fetch data, skipping game")
                continue            
            except Exception as e:
                print(f"Caught unexpected exception {e}, saving files")
                self.concatenate_and_save('advanced', advanced_stats_list, 'team')
                self.concatenate_and_save('advanced', advanced_player_stats_list, 'player')
                self.concatenate_and_save('hustle', hustle_stats_list, 'team')
                self.concatenate_and_save('hustle', hustle_player_stats_list, 'player')
                self.concatenate_and_save('misc', misc_stats_list, 'team')
                self.concatenate_and_save('misc', misc_player_stats_list, 'player')
                self.concatenate_and_save('track', track_stats_list, 'team')
                self.concatenate_and_save('track', track_player_stats_list, 'player')
                self.concatenate_and_save('traditional', traditional_stats_list, 'team')
                self.concatenate_and_save('traditional', traditional_player_stats_list, 'player')
                raise


        print(f"Data fetching successful, saving data for season {self.season}")
        self.concatenate_and_save('advanced', advanced_stats_list, 'team')
        self.concatenate_and_save('advanced', advanced_player_stats_list, 'player')
        self.concatenate_and_save('hustle', hustle_stats_list, 'team')
        self.concatenate_and_save('hustle', hustle_player_stats_list, 'player')
        self.concatenate_and_save('misc', misc_stats_list, 'team')
        self.concatenate_and_save('misc', misc_player_stats_list, 'player')
        self.concatenate_and_save('track', track_stats_list, 'team')
        self.concatenate_and_save('track', track_player_stats_list, 'player')
        self.concatenate_and_save('traditional', traditional_stats_list, 'team')
        self.concatenate_and_save('traditional', traditional_player_stats_list, 'player')

if __name__ == '__main__':
    seasons = ['2023-24', '2022-23', '2021-22', '2020-21', '2019-20', '2018-19', '2017-18', '2016-17']
    for season in seasons:
        print(season)
        data_fetcher = NBADataFetcher(season)
        data_fetcher.fetch_and_save_all_data()


            

                











    


        

            