import pandas as pd
import numpy as np
from travel import calculate_travel
from datetime import date, datetime
from elo import elo

def convert_minutes_to_float(time_str):
    if type(time_str) != str:
        return time_str
    if ':' in time_str:
        minutes, seconds = map(int, time_str.split(':'))
        return minutes + seconds / 60
    else:
        return 0

class Preprocessor:

    def __init__(self, seasons, span, shift, full):
        self.seasons = seasons
        self.games = pd.DataFrame()
        print("Loading games")
        self.load_all_games()
        self.player_stats = pd.DataFrame()
        self.team_stats = pd.DataFrame()
        self.rosters = pd.DataFrame()
        self.span = span
        self.shift = shift
        self.current = (self.shift == 0)
        self.complete_profiles = pd.DataFrame()
        
        print("Loading team data")
        self.load_team_data()
        if full:
            
            if not self.current:
                print("Generating travel statistics")
                self.generate_travel_statistics()
            print("Generating elo")
            self.generate_elo()
            print("Loading player data")
            self.load_player_data()
            self.impute_historical_positions()
            print("Compiling rosters")
            self.generate_rosters()
            print("Compiling complete profiles")
            self.compile_complete_profiles()

        
    def load_all_games(self):
        seasons = []
        for season in self.seasons:
            df_games = pd.read_csv(f'{season}_all_games.csv')
            seasons.append(df_games)
        self.games = pd.concat(seasons)
        self.games['GAME_DATE'] = self.games['GAME_DATE'].apply(lambda x: date(*map(int, x.split('-'))))
    
    def load_player_data(self):
        seasons = []
        for season in self.seasons:
            df_games = pd.read_csv(f'{season}_all_games.csv')
            advanced_player_logs = pd.read_csv(f'{season}_advanced_player_stats.csv')[['gameId', 'teamTricode', 'firstName', 'familyName', 'minutes','estimatedOffensiveRating','offensiveRating','estimatedDefensiveRating','defensiveRating','estimatedNetRating','netRating','assistPercentage','assistToTurnover','assistRatio','offensiveReboundPercentage','defensiveReboundPercentage','reboundPercentage','turnoverRatio','effectiveFieldGoalPercentage','trueShootingPercentage','usagePercentage','estimatedUsagePercentage','estimatedPace','pace','pacePer40','possessions','PIE']]
            hustle_player_logs = pd.read_csv(f'{season}_hustle_player_stats.csv')[['gameId', 'teamTricode','firstName', 'familyName', 'position', 'points','contestedShots','contestedShots2pt','contestedShots3pt','deflections','chargesDrawn','screenAssists','screenAssistPoints','looseBallsRecoveredOffensive','looseBallsRecoveredDefensive','looseBallsRecoveredTotal','offensiveBoxOuts','defensiveBoxOuts','boxOutPlayerTeamRebounds','boxOutPlayerRebounds','boxOuts']]
            misc_player_logs = pd.read_csv(f'{season}_misc_player_stats.csv')[['gameId', 'teamTricode', 'firstName', 'familyName', 'pointsOffTurnovers','pointsSecondChance','pointsFastBreak','pointsPaint','oppPointsOffTurnovers','oppPointsSecondChance','oppPointsFastBreak','oppPointsPaint','blocksAgainst','foulsDrawn']]
            traditional_player_logs = pd.read_csv(f'{season}_traditional_player_stats.csv')[['gameId', 'teamTricode', 'firstName', 'familyName', 'fieldGoalsMade','fieldGoalsAttempted','fieldGoalsPercentage','threePointersMade','threePointersAttempted','threePointersPercentage','freeThrowsMade','freeThrowsAttempted','freeThrowsPercentage','reboundsOffensive','reboundsDefensive','reboundsTotal','assists','steals','blocks','turnovers','foulsPersonal','plusMinusPoints']]
            track_player_logs = pd.read_csv(f'{season}_track_player_stats.csv')[['gameId', 'teamTricode', 'firstName', 'familyName', 'speed','distance','reboundChancesOffensive','reboundChancesDefensive','reboundChancesTotal','touches','secondaryAssists','freeThrowAssists','passes','contestedFieldGoalsMade','contestedFieldGoalsAttempted','contestedFieldGoalPercentage','uncontestedFieldGoalsMade','uncontestedFieldGoalsAttempted','uncontestedFieldGoalsPercentage','defendedAtRimFieldGoalsMade','defendedAtRimFieldGoalsAttempted','defendedAtRimFieldGoalPercentage']]
                    

            merge_columns = ['gameId', 'teamTricode', 'firstName', 'familyName']
            merged_df = pd.merge(advanced_player_logs, hustle_player_logs, on=merge_columns, how='inner')
            merged_df = pd.merge(merged_df, misc_player_logs, on=merge_columns, how='inner')
            merged_df = pd.merge(merged_df, traditional_player_logs, on=merge_columns, how='inner')
            merged_df = pd.merge(merged_df, track_player_logs, on=merge_columns, how='inner')

            merged_df = pd.merge(merged_df, df_games[['gameId', 'GAME_DATE']], on = 'gameId', how = 'left')
            merged_df['date'] = merged_df['GAME_DATE'].apply(lambda x: date(*map(int, x.split('-'))))
            merged_df = merged_df.drop(columns = ['GAME_DATE'])

            seasons.append(merged_df)
            
        combined = pd.concat(seasons)

        self.player_stats = self.preprocess_player_data(combined)
    
    def load_team_data(self):
        seasons = []
        for season in self.seasons:
            df_games = pd.read_csv(f'{season}_all_games.csv')
            df_advanced = pd.read_csv(f'{season}_advanced_stats.csv')[['gameId', 'teamTricode', 'estimatedOffensiveRating', 'offensiveRating', 'estimatedDefensiveRating', 'defensiveRating', 'estimatedNetRating', 'netRating','assistPercentage','assistToTurnover','assistRatio','offensiveReboundPercentage','defensiveReboundPercentage','reboundPercentage','turnoverRatio','effectiveFieldGoalPercentage','trueShootingPercentage','usagePercentage','estimatedUsagePercentage', 'estimatedPace', 'pace', 'pacePer40', 'possessions', 'PIE']]
            df_basic = pd.read_csv(f'{season}_traditional_stats.csv')[['gameId', 'teamTricode', 'fieldGoalsMade', 'fieldGoalsAttempted', 'fieldGoalsPercentage', 'threePointersMade', 'threePointersAttempted', 'threePointersPercentage', 'freeThrowsMade', 'freeThrowsAttempted', 'freeThrowsPercentage', 'reboundsOffensive', 'reboundsDefensive', 'reboundsTotal', 'assists', 'steals', 'blocks', 'turnovers', 'foulsPersonal', 'points', 'plusMinusPoints']]
            df_hustle = pd.read_csv(f'{season}_hustle_stats.csv')[['gameId', 'teamTricode', 'contestedShots', 'contestedShots2pt', 'contestedShots3pt', 'deflections', 'chargesDrawn', 'screenAssists', 'screenAssistPoints', 'looseBallsRecoveredOffensive', 'looseBallsRecoveredDefensive', 'looseBallsRecoveredTotal', 'offensiveBoxOuts', 'defensiveBoxOuts', 'boxOutPlayerTeamRebounds', 'boxOutPlayerRebounds', 'boxOuts']]
            df_misc = pd.read_csv(f'{season}_misc_stats.csv')[['gameId', 'teamTricode', 'pointsOffTurnovers', 'pointsSecondChance', 'pointsFastBreak', 'pointsPaint', 'oppPointsOffTurnovers', 'oppPointsSecondChance', 'oppPointsFastBreak', 'oppPointsPaint', 'blocksAgainst', 'foulsDrawn']]
            df_tracking = pd.read_csv(f'{season}_track_stats.csv')[['gameId', 'teamTricode', 'distance', 'reboundChancesOffensive', 'reboundChancesDefensive', 'reboundChancesTotal', 'touches', 'secondaryAssists', 'freeThrowAssists', 'passes', 'contestedFieldGoalsMade', 'contestedFieldGoalsAttempted', 'contestedFieldGoalPercentage', 'uncontestedFieldGoalsMade', 'uncontestedFieldGoalsAttempted', 'uncontestedFieldGoalsPercentage', 'defendedAtRimFieldGoalsMade', 'defendedAtRimFieldGoalsAttempted', 'defendedAtRimFieldGoalPercentage']]

            merge_columns = ['gameId', 'teamTricode']
            merged_df = pd.merge(df_advanced, df_basic, on = merge_columns, how = 'inner')
            merged_df = pd.merge(merged_df, df_hustle, on = merge_columns, how = 'inner')
            merged_df = pd.merge(merged_df, df_misc, on = merge_columns, how = 'inner')
            merged_df = pd.merge(merged_df, df_tracking, on = merge_columns, how = 'inner')
            merged_df = merged_df.drop_duplicates()



            merged_df = pd.merge(merged_df, df_games[['gameId', 'GAME_DATE']], on = 'gameId', how = 'left')
            merged_df['date'] = merged_df['GAME_DATE'].apply(lambda x: date(*map(int, x.split('-'))))
            merged_df = merged_df.drop(columns = ['GAME_DATE'])

            df_games['winner'] = np.where(df_games['HOME_TEAM_PTS'] > df_games['AWAY_TEAM_PTS'], df_games['HOME_TEAM_ABBREVIATION'], df_games['AWAY_TEAM_ABBREVIATION'])

            merged_df = pd.merge(merged_df, df_games[['gameId', 'winner', 'HOME_TEAM_ABBREVIATION', 'AWAY_TEAM_ABBREVIATION']], on='gameId', how='left')

            merged_df = merged_df.drop_duplicates(subset = ['gameId', 'teamTricode'])
            processed_df = self.preprocess_team_data(merged_df)

            seasons.append(processed_df)

        self.team_stats = pd.concat(seasons)
    
    def preprocess_team_data(self, df):

        grouped = df.groupby('teamTricode')
        modified_groups = []
        for name, group in grouped:
            group['game_count'] = group['gameId'].expanding().count().shift(self.shift).fillna(0)
            group['time_between_games'] = group['date'].diff().dt.days
            group['playoff'] = (group['game_count'] > 82).astype(int)

            group = self.generate_team_running_averages(group)
            group = self.generate_winning_percentages(group)
            group = self.generate_streak(group)
            
            
            modified_groups.append(group)
        
        return pd.concat(modified_groups)

    def generate_winning_percentages(self, group):
        # Calculate overall winning percentage
        group['win'] = (group['teamTricode'] == group['winner']).astype(int)
        col_name = f'winning_percentage_last_{self.span}'
        group[col_name] = group['win'].rolling(window=self.span, min_periods=1).mean().shift(self.shift)

        
        # Calculate winning percentage for home and away games
        home_games = group[group['teamTricode'] == group['HOME_TEAM_ABBREVIATION']]
        home_games[f'home_{col_name}'] = home_games['win'].rolling(window=self.span, min_periods=0).mean()
        
        away_games = group[group['teamTricode'] == group['AWAY_TEAM_ABBREVIATION']]
        away_games[f'away_{col_name}'] = away_games['win'].rolling(window=self.span, min_periods=0).mean()
        
        # Merge back the calculated home and away percentages to the main dataframe
        group = group.merge(home_games[['gameId', f'home_{col_name}']], on='gameId', how='left')
        group = group.merge(away_games[['gameId', f'away_{col_name}']], on='gameId', how='left')

        # Forward fill the NaN values in home and away winning percentages
        group[f'home_{col_name}'] = group[f'home_{col_name}'].ffill().shift(self.shift)
        group[f'away_{col_name}'] = group[f'away_{col_name}'].ffill().shift(self.shift)



        # Fill remaining NaN values with 0 (for the start of the series)
        group[[f'home_{col_name}', f'away_{col_name}']] = group[[f'home_{col_name}', f'away_{col_name}']].fillna(0)
        
        return group
    
    def calculate_streak(self, group, home_or_away):
    # Initialize streak column
        
        group[f'{home_or_away.lower()}_streak'] = np.nan
        current_streak = 0
        
        # Select only the relevant games
        relevant_games = group[group['teamTricode'] == group[f'{home_or_away}_TEAM_ABBREVIATION']]
        
        # Calculate the streaks
        for game in relevant_games.itertuples():
            if current_streak < 1:
                if game.win:
                    current_streak = 1
                else:
                    current_streak -= 1
            else:
                if game.win:
                    current_streak += 1
                else:
                    current_streak = -1
            
            # Use .at for a more efficient assignment
            group.at[game.Index, f'{home_or_away.lower()}_streak'] = current_streak
        
        # Shift the streaks to avoid lookahead bias
        group[f'{home_or_away.lower()}_streak'] = group[f'{home_or_away.lower()}_streak'].bfill().shift(self.shift).fillna(current_streak)


        return group

        
    def generate_streak(self, group):
        group['streak'] = 0
        current_streak = 0
        for i, row in group.iterrows():
            
            if current_streak < 1:
                if row['win']:
                    current_streak = 1
                else:
                    current_streak -= 1
            else:
                if row['win']:
                    current_streak += 1
                else:
                    current_streak = -1
            
            group.at[i, 'streak'] = current_streak
        group['streak'] = group['streak'].shift(self.shift).fillna(0)
        group = self.calculate_streak(group, 'HOME')
        group = self.calculate_streak(group, 'AWAY')
        group = group.drop(columns = ['win', 'winner', 'HOME_TEAM_ABBREVIATION', 'AWAY_TEAM_ABBREVIATION'])
        return group
        

    def generate_travel_statistics(self):
        
        for days in [1, 3, 5, 10]:  # The different periods you're interested in
            self.team_stats[f'avg_travel_last_{days}_days'] = self.team_stats.apply(lambda row: calculate_travel(row['teamTricode'], row['date'], days, self.games, self.current), axis=1)
        self.team_stats[['gameId', 'teamTricode', 'date', 'avg_travel_last_1_days', 'avg_travel_last_3_days', 'avg_travel_last_5_days', 'avg_travel_last_10_days']].to_csv('travel_data.csv')

        """
        try:
            travel_data = pd.read_csv('travel_data.csv')
            self.team_stats = pd.merge(self.team_stats, travel_data[['gameId', 'teamTricode', 'avg_travel_last_1_days', 'avg_travel_last_3_days', 'avg_travel_last_5_days', 'avg_travel_last_10_days']], on = ['gameId', 'teamTricode'], how = 'inner')
            for i in range(100):
                print(self.team_stats.shape)

            

        except:
        """

    def generate_elo(self):
        if self.current:
            self.team_stats, self.elo_ratings = elo(self.games, self.team_stats, self.current)
        else:
            self.team_stats, _ = elo(self.games, self.team_stats, self.current)
            assert 'elo' in list(self.team_stats.columns)
        
    
    



    def generate_team_running_averages(self, group):
        percentage_columns = ['assistPercentage','assistToTurnover','assistRatio','offensiveReboundPercentage','defensiveReboundPercentage','reboundPercentage','turnoverRatio','effectiveFieldGoalPercentage','trueShootingPercentage','usagePercentage','estimatedUsagePercentage', 'fieldGoalsPercentage', 'threePointersPercentage', 'freeThrowsPercentage', 'contestedFieldGoalPercentage', 'uncontestedFieldGoalsPercentage', 'defendedAtRimFieldGoalPercentage']
        averaging_columns = [col for col in group.columns if col not in ['teamTricode', 'gameId',  'date','game_count', 'time_between_games', 'playoff', 'winning_percentage', 'home_winning_percentage', 'away_winning_percentage', 'home_streak', 'streak', 'away_streak', 'win', 'winner', 'HOME_TEAM_ABBREVIATION', 'AWAY_TEAM_ABBREVIATION']]
        for col in averaging_columns:
            
            running_avg_col_name = f'running_avg_{col}_last_{self.span}'

            group[running_avg_col_name] = group[col].ewm(span=self.span, min_periods=1).mean().shift(self.shift)
        group = group.drop(columns = averaging_columns)
        return group

    def preprocess_player_data(self, df):
        df = df.sort_values(by = ['date'], ascending = True)
        df['minutes'] = df['minutes'].apply(convert_minutes_to_float)
        df = df[df['minutes'] > 0]
        grouped = df.groupby(['firstName', 'familyName'])
        modified_groups = []
        for name, group in grouped:

            group = self.generate_player_running_averages(group)
            modified_groups.append(group)

        return pd.concat(modified_groups)
    
    def generate_player_running_averages(self, group):
        averaging_columns = [col for col in group.columns if col not in ['gameId', 'teamTricode', 'firstName', 'familyName', 'date', 'position']]

        new_columns = {}
        for col in averaging_columns:
            for span_factor in [1, 2, 10]:
                running_avg_col_name = f'running_avg_{col}_{int(self.span/span_factor)}'
                # Perform the calculation and store the result in the container
                new_columns[running_avg_col_name] = group[col].ewm(span=int(self.span/span_factor), min_periods=1).mean().shift(self.shift)

        # Create a new DataFrame from the container
        new_columns_df = pd.DataFrame(new_columns, index=group.index)

        # Concatenate the new DataFrame with the original group DataFrame
        group = pd.concat([group, new_columns_df], axis=1)

        # Drop the original averaging columns
        group = group.drop(columns=averaging_columns)

        return group

    def generate_rosters(self):
        all_rosters = []
        if not self.current:
            for season in self.seasons:
                print(f'Compiling stats for season {season}')
                games = pd.read_csv(f'{season}_all_games.csv')
                generator = RosterGenerator(games, self.player_stats, self.team_stats)
                rosters = generator.rosters


                all_rosters.append(rosters)
        else:
            games = pd.read_csv('2023-24_all_games.csv')
            generator = RosterGenerator(games, self.player_stats, self.team_stats)
            rosters = generator.rosters


            all_rosters.append(rosters)

        self.rosters = pd.concat(all_rosters)
    
    def impute_historical_positions(self):
    # Sort player stats by date to ensure chronological order
        sorted_player_stats = self.player_stats.sort_values(by=['firstName', 'familyName', 'date'], ascending=True)

        # Initialize a DataFrame to store running frequencies
        running_frequencies = pd.DataFrame()

        # Process each player individually
        for (firstName, familyName), group in sorted_player_stats.groupby(['firstName', 'familyName']):
            # One-hot encode positions for each game
            position_dummies = pd.get_dummies(group['position']).reindex(columns=['G', 'F', 'C'], fill_value=0)

            # Apply a rolling window of 50 games and calculate the mean to get the running frequency
            # Use min_periods=1 to ensure we get values even if there are less than 50 games
            running_avg = position_dummies.rolling(window=50, min_periods=1).mean().shift(self.shift).fillna(0)

            # Add player name back to the running average DataFrame
            running_avg['firstName'] = firstName
            running_avg['familyName'] = familyName
            running_avg['gameId'] = group['gameId']  # Ensure gameId is included for merging

            # Append the results to the running frequencies DataFrame
            running_frequencies = pd.concat([running_frequencies, running_avg], ignore_index=True)

        running_frequencies.fillna(0)
        # Merge the running frequencies back to the original player stats DataFrame on gameId
        self.player_stats = pd.merge(self.player_stats, running_frequencies, on=['gameId', 'firstName', 'familyName'], how='left')

    
    def compile_complete_profiles(self):
        self.complete_profiles = pd.merge(self.team_stats, self.rosters, on = ['gameId', 'teamTricode', 'date'], how = 'inner')
        self.complete_profiles = self.complete_profiles.drop_duplicates(subset = ['gameId', 'teamTricode'])
    


class RosterGenerator:

    def __init__(self, games, player_stats, team_stats):
        self.games = games
        self.player_stats = player_stats
        
        self.team_stats = team_stats
        self.rosters = pd.DataFrame()
        self.generate_rosters()

    def generate_rosters(self):
        profiles = []

        for gameId in self.games['gameId'].unique():
            game_date = self.team_stats[self.team_stats['gameId'] == gameId]['date'].iloc[0]
            teams = self.team_stats[self.team_stats['gameId'] == gameId]['teamTricode'].unique()

            for team in teams:
                roster = self.get_roster(gameId, team)
                
                roster = roster.sort_values(by=['running_avg_minutes_50'], ascending=False)

                # Prepare the profile dictionary
                profile = {'gameId': gameId, 'teamTricode': team, 'date': game_date}

                # Add starters to profile
                nonstat_columns = ['gameId', 'teamTricode', 'firstName', 'familyName', 'position', 'date']
                count = 1
                for index, row in roster.head(8).iterrows():
                    stats = row.index
                    for stat in row.index:
                        if stat not in nonstat_columns:
                            profile_key = f"player_{count}_{stat}"
                            profile[profile_key] = row[stat]
                    count += 1
                profiles.append(profile)

        # Convert the list of dictionaries to DataFrame
        self.rosters = pd.DataFrame(profiles)
        
    def get_roster(self, gameId, team):
        # Filter players who played in the specified game and belong to the specified team
        game_players = self.player_stats[(self.player_stats['gameId'] == gameId) & 
                                        (self.player_stats['teamTricode'] == team)]

        # Identify starters (players with assigned positions)
        starters = game_players[game_players['position'].isin(['G', 'F', 'C'])]

        # Identify non-starters
        non_starters = game_players[~game_players.index.isin(starters.index)]

        # Sort non-starters by average minutes played
        sorted_non_starters = non_starters.sort_values(by='running_avg_minutes_50', ascending=False)

        # Select top 7 non-starters
        top_non_starters = sorted_non_starters.head(7)

        # Combine starters and top non-starters
        roster = pd.concat([starters, top_non_starters])

        return roster
    
    
    


                 

if __name__ == "__main__":
    seasons = ['2023-24', '2022-23', '2021-22', '2020-21', '2019-20', '2018-19', '2017-18', '2016-17']
    p = Preprocessor(seasons, 50, 1, full=True)
    p25 = Preprocessor(seasons, 25, 1, False)
    p10 = Preprocessor(seasons, 10, 1, False)
    p5 = Preprocessor(seasons, 5, 1, False)
    p3 = Preprocessor(seasons, 3, 1, False)


    common_cols = list(set(p.complete_profiles.columns).intersection(set(p25.team_stats.columns)))

    p.complete_profiles = pd.merge(p.complete_profiles, p25.team_stats, on = common_cols, how = 'inner')
    p.complete_profiles = pd.merge(p.complete_profiles, p10.team_stats, on = common_cols, how = 'inner')
    p.complete_profiles = pd.merge(p.complete_profiles, p5.team_stats, on = common_cols, how='inner')

    print("Processing complete, saving data")
    try:
        p.games.to_csv('all_games.csv')
        p.team_stats.to_csv('all_team_averages.csv')
        p.player_stats.to_csv('all_player_averages.csv')
        p.complete_profiles.to_csv('all_averages.csv')
    except PermissionError as e:
        print(f"Caught {e}, saving to backup files")
        p.games.to_csv('backup_all_games.csv')
        p.team_stats.to_csv('backup_all_team_averages')
        p.player_stats.to_csv('backup_all_player_averages.csv')
        p.complete_profiles.to_csv('backup_all_averages.csv')
        