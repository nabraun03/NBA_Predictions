from injuries import fetch_injured_players
from preprocessing import Preprocessor  # Assuming this is the function from your separate file
import pandas as pd
import numpy as np
from datetime import date
from travel import calculate_travel

class CurrentProfileGenerator:
    def __init__(self, seasons):
        p = Preprocessor(seasons, 50, 0, True)
        p25 = Preprocessor(seasons, 25, 0, False)
        p10 = Preprocessor(seasons, 10, 0, False)
        p5 = Preprocessor(seasons, 5, 0, False)
        self.games = p.games


        common_cols = list(set(p.team_stats.columns).intersection(set(p25.team_stats.columns)))


        p.team_stats = pd.merge(p.team_stats, p25.team_stats, on = common_cols, how = 'inner')
        p.team_stats = pd.merge(p.team_stats, p10.team_stats, on = common_cols, how = 'inner')
        p.team_stats = pd.merge(p.team_stats, p5.team_stats, on = common_cols, how='inner')
        self.team_stats = p.team_stats
        self.team_stats = pd.merge(self.team_stats, p.elo_ratings, on = ['teamTricode'], how = 'inner')
        assert 'elo' in self.team_stats.columns

        
        self.team_stats = self.team_stats.sort_values(by = 'date', ascending = False)
        self.player_stats = p.player_stats
        self.player_stats['fullName'] = self.player_stats['firstName'] + ' ' + self.player_stats['familyName']
        self.current_profiles = pd.DataFrame()
        self.injured_players = fetch_injured_players()

        if self.injured_players == None:
            self.injured_players = []
        
        print("Generating profiles")
        self.impute_current_teams()
        self.generate_current_profiles()

    def generate_current_profiles(self):
        teams = self.team_stats['teamTricode'].unique()
        team_profiles = []
        roster_profiles = []

        for team in teams:
            
            team_profile = self.get_current_team_profile(team)
            team_profile['time_between_games'] = (pd.to_datetime(date.today()) - pd.to_datetime(team_profile['date'])).dt.days
            team_profiles.append(team_profile)

            roster = self.get_current_roster(team)
            roster_profile = self.create_profile(team, roster)
            roster_profiles.append(roster_profile)

        team_profiles = pd.concat(team_profiles)
        print(team_profiles[['teamTricode', 'time_between_games']])
        team_profiles = self.generate_travel_statistics(team_profiles)
        roster_profiles = pd.DataFrame(roster_profiles)

        profiles = pd.merge(team_profiles, roster_profiles, on = ['teamTricode'], how = 'inner')
        self.current_profiles = pd.DataFrame(profiles)
    
    def get_current_team_profile(self, team):
        self.team_stats = self.team_stats.sort_values(by = 'date', ascending = False)
        print(self.team_stats[self.team_stats['teamTricode'] == team].head(1))
        return self.team_stats[self.team_stats['teamTricode'] == team].head(1)
    

    def get_current_roster(self, tricode):
        # Define a recent activity window (e.g., last 15 days)
        recent_date_threshold = pd.to_datetime('today') - pd.Timedelta(days=15)

        # Filter players with recent activity
        recent_active_players = self.player_stats[pd.to_datetime(self.player_stats['date']) >= recent_date_threshold]

        # Filter out players whose most recent game was not with the specified team
        most_recent_team_players = recent_active_players[recent_active_players['teamTricode'] == tricode]

        # Remove injured players
        active_roster = most_recent_team_players[~most_recent_team_players['fullName'].isin(self.injured_players)]
        
        active_roster = active_roster.groupby(['fullName']).apply(lambda x: x.sort_values(by='date', ascending=False).head(1)).reset_index(drop=True)

        # Select top 12 players based on average minutes played
        top_players = active_roster.sort_values(by='running_avg_minutes', ascending=False).head(12)
        
        print(top_players[['fullName', 'running_avg_minutes', 'running_avg_points', 'C', 'F', 'G']])
        return top_players

    def impute_current_teams(self):
        # Sort player stats by date in descending order to get the most recent entries first
        sorted_player_stats = self.player_stats.sort_values(by='date', ascending=False)

        # Group by player and get the first (most recent) non-null team
        most_recent_teams = sorted_player_stats.groupby(['firstName', 'familyName'])['teamTricode'].first().reset_index()

        # Merge the most recent teams with the player stats
        self.player_stats = self.player_stats.merge(most_recent_teams, on=['firstName', 'familyName'], how='left', suffixes=('', '_recent'))

        

        # Update the team column with the most recent non-null teams
        self.player_stats['teamTricode'] = self.player_stats['teamTricode_recent']

        # Drop the temporary column used for merging
        self.player_stats = self.player_stats.drop(columns=['teamTricode_recent'])

    
    def create_profile(self, team, roster):
        # Sort players by average minutes
        roster_sorted = roster.sort_values(by='running_avg_minutes', ascending=False)

        # Prepare the profile dictionary
        profile = {'teamTricode': team}

        count = 1
        nonstat_columns = ['gameId', 'teamTricode', 'firstName', 'familyName', 'position', 'date', 'fullName']
        for i, (index, row) in enumerate(roster_sorted.head(8).iterrows(), start=1):
            stats = row.index
            for stat in row.index:
                if stat not in nonstat_columns:
                    profile_key = f"player_{count}_{stat}"
                    profile[profile_key] = row[stat]
            count += 1
            


        return profile
    
    def generate_travel_statistics(self, team_profiles):
        for days in [1, 3, 5, 10]: 
            print(days) # The different periods you're interested in
            
            team_profiles[f'avg_travel_last_{days}_days'] = team_profiles.apply(lambda row: calculate_travel(row['teamTricode'], date.today(), days, self.games, True), axis=1)
        return team_profiles




    
# Example usage

seasons = ['2023-24', '2022-23', '2021-22', '2020-21', '2019-20', '2018-19', '2017-18', '2016-17']
profile_generator = CurrentProfileGenerator(seasons)
print(profile_generator.current_profiles)
profile_generator.current_profiles.to_csv('current_profiles.csv')

