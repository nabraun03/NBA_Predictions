from geopy.distance import geodesic
from geopy.geocoders import Nominatim
import time
import csv
from datetime import date
from games import fetch_games_today

class TravelCollector:
    
    def __init__(self):
        self.team_to_stadium = {'PHI' : 'Wells Fargo Center, Philadelphia, Pennsylvania',
                                'BOS' : 'TD Garden, Boston, Massachusetts',
                                'LAL' : 'Crypto.com Arena, Los Angeles, California',
                                'GSW' : 'Chase Center, San Francisco, California',
                                'WAS' : 'Capital One Arena, Washington, D.C.',
                                'IND' : 'Gainbridge Fieldhouse, Indianapolis, Indiana',
                                'ORL' : 'Kia Center, Orlando, Florida',
                                'DET' : 'Little Caesars Arena, Detroit, Michigan',
                                'NYK' : 'Madison Square Garden, New York City, New York',
                                'MEM' : 'FedExForum, Memphis, Tennessee',
                                'NOP' : 'Smoothie King Center, New Orleans, Louisiana',
                                'BKN' : 'Barclays Center, Brooklyn, New York',
                                'CLE' : 'Rocket Mortgage FieldHouse, Cleveland, Ohio',
                                'TOR' : 'Scotiabank Arena, Toronto, Ontario',
                                'HOU' : 'Toyota Center, Houston, Texas',
                                'ATL' : 'State Farm Arena, Atlanta, Georgia',
                                'CHI' : 'United Center, Chicago, Illinois',
                                'MIA' : 'Kaseya Center, Miami, Florida',
                                'CHA' : 'Spectrum Center, Charlotte, North Carolina',
                                'SAS' : 'Frost Bank Center, San Antonio, Texas',
                                'OKC' : 'Paycom Center, Oklahoma City, Oklahoma',
                                'MIN' : 'Target Center, Minneapolis, Minnesota',
                                'DEN' : 'Ball Arena, Denver, Colorado',
                                'UTA' : 'Delta Center, Salt Lake City, Utah',
                                'DAL' : 'American Airlines Center, Dallas, Texas',
                                'PHX' : 'Footprint Center, Phoenix, Arizona',
                                'POR' : 'Moda Center, Portland, Oregon',
                                'SAC' : 'Golden 1 Center, Sacramento, California',
                                'MIL' : 'Fiserv Forum, Milwaukee, Wisconsin',
                                'LAC' : 'Crypto.com Arena, Los Angeles, California'}
        self.distances = {}
        self.geocodes = {}
        self.geolocator = Nominatim(user_agent="NBATravelDataCollection")
        self.generate_distances()
        self.save_distances()

    def generate_distances(self):
        for team1, stadium1 in self.team_to_stadium.items():
            if team1 in self.geocodes:
                location1 = self.geocodes[team1]
            else:
                location1 = self.geolocator.geocode(stadium1)
                time.sleep(1)
                self.geocodes[team1] = location1
            
            for team2, stadium2 in self.team_to_stadium.items():

                if (team2, team1) in self.distances:

                    self.distances[(team1, team2)] = self.distances[(team2, team1)]

                else:

                    if team2 in self.geocodes:
                        location2 = self.geocodes[team2]
                    else:
                        location2 = self.geolocator.geocode(stadium2)
                        time.sleep(1)
                        self.geocodes[team2] = location2
                
                    distance = geodesic((location1.latitude, location1.longitude), (location2.latitude, location2.longitude)).kilometers
                    # Save the distance using team tricodes
                    self.distances[(team1, team2)] = distance

                print(f'Distance between {team1} and {team2}: {self.distances[(team1, team2)]}')
    
    def save_distances(self):
        # Write the distances to a CSV file
        with open('team_distances.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            # Write the header
            writer.writerow(['Team1', 'Team2', 'Distance_km'])
            # Write the distances
            for (team1, team2), distance in self.distances.items():
                writer.writerow([team1, team2, distance])

            
import pandas as pd
from datetime import timedelta
count = 0


averages = pd.read_csv('all_averages.csv', index_col = 0)
# Function to calculate travel for a given period
def calculate_travel(team, game_date, days, games_df, current):
    global count
    global averages

    count += 1
    if count % 1000 == 0:
        print(team, game_date)
    
    existing = averages[(averages['teamTricode'] == team) & (pd.to_datetime(averages['date']) == pd.to_datetime(game_date))] 
    if len(existing) == 1:
        return existing[f'avg_travel_last_{days}_days'].iloc[0]

    
    distances_df = pd.read_csv('team_distances.csv')
    distances_dict = {(row['Team1'], row['Team2']): row['Distance_km'] for index, row in distances_df.iterrows()}
    end_date = game_date
    start_date = end_date - timedelta(days=days)
    # Filter games for the team and date range
    team_games = games_df[((games_df['HOME_TEAM_ABBREVIATION'] == team) | (games_df['AWAY_TEAM_ABBREVIATION'] == team)) & (games_df['GAME_DATE'] <= end_date) & (games_df['GAME_DATE'] >= start_date)]
    team_games = team_games.sort_values(by='GAME_DATE')

    
    # Calculate travel distances
    total_distance = 0
    to_stadium = None
    for i in range(len(team_games) - 1):
        from_stadium = team_games.iloc[i]['HOME_TEAM_ABBREVIATION']
        to_stadium = team_games.iloc[i + 1]['HOME_TEAM_ABBREVIATION']
        total_distance += distances_dict.get((from_stadium, to_stadium), 0)
    
    if current:
        
       
        games = fetch_games_today()
        for game in games:
            away, home = game.split('-')

            if (team == away or team == home) and len(team_games) > 0:
            
                to_stadium = team_games.iloc[-1]['HOME_TEAM_ABBREVIATION']
                print(team, to_stadium)
                total_distance += distances_dict.get((to_stadium, home), 0)

    if len(team_games) > 0:
        return total_distance
    else:
        return 0
    

