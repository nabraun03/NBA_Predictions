import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import numpy as np

seasons = ['2022-23', '2021-22', '2020-21', '2019-20', '2018-19', '2017-18', '2016-17']

team_map = {'Philadelphia' : 'PHI',
            'Boston' : 'BOS',
            'LALakers' : 'LAL',
            'GoldenState' : 'GSW',
            'Washington' : 'WAS',
            'Indiana' : 'IND',
            'Orlando' : 'ORL',
            'Detroit' : 'DET',
            'NewYork' : 'NYK',
            'Memphis' : 'MEM',
            'NewOrleans' : 'NOP',
            'Brooklyn' : 'BKN',
            'Cleveland' : 'CLE',
            'Toronto' : 'TOR',
            'Houston' : 'HOU',
            'Atlanta' : 'ATL',
            'Chicago' : 'CHI',
            'Miami' : 'MIA',
            'Charlotte' : 'CHA',
            'SanAntonio' : 'SAS',
            'OklahomaCity' : 'OKC',
            'Minnesota' : 'MIN',
            'Denver' : 'DEN',
            'Utah' : 'UTA',
            'Dallas' : 'DAL',
            'Phoenix' : 'PHX',
            'Portland' : 'POR',
            'Sacramento' : 'SAC',
            'Milwaukee' : 'MIL',
            'LAClippers' : 'LAC'}

dfs = []

for season in seasons:

    
    def convert_date(mmdd):
        if int(mmdd) > 1011: 

            return datetime.strptime(f"{season[0:4]}{mmdd}", "%Y%m%d").strftime("%Y-%m-%d")
        else:
            return datetime.strptime(f"20{season[5:7]}{mmdd}", "%Y%m%d").strftime("%Y-%m-%d")
        

    

    url = f"https://www.sportsbookreviewsonline.com/scoresoddsarchives/nba-odds-{season}/"

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
    }

    # Request the webpage
    response = requests.get(url, headers=headers)

    data = response.text

    # Parse the HTML
    soup = BeautifulSoup(data, 'html.parser')

    # Find the table or data points
    table = soup.find('table') # or the appropriate tag and class/id

    # Extract data
    # This part depends heavily on the structure of your HTML
    # You might need to loop through rows and columns
    table_data = []
    for row in table.find_all('tr'):
        row_data = [cell.text.strip() for cell in row.find_all('td')]
        table_data.append(row_data)

    df = pd.DataFrame(table_data)
    df.columns = df.iloc[0]
    df = df.drop(columns = ['Rot', '1st', '2nd', '3rd', '4th', 'Open', 'Close', '2H'])
    df=df[1:]

    df['Team'] = df['Team'].map(team_map)

    df['Date'] = df['Date'].apply(lambda x : convert_date(x))
    df = df.drop(columns = ['VH'])
    df['Final'] = df['Final'].astype(np.int64)
    df['ML'] = df['ML'].astype(np.int64)

    # Assuming df_betting is your DataFrame

    # Separate odd and even indexed rows
    df_odd = df.iloc[1::2].reset_index(drop=True)  # Away teams
    df_even = df.iloc[::2].reset_index(drop=True)  # Home teams
    df_odd = df_odd.rename(columns = {'Team' : 'HOME_TEAM_ABBREVIATION', 'Final' : 'HOME_TEAM_PTS', 'ML' : 'ML_HOME'})
    df_even = df_even.rename(columns = {'Team' : 'AWAY_TEAM_ABBREVIATION', 'Final' : 'AWAY_TEAM_PTS', 'ML' : 'ML_AWAY'})
    df_even = df_even.drop(columns = ['Date'])

    # Concatenate the two DataFrames side by side
    df = pd.concat([df_even, df_odd], axis=1)

    # Now combined_df has one row for each game with both teams' data
    if season == '2017-18' or season == '2023-24' or season == '2016-17':
        df_games = pd.read_csv(f'{season}_all_games.csv')
    else:
        df_games = pd.read_csv(f'{season}_all_games.csv')
        df_games = pd.concat([df_games, pd.read_csv(f'{season}_playoffs_games.csv')])
    df = df.rename(columns={'Date' : 'GAME_DATE'})

    df_games = pd.merge(df_games, df, on=['GAME_DATE', 'AWAY_TEAM_ABBREVIATION', 'HOME_TEAM_ABBREVIATION', 'HOME_TEAM_PTS', 'AWAY_TEAM_PTS'])
        
    dfs.append(df_games)


df_games = pd.concat(dfs)
df_games.to_csv(f'moneyline_data.csv')
# Save to CSV
# Assuming you have a DataFrame 'df' with the extracted data
#df.to_csv('moneyline_data.csv', index=False)
