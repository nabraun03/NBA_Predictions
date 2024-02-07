import requests
from bs4 import BeautifulSoup
from datetime import datetime
from datetime import timedelta

team_map = {'76ers' : 'PHI',
            'Celtics' : 'BOS',
            'Lakers' : 'LAL',
            'Warriors' : 'GSW',
            'Wizards' : 'WAS',
            'Pacers' : 'IND',
            'Magic' : 'ORL',
            'Pistons' : 'DET',
            'Knicks' : 'NYK',
            'Grizzlies' : 'MEM',
            'Pelicans' : 'NOP',
            'Nets' : 'BKN',
            'Cavaliers' : 'CLE',
            'Raptors' : 'TOR',
            'Rockets' : 'HOU',
            'Hawks' : 'ATL',
            'Bulls' : 'CHI',
            'Heat' : 'MIA',
            'Hornets' : 'CHA',
            'Spurs' : 'SAS',
            'Thunder' : 'OKC',
            'Timberwolves' : 'MIN',
            'Nuggets' : 'DEN',
            'Jazz' : 'UTA',
            'Mavericks' : 'DAL',
            'Suns' : 'PHX',
            'Trail Blazers' : 'POR',
            'Kings' : 'SAC',
            'Bucks' : 'MIL',
            'Clippers' : 'LAC'}

def fetch_games_today(yesterday=False):

    url = f"https://www.espn.com/nba/scoreboard/_/date/"

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
    }

    if yesterday:
        current_date = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
    else:
        current_date = datetime.now().strftime("%Y%m%d")
    url = f'{url}{current_date}'

    # Request the webpage
    response = requests.get(url, headers=headers)

    data = response.text

    

    # Parse the HTML
    soup = BeautifulSoup(data, 'html.parser')


    team_names_raw = soup.find_all("div", class_="ScoreCell__TeamName ScoreCell__TeamName--shortDisplayName truncate db")

    games = []
    for i in range(0, len(team_names_raw), 2):
        games.append(f'{team_map[team_names_raw[i].text]}-{team_map[team_names_raw[i+1].text]}')
    return games
