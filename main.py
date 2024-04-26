import argparse

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

from injuries import fetch_injured_players
from preprocessing import Preprocessor  # Assuming this is the function from your separate file
import numpy as np
from datetime import date, datetime
from travel import calculate_travel

import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from games import fetch_games_today
import xgboost as xgb

common_scope = {
    "pd": pd,
    "np": np,
    "leaguegamelog": leaguegamelog,
    "boxscoreadvancedv3": boxscoreadvancedv3,
    "boxscoretraditionalv3": boxscoretraditionalv3,
    "boxscorehustlev2": boxscorehustlev2,
    "boxscoremiscv3": boxscoremiscv3,
    "boxscoreplayertrackv3": boxscoreplayertrackv3,
    "PlayerGameLogs": PlayerGameLogs,
    "commonteamroster": commonteamroster,
    "teams": teams,
    "date": date,
    "datetime": datetime,
    "time": time,
    "ReadTimeout": ReadTimeout,
    "JSONDecodeError": JSONDecodeError,
    "fetch_injured_players": fetch_injured_players,
    "Preprocessor": Preprocessor,
    "calculate_travel": calculate_travel,
    "pickle": pickle,
    "StandardScaler": StandardScaler,
    "fetch_games_today": fetch_games_today,
    "xgb": xgb
}

import subprocess
def run_scripts(scripts, args):

    for script in scripts:
        if (args.v == True):
            print(script)
        subprocess.run(['python', script])


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        prog="main",
        description="Run sequences of relevant Python scripts at once",
    )

    parser.add_argument(
        "-f",
        "--full-predictions",
        help="Fetches data from online, creates current profiles, and generates predictions",
        action='store_true',
        default=False
    )

    parser.add_argument(
        "-p",
        "--preprocess",
        help="Fetches data from online and generates all training data",
        action='store_true',
        default=False
    )

    parser.add_argument(
        "-u",
        "--update",
        help="Creates current profiles and generates predictions",
        action='store_true',
        default=False
    )

    parser.add_argument(
        "-v",
        help="verbose",
        action='store_true',
        default=False
    )

    args = parser.parse_known_args()[0]
    parser.set_defaults(full_predictions = False, training_data = False, new_predictions = False)
    
    if args.preprocess:
        scripts = ['apirequests.py', 'preprocessing.py']
    elif args.update:
        scripts = ['currentprofiles.py', 'predictions.py']
    else:
        scripts = ['apirequests.py', 'currentprofiles.py', 'predictions.py']
    
    run_scripts(scripts, args)

        

    
