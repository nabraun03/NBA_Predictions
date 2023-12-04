import pandas as pd
import math
import pickle
from sklearn.preprocessing import StandardScaler
import numpy as np
import os

print("here")


def calculate_winnings(odds, bet_amount):
    
    if odds > 0:
        # Positive odds
        winnings = bet_amount * (odds / 100) - bet_amount
    else:
        # Negative odds (use absolute value for calculation)
        winnings = bet_amount * (100 / abs(odds)) - bet_amount

    # Total return if the bet wins
    total_return = bet_amount + winnings
    return total_return

def compare_lines(my_line, their_line):
    my_line = str(my_line)
    their_line = str(their_line)
    if my_line[0] == '-' and their_line[0] != '-':
        return 1
    elif my_line[0] == '+' and their_line[0] == '-':
        return 0
    elif (my_line[0] == '-' and int(my_line[1:]) > int(their_line[1:])) or (my_line[0] == '+' and int(my_line[1:]) < int(their_line)):
        return 1
    else:
        return 0

def find_implied_lines(predictions):

    away_odds, home_odds = predictions[0], predictions[1]
    

    if away_odds > 0.5:
        away_odds = '-' + str(away_odds * 100 / (1 - away_odds))
    elif away_odds > 0:
        away_odds = '+' + str((100 - away_odds * 100) / away_odds)
    else:
        away_odds = '+' + str(999)

    if home_odds > 0.5:
        home_odds = '-' + str(home_odds * 100 / (1 - home_odds))
    elif home_odds > 0:
        home_odds = '+' + str((100 - home_odds * 100) / home_odds)
    else:
        home_odds = '+' + str(999)

    return home_odds, away_odds
    
seasons = ['2022-23', '2021-22', '2020-21', '2019-20', '2018-19', '2017-18', '2016-17']

num_models = 500
calibrated_models = []
for i in range(num_models):
    print(f'Loading model {i}/{num_models}')
    calibrated_model = pickle.load(open(f"calibrated_xgb_model_{i}.pkl", "rb"))
    os.system('cls')
print("Loaded models")


# Assuming you have a function that makes predictions based on team profiles
def make_prediction(combined):

    ensemble_probabilities = []
    
    for i in range(num_models):
        probabilities = calibrated_model.predict_proba(combined)
        ensemble_probabilities.append(probabilities)
    
    # Average the probabilities from the calibrated models
    mean_calibrated_probabilities = np.mean(ensemble_probabilities, axis=0)
    return mean_calibrated_probabilities[0]

df_odds = pd.read_csv('moneyline_data.csv')
df_profiles = pd.read_csv(f'test_averages_with_ids.csv', index_col=0)

df_odds = df_odds.sort_values(by = ['GAME_DATE'])
grouped_bets = df_odds.groupby('GAME_DATE')

def calculate_daily_result(df_odds, bankroll, bet_proportion, MOE, date):
    

    daily_result = 0

    for index, row in df_odds.iterrows():

        combined = df_profiles[df_profiles['gameId'] == row['gameId']]
        if combined.size < 1:
            continue
        if int(date.split('-')[0]) < 2021:
            continue
        combined = combined.drop(columns = ['gameId'])

        predictions = make_prediction(combined)

        predictions = [predictions[0] - MOE / 100, predictions[1] - MOE / 100]
        lines = find_implied_lines(predictions)

        my_home_line = lines[0]
        my_away_line = lines[1]

        their_home_line = row['ML_HOME']
        their_away_line = row['ML_AWAY']

        #print(my_home_line, their_home_line)
        #print(my_away_line, their_away_line)

        if compare_lines(my_home_line[0:4], their_home_line):
            print(row['HOME_TEAM_ABBREVIATION'], row['AWAY_TEAM_ABBREVIATION'])
            print("BET HOME")
            print(f"My line: {my_home_line[0:4]}")
            print(f"Their line: {their_home_line}")
            print(f"Score: {row['HOME_TEAM_PTS']}-{row['AWAY_TEAM_PTS']}")
            if row['HOME_TEAM_PTS'] > row['AWAY_TEAM_PTS']:
                winnings = calculate_winnings(their_home_line, bankroll * bet_proportion)
                print(f'Won {winnings} on {bankroll * bet_proportion}')
                daily_result += math.floor(winnings * 100)/100.0
            else:
                print(f'Lost {bankroll * bet_proportion}')
                daily_result -= bankroll * bet_proportion
                
        if compare_lines(my_away_line[0:4], their_away_line):
            print(row['HOME_TEAM_ABBREVIATION'], row['AWAY_TEAM_ABBREVIATION'])
            print("BET AWAY")
            print(f"My line: {my_away_line[0:4]}")
            print(f"Their line: {their_away_line}")
            print(f"Score: {row['HOME_TEAM_PTS']}-{row['AWAY_TEAM_PTS']}")
            if row['HOME_TEAM_PTS'] < row['AWAY_TEAM_PTS']:
                winnings = calculate_winnings(their_away_line, bankroll * bet_proportion)
                print(f'Won {winnings} on {bankroll * bet_proportion}')
                daily_result += math.floor(winnings * 100)/100.0

            else:
                daily_result -= bankroll * bet_proportion
                print(f'Lost {bankroll * bet_proportion}')

        
        print(' ')
    return daily_result


bet_proportion = .02  # Example: 5% of bankroll on each game
bankroll = 50
no_moe_bankroll = 50
MOE = 5

for date, daily_bets in grouped_bets:
    print(date)
    daily_result = calculate_daily_result(daily_bets, bankroll, bet_proportion, MOE, date)
    bankroll += math.floor(daily_result * 100)/100.0  # Update the bankroll
    bankroll = math.floor(bankroll * 100)/100.0 
    
    daily_no_moe_result = calculate_daily_result(daily_bets, no_moe_bankroll, bet_proportion, MOE - 5, date)
    no_moe_bankroll += math.floor(daily_no_moe_result * 100)/100.0  # Update the bankroll
    no_moe_bankroll = math.floor(no_moe_bankroll * 100)/100.0 
    os.system('cls')
    print(f"MOE = 10 Date: {date}, Daily Result: {daily_result}, New Bankroll: {bankroll}")
    print(f"MOE = 5 Date: {date}, Daily Result: {daily_no_moe_result}, New Bankroll: {no_moe_bankroll}")





    


