import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
from games import fetch_games_today

df_current_profiles = pd.read_csv('current_profiles.csv', index_col = 0)
df_averages = pd.read_csv('all_averages.csv', index_col = 0)
df_games = pd.read_csv('all_games.csv', index_col = 0)

df_averages.drop(columns = ['date'])
df_current_profiles = pd.get_dummies(df_current_profiles, columns = ['teamTricode'])


def fetch_most_recent_profile(team):
    return df_current_profiles[df_current_profiles[f'teamTricode_{team}'] == 1]


# Merge for home and away teams
df_home = pd.merge(df_averages, df_games[['gameId', 'HOME_TEAM_ABBREVIATION']], left_on=['gameId', 'teamTricode'], right_on=['gameId', 'HOME_TEAM_ABBREVIATION'], how='inner')
df_away = pd.merge(df_averages, df_games[['gameId', 'AWAY_TEAM_ABBREVIATION']], left_on=['gameId', 'teamTricode'], right_on=['gameId', 'AWAY_TEAM_ABBREVIATION'], how='inner')
# Sort by 'gameid'

df_home = df_home.drop(columns=['HOME_TEAM_ABBREVIATION', 'date'])  # Drop 'home' only if it exists
df_away = df_away.drop(columns=['AWAY_TEAM_ABBREVIATION', 'playoff', 'date'])  # Drop 'away' only if it exists

numerical_cols = [col for col in df_home.columns if col not in ['elo', 'gameId', 'home_win', 'game_count', 'playoff', 'time_between_games', 'teamTricode', 'date']]

X_home = pd.get_dummies(df_home, columns = ['teamTricode', 'playoff'])
X_away = pd.get_dummies(df_away, columns = ['teamTricode'])

X_combined = X_home.set_index('gameId').sub(X_away.set_index('gameId'), fill_value=0).reset_index()


X_combined = X_combined[abs(X_combined['game_count']) < 5]
X_combined = X_combined.sort_values(by='gameId')

scaler = StandardScaler()

X_combined[numerical_cols] = scaler.fit_transform(X_combined[numerical_cols])


def find_implied_lines(predictions):

    away_odds, home_odds = predictions[0], predictions[1]
 

    if away_odds > 0.5:
        away_odds = '-' + str(away_odds * 100 / (1 - away_odds))
    else:
        away_odds = '+' + str((100 - away_odds * 100) / away_odds)

    if home_odds > 0.5:
        home_odds = '-' + str(home_odds * 100 / (1 - home_odds))
    else:
        home_odds = '+' + str((100 - home_odds * 100) / home_odds)

    return home_odds, away_odds
    
num_models = 500

def load_models(num_models):
    models = []
    for i in range(num_models):
        loaded_model = pickle.load(open(f"xgb_model_{i}.pkl", "rb"))
        models.append(loaded_model)
    return models

def load_calibrated_models(num_models):
    c_models = []
    for i in range(num_models):
        calibrated_model = pickle.load(open(f"calibrated_xgb_model_{i}.pkl", "rb"))
        c_models.append(calibrated_model)
    return c_models

    

def predict(combined, models):
    # Initialize an empty list to store probabilities
    ensemble_probabilities = []

    # Number of models in the ensemble


    # Load each model and make predictions
    for model in models:
        
        # Make probability predictions for the new games
        probabilities = model.predict_proba(combined)

        # Append the probabilities to the ensemble list
        ensemble_probabilities.append(probabilities)

    # Convert the list to a numpy array for easier manipulation
    ensemble_probabilities = np.array(ensemble_probabilities)
    # Calculate the mean probabilities across all ensemble models
    mean_probabilities = np.mean(ensemble_probabilities, axis=0)



    # Extract probabilities of winning for each team

    prob_team1_wins = float(mean_probabilities[:, 0])
    prob_team2_wins = float(mean_probabilities[:, 1])
    return [prob_team1_wins, prob_team2_wins]

def predict_calibrated(combined, models):
    ensemble_probabilities = []
    
    for calibrated_model in models:
        
        probabilities = calibrated_model.predict_proba(combined)
        ensemble_probabilities.append(probabilities)
    
    # Average the probabilities from the calibrated models
    mean_calibrated_probabilities = np.mean(ensemble_probabilities, axis=0)
    return mean_calibrated_probabilities[0]


games = fetch_games_today()

MOE = 5

print("Loading models...")
models = load_models(num_models)
c_models = load_calibrated_models(num_models)
#os.system('cls')
print("Models loaded")

for game in games:
    teams = game.split('-')
    home_team = fetch_most_recent_profile(teams[1])
    away_team = fetch_most_recent_profile(teams[0])

    home_team = home_team.drop(columns = ['date'])
    away_team = away_team.drop(columns = ['date'])
    
    
 
    home_team = pd.get_dummies(home_team, columns = ['playoff'])
    home_team['playoff_True'] = 0
    
    X_train = pd.read_csv('all_averages_with_ids.csv', index_col = 0)

    combined = home_team.set_index('gameId').sub(away_team.set_index('gameId'), fill_value=0).reset_index()
    combined = combined[X_train.columns]
    combined[numerical_cols] = scaler.transform(combined[numerical_cols])

    
    combined = combined.drop(columns = ['gameId'])

    combined.to_csv('combined_temp.csv')

    #predictions = predict(combined, models)
    calibrated_predictions = predict_calibrated(combined, c_models)

    calibrated_predictions_h = [calibrated_predictions[0] - (MOE / 100), calibrated_predictions[1] + (MOE / 100)]
    calibrated_predictions_l = [calibrated_predictions[0] + (MOE / 100), calibrated_predictions[1] - (MOE / 100)]

    #home_odds, away_odds = find_implied_lines(predictions)

    home_c_odds_h, away_c_odds_h = find_implied_lines(calibrated_predictions_h)
    home_c_odds, away_c_odds = find_implied_lines(calibrated_predictions)
    home_c_odds_l, away_c_odds_l = find_implied_lines(calibrated_predictions_l)


    #print(f'Win probabilities: {teams[1]} : {predictions[1]}, {teams[0]} : {predictions[0]}')
    #print(f'Fair odds: {teams[1]}: {home_odds}, {teams[0]}: {away_odds}')
    #  print('')
    
    home_c_pred_range = sorted([calibrated_predictions_h[1], calibrated_predictions_l[1]])
    away_c_pred_range = sorted([calibrated_predictions_h[0], calibrated_predictions_l[0]])

    print(f"{teams[0]} @ {teams[1]}")
    print("Win probabilities:")
    print(f"{teams[0]}: {round(away_c_pred_range[0] * 100, 2)}-{round(away_c_pred_range[1] * 100, 2)}%")
    print(f"{teams[1]}: {round(home_c_pred_range[0] * 100, 2)}-{round(home_c_pred_range[1] * 100, 2)}%")
    print("Fair odds:")
    print(f"{teams[0]}: {away_c_odds[0:4]}")
    print(f"{teams[1]}: {home_c_odds[0:4]}")
    print("Cautious odds:")
    print(f"{teams[0]}: {away_c_odds_h[0:4]}")
    print(f"{teams[1]}: {home_c_odds_l[0:4]}")


    print('')





 