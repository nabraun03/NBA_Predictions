import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from games import fetch_games_today
import xgboost as xgb
import matplotlib.pyplot as plt


scripts = ['currentprofiles.py']

#for script in scripts:
#    print(script)
#    with open(script, "r") as file:
#        exec(file.read())

def team_helper(teams, value):
    if value > 0:
        return teams[1]
    else:
        return teams[0]

class Predictor:

    def __init__(self, n_ml_models, n_spread_models, current_profiles, training_data):
        self.n_ml_models = n_ml_models
        self.n_spread_models = n_spread_models
        self.current_profiles = current_profiles
        self.training_data = training_data
        self.games = fetch_games_today()
        print(self.games)
        self.ml_models = []
        self.spread_models = []
        print("Loading spread models")
        self.load_spread_models()
        print("Loading scalers")
        self.load_scalers()
        print("Generating predictions")
        self.generate_predictions()

    def load_spread_models(self):
        for i in range(1, self.n_spread_models + 1):
            loaded_model = xgb.XGBRegressor()
            loaded_model = pickle.load(open(f"models/spread_model_{i}.pkl", "rb"))
            self.spread_models.append(loaded_model)
    
    def fetch_most_recent_profile(self, team):
        return self.current_profiles[self.current_profiles[f'teamTricode_{team}'] == 1]

    def load_scalers(self):
        with open('spreadscaler.pkl', 'rb') as file:
            self.spread_scaler = pickle.load(file)

    def predict_spread(self, combined):
        total = 0
        predictions = []
        for model in self.spread_models:
            prediction = model.predict(combined)[0]
            total += prediction
            predictions.append(prediction)

        return predictions
    
    def generate_predictions(self):
        for game in self.games:
            teams = game.split('-')
            home_team = self.fetch_most_recent_profile(teams[1])
            away_team = self.fetch_most_recent_profile(teams[0])

            home_team = home_team.drop(columns = ['date', 'playoff'])
            away_team = away_team.drop(columns = ['date', 'playoff'])
            
            
            
            home_team['gameId'] = 1
            away_team['gameId'] = 1

            assert list(home_team.columns) == list(away_team.columns)

            combined = pd.merge(home_team, away_team, on = ['gameId'], suffixes=['_home', '_away'])

                        # Calculate the differential features using .sub()
            home_team = home_team.set_index('gameId')
            away_team = away_team.set_index('gameId')
            
            diff_features = home_team.sub(away_team, fill_value=0)

            # Reset index to make 'gameId' a column again for the merge operation
            diff_features = diff_features.reset_index()

            # Rename the columns to indicate they are differential features
            diff_features.columns = ['gameId'] + [f'diff_{col}' for col in diff_features.columns if col != 'gameId']

            # Merge the differential features onto X_combined based on 'gameId'
            combined = combined.merge(diff_features, on='gameId', how='left')

            
            combined['playoff'] = 1
            combined = combined[self.training_data.columns]
            combined_ml = combined.copy()
            combined_spread = combined.copy()

            numerical_cols = [col for col in combined.columns if col not in ['elo', 'gameId', 'home_win', 'game_count_home', 'game_count_away' 'playoff', 'time_between_games_home', 'time_between_games_away'] and 'teamTricode' not in col]
            #combined_ml[numerical_cols] = self.ml_scaler.transform(combined_ml[numerical_cols])
            combined_spread[numerical_cols] = self.spread_scaler.transform(combined_spread[numerical_cols])

            for col in set(combined_spread.columns):
                if col not in set(self.training_data.columns):
                    print(col)
            for col in set(self.training_data.columns):
                if col not in set(combined_spread.columns):
                    print(col)
            
            combined_spread = combined_spread[self.training_data.columns]
            predictions = self.predict_spread(combined_spread)




            print(f"{teams[0]} @ {teams[1]}")
            print(f"Spread: {team_helper(teams, np.mean(predictions))} by {abs(np.mean(predictions))}")
            print(f"Median spread: {team_helper(teams, np.median(predictions))} by {abs(np.median(predictions))}")
            print(f"Standard deviation: {np.std(predictions)}")
            print(f"Minimum: {team_helper(teams, np.min(predictions))} by {abs(np.min(predictions))}")
            print(f"Maximum: {team_helper(teams, np.max(predictions))} by {abs(np.max(predictions))}")
            print(f"25th Percentile: {team_helper(teams, np.percentile(predictions, 25))} by {abs(np.percentile(predictions, 25))}")
            print(f"75th Percentile: {team_helper(teams, np.percentile(predictions, 75))} by {abs(np.percentile(predictions, 75))}")
            print('')

            plt.hist(predictions, bins=10, alpha=0.7, color='blue', edgecolor='black')
            plt.title('Histogram of Data')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.show()

        





if __name__ == '__main__':
    current_profiles = pd.read_csv('current_profiles.csv', index_col = 0)
    current_profiles = pd.get_dummies(current_profiles, columns = ['teamTricode'])
    train = pd.read_csv('X_train.csv', index_col = 0)
    p = Predictor(5, 28, current_profiles, train)








 