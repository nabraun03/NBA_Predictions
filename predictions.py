import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from games import fetch_games_today
import xgboost as xgb

class Predictor:

    def __init__(self, n_ml_models, n_spread_models, current_profiles, training_data):
        self.n_ml_models = n_ml_models
        self.n_spread_models = n_spread_models
        self.current_profiles = current_profiles
        self.training_data = training_data
        self.games = fetch_games_today()
        self.ml_models = []
        self.spread_models = []
        #print("Loading ML Models")
        #self.load_ml_models()
        print("Loading spread models")
        self.load_spread_models()
        print("Loading scalers")
        self.load_scalers()
        print("Generating predictions")
        self.generate_predictions()



    def load_ml_models(self):
        for i in range(1, self.n_ml_models + 1):
            loaded_model = xgb.XGBClassifier()
            loaded_model = pickle.load(open(f"models/xgb_model_{i}.pkl", "rb"))
            self.ml_models.append(loaded_model)

    def load_spread_models(self):
        for i in range(1, self.n_spread_models + 1):
            loaded_model = xgb.XGBRegressor()
            loaded_model = pickle.load(open(f"models/spread_model_{i}.pkl", "rb"))
            self.spread_models.append(loaded_model)
    
    def fetch_most_recent_profile(self, team):
        return self.current_profiles[self.current_profiles[f'teamTricode_{team}'] == 1]

    def load_scalers(self):
        #with open('scaler.pkl', 'rb') as file:
        #    self.ml_scaler = pickle.load(file)
        
        with open('spreadscaler.pkl', 'rb') as file:
            self.spread_scaler = pickle.load(file)

    def predict_ml(self, combined):
        total = 0
        for model in self.ml_models:
            total += model.predict(combined)
        total = total / self.n_ml_models

        if total == 1:
            return "Home"
        elif total == 0:
            return "Away"
        else:
            return f"Split: {total[0]}"

    def predict_spread(self, combined):
        total = 0
        for model in self.spread_models:
            total += model.predict(combined)
        return total / self.n_spread_models
    
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

            
            combined['playoff'] = 0
            combined = combined[self.training_data.columns]
            combined_ml = combined.copy()
            combined_spread = combined.copy()

            numerical_cols = [col for col in combined.columns if col not in ['elo', 'gameId', 'home_win', 'game_count_home', 'game_count_away' 'playoff', 'time_between_games_home', 'time_between_games_away'] and 'teamTricode' not in col]
            #combined_ml[numerical_cols] = self.ml_scaler.transform(combined_ml[numerical_cols])
            combined_spread[numerical_cols] = self.spread_scaler.transform(combined_spread[numerical_cols])

            #ml_prediction = self.predict_ml(combined_ml)
            for col in set(combined_spread.columns):
                if col not in set(self.training_data.columns):
                    print(col)
            for col in set(self.training_data.columns):
                if col not in set(combined_spread.columns):
                    print(col)
            
            combined_spread = combined_spread[self.training_data.columns]
            spread_prediction = self.predict_spread(combined_spread)


            print(f"{teams[0]} @ {teams[1]}")
            #print(f"Moneyline: {ml_prediction}")
            if spread_prediction[0] > 0:
                winner = teams[1]
            else:
                winner = teams[0]
            print(f"Spread: {winner} by {abs(spread_prediction[0])}")
            print('')
        





if __name__ == '__main__':
    current_profiles = pd.read_csv('current_profiles.csv', index_col = 0)
    current_profiles = pd.get_dummies(current_profiles, columns = ['teamTricode'])
    train = pd.read_csv('X_train.csv', index_col = 0)
    p = Predictor(5, 46, current_profiles, train)








 