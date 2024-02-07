import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df_games = pd.read_csv('all_games.csv', index_col = 0)
df_averages = pd.read_csv('all_averages.csv', index_col = 0)
df_averages = df_averages.drop(columns = ['date'])
df_averages = df_averages[df_averages['game_count'] >= 10]


# Create 'home_win' column
df_games['home_win'] = (df_games['HOME_TEAM_PTS'] > df_games['AWAY_TEAM_PTS']).astype(int)

# Merge for home and away teams
print("Merging...")
df_home = pd.merge(df_averages, df_games[['gameId', 'HOME_TEAM_ABBREVIATION', 'home_win']], left_on=['gameId', 'teamTricode'], right_on=['gameId', 'HOME_TEAM_ABBREVIATION'], how='inner')
df_away = pd.merge(df_averages, df_games[['gameId', 'AWAY_TEAM_ABBREVIATION', 'home_win']], left_on=['gameId', 'teamTricode'], right_on=['gameId', 'AWAY_TEAM_ABBREVIATION'], how='inner')
# Sort by 'gameid'
df_home = df_home.sort_values(by='gameId')
df_away = df_away.sort_values(by='gameId')
df_games = df_games.sort_values(by='gameId')
# Prepare the feature and label DataFrames
X_home = df_home.drop(columns=['HOME_TEAM_ABBREVIATION'])  # Drop 'home' only if it exists
X_away = df_away.drop(columns=['home_win', 'AWAY_TEAM_ABBREVIATION', 'playoff'])  # Drop 'away' only if it exists


numerical_cols = [col for col in X_home.columns if col not in ['elo', 'gameId', 'home_win', 'game_count', 'playoff', 'time_between_games', 'teamTricode']]



scaler = StandardScaler()

#X_home[numerical_cols] = scaler.fit_transform(X_home[numerical_cols])
#X_away[numerical_cols] = scaler.fit_transform(X_away[numerical_cols])


X_home = pd.get_dummies(X_home, columns = ['teamTricode'])
X_away = pd.get_dummies(X_away, columns = ['teamTricode'])

X_home = X_home.drop(columns = ['playoff'])

X_combined = X_home.set_index('gameId').sub(X_away.set_index('gameId'), fill_value=0).reset_index()

# Rename columns to indicate 'home' or 'away'

#X_home.columns = ['gameId', 'playoff'] + [str(col) + '_home' for  col in X_away.columns if col not in ['gameId', 'home_win', 'playoff']] + ['home_win']

X_combined = pd.merge(X_combined, df_averages[['gameId', 'playoff']], on = ['gameId'], how = 'inner')

X_combined = X_combined[abs(X_combined['game_count']) < 5]
X_combined = X_combined.sort_values(by='gameId')

y = X_combined[['home_win']]
X_combined = X_combined.drop(columns = ['home_win'])

X_combined[numerical_cols] = scaler.fit_transform(X_combined[numerical_cols])

# Identify rows with NaN values in X_combined
nan_rows = X_combined[X_combined.isna().any(axis=1)]

# Drop corresponding rows in y
y = y.drop(nan_rows.index)
y = y.values.ravel()

# Drop rows with NaN values in X_combined
X_combined = X_combined.dropna()

X_combined.to_csv('all_averages_with_ids.csv')


X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2)
X_train, X_calib, y_train, y_calib = train_test_split(X_combined, y, test_size=0.2)

X_test.to_csv('test_averages_with_ids.csv')
X_train = X_train.drop(columns = ['gameId'])
X_train.to_csv('X_train.csv')
X_test = X_test.drop(columns = ['gameId'])

"""
from tensorflow import keras
from keras import Sequential, layers, regularizers
def create_model(n_layers = 1, n_neurons = 32, activation = "relu", dropout = 0.5, regularization = 0.01):
    model = Sequential()
    model.add(layers.Input(shape = (X_train.shape[1],)))
    for _ in range(n_layers):
        model.add(tf.keras.layers.Dense(n_neurons * (2 ** n_layers), kernel_regularizer=regularizers.l1(regularization), activation = activation, kernel_initializer='glorot_uniform'))
        n_layers -= 1
        model.add(layers.Dropout(dropout))
    model.add(layers.Dense(1, activation = 'sigmoid'))

    model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])
    
    print("Model compiled")
    return model

"""
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import random
import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid_xgb = {
    'learning_rate': [0.05, 0.01],
    'n_estimators': [100, 500],
    'max_depth': [3, 10],
    'alpha' : [0, 0.01]

}

# Create GridSearchCV object
grid_search_xgb = GridSearchCV(estimator=XGBClassifier(objective='binary:logistic', eval_metric='logloss'),
                               param_grid=param_grid_xgb,
                               cv=20, verbose=3, n_jobs=-1)

# Fit to the data
#grid_search_xgb.fit(X_train, y_train)

# Get the best parameters
#best_params_xgb = grid_search_xgb.best_params_
#print("Best parameters for XGBoost:", best_params_xgb)

param_grid_rf = {
    'n_estimators': [50, 100, 300, 500, 1000],
    'max_depth': [3, 5, 7, 10],
    'criterion': ['entropy'],
    'max_features' : [3, 'log2', 10, 'sqrt'],
    
}

import pickle
num_samples = X_test.shape[0]
sum_probabilities = np.zeros((num_samples, 2))  

n_models = 50


from sklearn.calibration import CalibratedClassifierCV
calibrated_models = []
for i in range(n_models):
    print(f"Training model {i} / {n_models}...")
    xgb_model = XGBClassifier(max_depth = random.randint(2, 5), learning_rate=random.randrange(5,10,1)/1000.0, n_estimators=random.randint(300, 600), objective='binary:logistic', eval_metric='logloss', alpha=random.randrange(0,10,1)/100.0)
    xgb_model.fit(X_train, y_train,
                eval_set=[(X_train, y_train)],
                verbose = False)
    
    # Get feature importances
    feature_names = X_train.columns.tolist()
    importances = xgb_model.feature_importances_

    # Pair feature names with their importances
    feature_importances = [(feature, round(importance, 4)) for feature, importance in zip(feature_names, importances)]

    # Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)

    # Display the feature importances
    for feature, importance in feature_importances:
        print(f"Feature: {feature}, Importance: {importance}")


    # Calibrate model on calibration dataset
    calibrated_model = CalibratedClassifierCV(xgb_model, method='isotonic', cv='prefit')
    #calibrated_model.fit(X_calib, y_calib)
    
    # Store the calibrated model
    calibrated_models.append(calibrated_model)
    
    # Save model to file
    pickle.dump(calibrated_model, open(f"calibrated_xgb_model_{i}.pkl", "wb"))

    sum_probabilities +=  xgb_model.predict_proba(X_test)



    # Save model to file
    pickle.dump(xgb_model, open(f"xgb_model_{i}.pkl", "wb"))


# Calculate the mean probabilities
mean_probabilities = sum_probabilities / n_models

# Initialize an array to store the probabilities
sum_calibrated_probabilities = np.zeros((num_samples, 2))

for calibrated_model in calibrated_models:
    sum_calibrated_probabilities += calibrated_model.predict_proba(X_test)

# Calculate the average probabilities
mean_calibrated_probabilities = sum_calibrated_probabilities / len(calibrated_models)
print(f"Calibrated accuracy: {mean_calibrated_probabilities}")

# Convert these mean probabilities to class labels
# Apply a threshold, e.g., 0.5 for binary classification
mean_predictions = (mean_probabilities[:, 1] > 0.5).astype(int)
accuracy = accuracy_score(y_test, mean_predictions)
print(f"Ensemble model accuracy: {accuracy * 100:.2f}%")

"""
grid_search_rf = GridSearchCV(estimator = RandomForestClassifier(), param_grid=param_grid_rf, cv = 5, verbose = 2, n_jobs = -1)
grid_search_rf.fit(X_train, y_train)

best_params_rf = grid_search_rf.best_params_
print("Decision tree best parameters: ", best_params_rf)


rf_model = RandomForestClassifier(n_estimators=50, max_depth = 10, max_features=10, criterion = 'entropy')
rf_model.fit(X_train, y_train)


# Train Neural Networks (for demonstration, let's assume 2)
# You can use your `create_model` function here


ensemble_nn = []
for i in range(10):  # Create 100 neural networks
    n_layers = random.randint(1, 6)
    n_neurons = random.randint(1, 64)
    activation = random.choice(['relu', 'tanh'])
    dropout = random.randint(0, 4) / 10
    regularization = random.randint(0, 40) / 1000
    model = KerasClassifier(model=create_model(n_layers, n_neurons, activation, dropout, regularization), batch_size=16, verbose=0)
    model.fit(X_train, y_train)
    ensemble_nn.append(model)

# Step 2: Extract Trees
individual_trees = rf_model.estimators_

# Step 3: Collect Predictions

# Collect predictions from each tree
tree_predictions = rf_model.predict(X_test)

tree_accuracy = accuracy_score(y_test, tree_predictions)
#print("XGBoost accuracy: ", accuracy_score(y_test, xgb_predictions))
print("Tree accuracy:", tree_accuracy)


xgb_results_df = pd.DataFrame({'True_Labels': y_test, 'Predicted_Labels': xgb_predictions})
xgb_results_df['home_game_count'] = X_test['game_count_home']
xgb_results_df['away_game_count'] = X_test['game_count_away']


grouped_results = xgb_results_df.groupby(['home_game_count', 'away_game_count']).apply(lambda x: accuracy_score(x['True_Labels'], x['Predicted_Labels']))
grouped_results = grouped_results.reset_index()
grouped_results.columns = ['home_game_count', 'away_game_count', 'accuracy']

import matplotlib.pyplot as plt
import seaborn as sns

sns.lineplot(x='home_game_count', y='accuracy', data=grouped_results.reset_index(), label='Home Team')
sns.lineplot(x='away_game_count', y='accuracy', data=grouped_results.reset_index(), label='Away Team')

plt.xlabel('Game Count')
plt.ylabel('Model Accuracy')
plt.title('Model Accuracy vs Game Count')
plt.legend()
plt.show()

# Collect predictions from each neural network
nn_predictions = [nn.predict(X_test) for nn in ensemble_nn]
# Transpose the array of neural network predictions
nn_predictions_array = np.array(nn_predictions)
nn_predictions_transposed = nn_predictions_array.T  # Shape will be (261, 100)

# Find the mode along each row
nn_final_predictions = stats.mode(nn_predictions_transposed, axis=1)[0]

# Flatten the array to get a 1D array of final predictions
nn_final_predictions = nn_final_predictions.flatten()
nn_accuracy = accuracy_score(y_test, nn_final_predictions)
print("NN Accuracy: ", nn_accuracy)

# Step 4: Custom Voting

# Combine all predictions
all_predictions = np.column_stack((tree_predictions, nn_final_predictions))

# Perform voting
final_predictions = stats.mode(all_predictions, axis=1)[0]

# Evaluate
ensemble_score = accuracy_score(y_test, final_predictions)
print("Ensemble Score:", ensemble_score)
"""