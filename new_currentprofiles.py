from preprocessing import Preprocessor
import pandas as pd

seasons = ['2023-24', '2022-23']

if __name__ == '__main__':
    p = Preprocessor(seasons, 50, 0)
    complete_profiles = p.complete_profiles
    current_profiles = []

    groups = complete_profiles.groupby('teamTricode')
    for group in groups:
        group.sort_values(by = 'date', ascending = False)
        current_profiles.append(group.iloc[1])
    
    current_profiles = pd.concat(current_profiles)
    current_profiles.to_csv('currentprofiles.csv')