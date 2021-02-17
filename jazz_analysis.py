import pandas as pd
import numpy as np
import pandas_profiling as pp


df = pd.read_csv('jazz.csv', sep='|')

#%%

num_cols = [
            'danceability',
            'energy',
            'speechiness',
            'acousticness',
            'instrumentalness',
            'liveness',
            'valence',
            'num_samples',
            'loudness',
            'tempo',
            'bars_num',
            'bars_duration_mean',
            'bars_duration_var',
            'beats_num',
            'beats_duration_mean',
            'beats_duration_var',
            'sections_num',
            'sections_duration_mean',
            'sections_duration_var',
            'loudness_var',
            'tempo_var',
            'key_var',
            'mode_var',
            'segments_num',
            'segments_duration_mean',
            'segments_duration_var',
            'pitches_mean',
            'pitches_var',
            'timbre_mean',
            'timbre_var',
            'tatums_num',
            'tatums_duration_mean',
            'tatums_duration_var',
            'time_signature',
            'key',
            'mode'
           ]

description = df[num_cols].describe()
num_zeros = (df[num_cols] == 0).sum()
null_data = df[df[num_cols].isnull().any(axis=1)]

#%% Save pandas profiler report

report = pp.ProfileReport(df)
report.to_file('profile_report.html')
    



