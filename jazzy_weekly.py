import feature_collection as mdat
from playlist_config import PLAYLIST_CREATOR, WEEK_11_2021, NEW_FRIDAY_11_2021
import pandas as pd
import keras
from joblib import load

# Load model
jazzy_model = keras.models.load_model('jazz_model')

# Load weekly playlists
weekly_df = mdat.get_playlist_df(PLAYLIST_CREATOR, WEEK_11_2021)
new_df = mdat.get_playlist_df(PLAYLIST_CREATOR, NEW_FRIDAY_11_2021)

features = [
            'danceability',
            'energy',
            'speechiness',
            'acousticness',
            'instrumentalness',
            'liveness',
            'valence',
            'num_samples',
            'end_of_fade_in',
            'loudness',
            'tempo',
            'key',
            'mode',
            'bars_num',
            'bars_duration_var',
            'beats_duration_var',
            'sections_num',
            'sections_duration_mean',
            'sections_duration_var',
            'loudness_var',
            'tempo_var',
            'key_var',
            'mode_var',
            'segments_duration_var',
            'segments_duration_mean',
            'pitches_mean',
            'pitches_var',
            'timbre_mean',
            'timbre_var',
            'tatums_duration_var'
           ]
X = pd.concat([weekly_df.copy(), new_df.copy()], axis=0)
X = X[features]

# Find the jazzy songs - first have to scale model as in trained dataset
scaler = load('scaler.joblib')
X_scaled = scaler.transform(X)

# Find songs to try out
songs = pd.concat([weekly_df, new_df], axis=0)
songs['Scores'] = jazzy_model.predict(X_scaled)
songs['Predictions'] = songs['Scores'].round(0)
songs = songs[songs['Predictions']==1]
