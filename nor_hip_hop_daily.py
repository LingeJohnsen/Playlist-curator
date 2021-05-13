import feature_collection as mdat
import pandas as pd
import joblib
#%%
# from config import SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET

# Start with a base dict of favourite Norwegian hip artists to base list around
start_dict = {
              'Karpe': '3X23gpg1vPacr0hBARyxtN',
              'Klovner i kamp': '4CSD8L9KIPNIYFaUQPgENh',
              'Jaa9 & OnklP': '4EixyNiEp6dEMjtXdcyrZh',
              'Kamelen': '59WNMskn4tSvgnWKXHXj61',
              'Svartepetter': '3Ruv0CUM5HRukqTFs4ZJf1',
              'Sirkel Sag': '4pgT4VCXp66MRcH1L6Ewi6',
              'Tungtvann': '6FJ34pKFwXJc6zSHCpYplu',
              'Don Martin': '5XUjG97WeK2UDtIm9GST9t',
              'Gatas Parlament': '4IrzLo1bn7YuhBd0gSn9Ti',
              'Arif': '3l4RyQwQ0kHZ9Q9cQbRNMr',
              'Oscar Blesson': '34yalNbmu76FgEtORg2yVp',
              'KingSkurkOne': '0GJpcPq3tHjeDUB8I05Wxc'
             }

sp = mdat.authenticate()

artist_list = []

for artist, aid in start_dict.items():
    artist_list.append((artist, aid))
    related_artists = sp.artist_related_artists(aid)
    for i in related_artists['artists']:
        artist_list.append((i['name'], i['id']))

# Remove duplicates
final_list = list(set(artist_list))

df_list = []

for tpl in final_list:
    df_list.append(mdat.get_artist_df(tpl[1]))
    
#%%
# Create base df
df = pd.concat(df_list) 

# Load model 
model = joblib.load('hip-hop.joblib.dat')

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
            
X = df.copy()
X = X[features]

# Add column used during training for ColumnTransformer to work
X['dev_flag'] = 0

# Made predictions and filter out predictions that are not 1
df['predictions'] = model.predict(X)
songs = df[df['predictions']==1]

# Save songs in dataset to pick from for daily mix
songs.to_csv('nor-hip-hop.csv', sep='|')


   