# Playlist-curator
This playlist curator uses the Spotify API together with classification models to generate curated Spotify playlists. 

# Jazzy weekly playlist

This playlist is built through a neural network model that curates various generic Spotify "Weekly discovery" playlists to produce a single playlist of new "jazzy" songs. "Jazzy" songs do not only refer to songs of the jazz genre, but songs that have a "jazzy" feel to, such as funk and hip hop with jazzy or funky beats. 

### Jazzy playlist files

feature_collection.py is the main file containing functions used to connect to the Spotify API and collect features for specified songs/playlists. The main function, called for example in jazz_df_prep.py, is get_playlist_df() which takes as parameters username and playlist id to read in songs from a given playlist. 

jazz_df_prep.py collects and prepares a dataset for training using functions in feature_collection.py.

Jazzy_notebook.ipynb contains a range of analyses and visualizations of the data collected in jazz_df_prep.py.

Jazzy_training_notebook.ipynb trains a neural network model based on data from jazz_df_prep.py.

jazzy_weekly.py reads data features from weekly generated playlists (Discover Weekly, Release Radar, Hot Hits Canada, and New Songs Friday Canada/Norway/Sweden/UK/Naija/France) using feature_collection.py, and uses the model from Jazzy_training_notebook.ipynb to score the data and identify songs with a "jazzy" feel that I can then curate these weekly playlists by.

### Training

The training dataset consists of 1487 "jazzy" songs, and 2121 "non-jazzy" songs. The dataset was built by building two playlists in Spotify. The "jazzy" dataset was built by selecting a range of albums with the "jazzy" feel I seek and adding them to the playlist. The same approach was used for the "non-jazzy" playlist, except the albums were songs with a feel I do not want for this particular playlist, from various genres and artists. I generally chose albums I like for both playlists. The model is trained in a Jupyter notebook named "Jazzy_training_notebook".

### Scoring

After the model is trained, it is saved and loaded in another file named "jazzy_weekly". This model loads data from Spotify playlists and prepares it for scoring. Then finally a dataset with songs with a score >=0.5 are returned. These songs can then be used to curate your weekly playlist in Spotify.

# Norwegian Hip Hop Weekly playlist

This is a playlist of 60 randomly chosen Norwegian hip hop songs. The songs are sampled from a prepared list of songs produced using the Spotify API "related artist" feature built around a set of pre-selected Norwegian Hip Hop artists that is curated with a trained hip hop model to filter out non-hip hop songs. 

### Norwegian hip hop files

norwegian_hip_hop_training_df.py collects and prepares datasets for training, validation, and testing using functions in feature_collection.py.

Hip_hop_analysis.ipynb contains a range of analyses and visualizations of the data collected in norwegian_hip_hop_training_df_prep.py.

Hip_hop_training.ipynb trains an XGBoost model based on data from norwegian_hip_hop_training_df.py.

nor_hip_hop_artist_df.py produces a large dataset to sample playlist songs from. The dataset is built using a core of 12 Norwegian hip hop artists used as parameters for the "related artists" functionality in the Spotify API, which is then curated by the trained hip hop model to weed out any non-hip hop songs. 

nor-hip-hop-weekly-mix-generator.py reads data prepared in nor_hip_hop_artist_df.py and randomly samples 60 songs from the playlist which is then exported to Spotify via the Spotify API.

### Training

The training dataset consists of 1,957 hip hop songs and 3,293 non-hip hop songs, mainly of American or British origin. The validation and test sets consists of 1,501 and 1,527 hip hop songs and 1,512 and 1,523 non-hip hop songs, mainly of Norwegian or Swedish origin. Each dataset was built by building two playlists in Spotify, one for "positives" and one for "negatives", through adding albums for the various genres. The playlists were then read and combined in norwegian_hip_hop_training_df.py.

### Scoring

After the model is trained, it is saved and loaded in another file named nor-hip-hop-weekly-mix-generator.py. This model loads data prepared in nor_hip_hop_artist_df.py and returns all positive (1) predictions from the model. These songs are then used to refresh the Norwegian Hip Hop Weekly playlist in Spotify.

# General files

feature_collection.py is the main file containing functions used to connect to the Spotify API and collect features for specified songs/playlists. The main function, called for example in jazz_df_prep.py, is get_playlist_df() which takes as parameters username and playlist id to read in songs from a given playlist. 

## Notes about config.py and playlist_config.py

IDs, username, and Spotify token, string parameters necessary to load data via the Spotify API, are stored as parameter strings in two python files named config.py and playlist_config.py. These files are not stored in this repository, but they are called in the files feature_collection.py, jazzy_df_prep.py, jazzy_weekly.py, norwegian_hip_hop_training_df.py, nor_hip_hop_artist_df.py, nor-hip-hop-weekly-mix-generator.py. 

