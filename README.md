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

The training dataset consists of 1487 "jazzy" songs, and 3113 "non-jazzy" songs. The dataset was built by building two playlists in Spotify. The "jazzy" dataset was built by selecting a range of albums with the "jazzy" feel I seek and adding them to the playlist. More songs were added from the weekly generated playlist, as I added true positives to the training dataset. The same approach was used for the "non-jazzy" playlist, except the albums were songs with a feel I do not want for this particular playlist, from various genres and artists. I also added false positives from each weekly playlist to the "non-jazzy" playlist. I generally chose albums I like for both playlists. The model is trained in a Jupyter notebook named "Jazzy_training_notebook".

### Scoring

After the model is trained, it is saved and loaded in another file named "jazzy_weekly". This model loads data from Spotify playlists and prepares it for scoring. Then finally a dataset with songs with a score >=0.5 are returned. These songs can then be used to curate your weekly playlist in Spotify.

all positive (1) predictions from the model. These songs are then used to refresh the Norwegian Hip Hop Weekly playlist in Spotify.

### Metrics on test set

AUC: 0.79
Precision: 0.74
Recall: 0.67

# General files

feature_collection.py is the main file containing functions used to connect to the Spotify API and collect features for specified songs/playlists. The main function, called for example in jazz_df_prep.py, is get_playlist_df() which takes as parameters username and playlist id to read in songs from a given playlist. 

## Notes about config.py and playlist_config.py

IDs, username, and Spotify token, string parameters necessary to load data via the Spotify API, are stored as parameter strings in two python files named config.py and playlist_config.py. These files are not stored in this repository, but they are called in the files feature_collection.py, jazzy_df_prep.py, jazzy_weekly.py, norwegian_hip_hop_training_df.py, nor_hip_hop_artist_df.py, nor-hip-hop-weekly-mix-generator.py. 

