# Playlist-curator
This repository explores the possibilities of the features available through the Spotify API (via spotipy in this repo). The goal is to make useful playlist generators to enhance my Spotify experience.

## General files

feature_collection.py is the main file containing functions used to connect to the Spotify API and collect features for specified songs/playlists. The main function, called for example in jazz_df_prep.py, is get_playlist_df() which takes as parameters username and playlist id to read in songs from a given playlist. 

jazz_df_prep.py collects and prepares a dataset for training using functions in feature_collection.py.

Jazzy_notebook.ipynb contains a range of analyses and visualizations of the data collected in jazz_df_prep.py.

Jazzy_training_notebook.ipynb trains a neural network model based on data from jazz_df_prep.py.

jazzy_weekly.py reads data features from weekly generated playlists (Discover Weekly and New Songs Friday Canada/Norway/Sweden/UK/Naija) using feature_collection.py, and uses the model from Jazzy_training_notebook.ipynb to score the data and identify songs with a "jazzy" feel that I can then curate these weekly playlists by.

## Jazzy playlist generator

This project aims to identify "jazzy" songs. "Jazzy" songs do not only refer to songs of the jazz genre, but songs that have a "jazzy" feel to, such as funk and hip hop with jazzy or funky beats. 

#### Training

The training dataset consists of 1487 "jazzy" songs, and 2121 "non-jazzy" songs. The dataset was built by building two playlists in Spotify. The "jazzy" dataset was built by selecting a range of albums with the "jazzy" feel I seek and adding them to the playlist. The same approach was used for the "non-jazzy" playlist, except the albums were songs with a feel I do not want for this particular playlist, from various genres and artists. I generally chose albums I like for both playlists. The model is trained in a Jupyter notebook named "Jazzy_training_notebook".

#### Scoring

After the model is trained, it is saved and loaded in another file named "jazzy_weekly". This model loads data from Spotify playlists and prepares it for scoring. Then finally a dataset with songs with a score >=0.5 are returned. These songs can then be used to curate your weekly playlist in Spotify.

