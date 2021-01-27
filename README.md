# Playlist-generator
This repository explores the possibilities of the features available through the Spotify API (via spotipy in this repo). The goal is to make useful playlist generators to enhance my Spotify experience.

## General files

feature_collection.py is the main file containing functions used to connect to the Spotify API and collect features for specified songs/playlists. The main function, called for example in jazz_df_prep.py, is get_playlist_df() which takes as parameters username and playlist id to read in songs from a given playlist. 

## Jazzy playlist generator

This project aims to identify "jazzy" songs. "Jazzy" songs do not only refer to songs of the jazz genre, but also songs and genres influenced by jazz, such as funk and certain elements of hip hop. 

The training dataset consists of 940 "jazzy" songs, and 2121 "non-jazzy" songs. The dataset was built by building two playlists in Spotify. The "jazzy" dataset was built by selecting a range of albums with the "jazzy" feel I seek and adding them to the playlist. The same approach was used for the "non-jazzy" playlist, except the albums were songs with a feel I do not want for this particular playlist, from various genres and artists. I generally chose albums I like for both playlists. 
