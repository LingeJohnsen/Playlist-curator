import pandas as pd

from playlist_config import PLAYLIST_CREATOR, NOR_HIP_HOP_DAILY_MIX
from config import SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET

import spotipy
import spotipy.util as util

df = pd.read_csv('nor-hip-hop.csv', sep='|')

# Random sample of 60 songs
daily_mix_songs = df['track_id'].sample(n=30).tolist()

# Save songs to playlist - replace songs already in playlist
token = util.prompt_for_user_token(
                                    PLAYLIST_CREATOR,
                                    scope = 'playlist-modify-public',
                                    client_id = SPOTIPY_CLIENT_ID, 
                                    client_secret = SPOTIPY_CLIENT_SECRET,
                                    redirect_uri = 'http://localhost:8888/callback'
                                  )
sp = spotipy.Spotify(token)
sp.user_playlist_replace_tracks(
                                user = PLAYLIST_CREATOR,
                                playlist_id = NOR_HIP_HOP_DAILY_MIX,
                                tracks = daily_mix_songs
                               )
