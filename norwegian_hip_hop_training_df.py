import feature_collection as mdat
from playlist_config import PLAYLIST_CREATOR
import playlist_config as pc
import pandas as pd

# Training set
train_pos_df = mdat.get_playlist_df(PLAYLIST_CREATOR, pc.HIP_HOP_TRAIN)
train_pos_df['label'] = 1
train_neg_df = mdat.get_playlist_df(PLAYLIST_CREATOR, pc.NON_HIP_HOP_TRAIN)
train_neg_df['label'] = 0
train_df = pd.concat([train_pos_df, train_neg_df])

train_df.to_csv('hip_hop_train.csv', sep='|')

# Dev set
nor_pos_dev = mdat.get_playlist_df(PLAYLIST_CREATOR, pc.NOR_HIP_HOP_DEV)
nor_pos_dev['label'] = 1
nor_neg_dev = mdat.get_playlist_df(PLAYLIST_CREATOR, pc.NOR_NON_HIP_HOP_DEV)
nor_neg_dev['label'] = 0
dev_df = pd.concat([nor_pos_dev, nor_neg_dev])

dev_df.to_csv('nor_hip_hop_dev.csv', sep='|')

# Test set
nor_pos_test = mdat.get_playlist_df(PLAYLIST_CREATOR, pc.NOR_HIP_HOP_TEST)
nor_pos_test['label'] = 1
nor_neg_test = mdat.get_playlist_df(PLAYLIST_CREATOR, pc.NOR_NON_HIP_HOP_TEST)
nor_neg_test['label'] = 0
test_df = pd.concat([nor_pos_test, nor_neg_test])

test_df.to_csv('nor_hip_hop_test.csv', sep='|')
