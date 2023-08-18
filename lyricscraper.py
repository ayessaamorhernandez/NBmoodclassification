from requests import Timeout
import api_key
import lyricsgenius as lg
import pandas as pd #installed for data/dataframe manipulation

# import random as rand

from langdetect import detect #run `pip install langdetect` first
from langdetect import DetectorFactory
DetectorFactory.seed = 0 # for consistent result if a text is too short

from googletrans import Translator # run `pip install googletrans` or `$ pip3 install googletrans==3.1.0a0`
from google_trans_new import google_translator #run `pip install google_trans_new` first

import main

client_access_token = api_key.your_client_access_token
lyrics_genius = lg.Genius(client_access_token)
lyrics_genius.timeout = 20

translator = Translator()

def get_lyrics():
    song_df = pd.read_csv("moody_lyrics_data\ml_pn_balanced.csv", index_col='index')
    lyrics_found = lyrics_not_found = 0
    lyrics_col = []
    for info in song_df.values:
        try:
            song = lyrics_genius.search_song(info[1], info[0])
        except Timeout as e:
            continue
        if song:
            lyrics_found += 1
            # For detecting if lyrics are english
            song_in_en = detect(song.lyrics)
            if song_in_en == 'en':
                lyrics_col.append(song.lyrics)
            else:
                lyrics_col.append("Not Found.")
        else:
            lyrics_not_found += 1
            lyrics_col.append("Not Found.")
    song_df[main.LYRICS] = lyrics_col
    song_df = song_df[song_df[main.LYRICS] != 'Not Found.']
    song_df.to_csv('with_lyrics.csv')
    # print(lyrics_found, lyrics_not_found)
    
if __name__ == '__main__':
    get_lyrics()