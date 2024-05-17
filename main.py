import streamlit as st
import pickle
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from collections import defaultdict
from sklearn.metrics import euclidean_distances
from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd
# Load the recommend_songs function from pickle
# with open("recommend_songs.pkl", "rb") as f:
#   recommend_songs = pickle.load(f)

class Recommender:
    def __init__(self,dat):
        self.data=dat
        self.number_cols = ['valence', 'year', 'acousticness', 'danceability', 'duration_ms', 'energy', 'explicit',
 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'popularity', 'speechiness', 'tempo']
        self.X = self.data[self.number_cols]
        self.sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id="5dabebb01ce441e7a9ddbbb02cae5f42", client_secret="61f2cf92dbdd4a84bf46b7fb1fa99904"))
        self.song_cluster_pipeline = Pipeline([
          ('scaler', StandardScaler()),
          ('kmeans', KMeans(n_clusters=20, verbose=False))
        ], verbose=False)
        self.song_cluster_pipeline.fit(self.X)
    def select_numerical_features(self, data):
        return data.select_dtypes(include=[np.number])
    def find_song(self,name, year):
        song_data = defaultdict()
        results = self.sp.search(q= 'track: {} year: {}'.format(name,year), limit=1)
        if results['tracks']['items'] == []:
            return None

        results = results['tracks']['items'][0]
        track_id = results['id']
        audio_features = self.sp.audio_features(track_id)[0]

        song_data['name'] = [name]
        song_data['year'] = [year]
        song_data['explicit'] = [int(results['explicit'])]
        song_data['duration_ms'] = [results['duration_ms']]
        song_data['popularity'] = [results['popularity']]

        for key, value in audio_features.items():
            song_data[key] = value

        return pd.DataFrame(song_data)
    def get_song_data(self,song, spotify_data):
    
        try:
            song_data = spotify_data[(spotify_data['name'] == song['name']) 
                                    & (spotify_data['year'] == song['year'])].iloc[0]
            return song_data

        except IndexError:
            return self.find_song(song['name'], song['year'])
    def get_mean_vector(self,song_list, spotify_data):
        metadata_cols = ['name', 'year', 'artists']
        n_features = len(spotify_data.columns) - len(metadata_cols)
        song_vectors = []
        for song in song_list:
            song_data = self.get_song_data(song, spotify_data)
            if song_data is None:
                print('Warning: {} does not exist in Spotify or in database'.format(song['name']))
                continue

        song_vector = song_data[number_cols].values[:n_features]
        song_vectors.append(song_vector)

        if not song_vectors:
            raise ValueError("No songs found with matching features in Spotify data")
        song_matrix = np.vstack(song_vectors)
        return np.mean(song_matrix, axis=0)
    def flatten_dict_list(self,dict_list):
    
        flattened_dict = defaultdict()
        for key in dict_list[0].keys():
            flattened_dict[key] = []

        for dictionary in dict_list:
            for key, value in dictionary.items():
                flattened_dict[key].append(value)

        return flattened_dict
    def recommend_songs(self,song_list, spotify_data, n_songs=10):
        metadata_cols = ['name', 'year', 'artists']
        song_dict = self.flatten_dict_list(song_list)

        song_center = self.get_mean_vector(song_list, spotify_data)
        scaler = self.song_cluster_pipeline.steps[0][1]
        scaled_data = scaler.transform(spotify_data[self.number_cols])
        scaled_song_center = scaler.transform(song_center.reshape(1, -1))
        distances = cdist(scaled_song_center, scaled_data, 'cosine')
        index = list(np.argsort(distances)[:, :n_songs][0])

        rec_songs = spotify_data.iloc[index]

        rec_songs = rec_songs[~rec_songs['name'].isin(song_dict['name'])]
        return rec_songs[metadata_cols].to_dict(orient='records')
    
    
with open("data.pkl", "rb") as f:
  spotify_data = pickle.load(f)
recommender=Recommender(spotify_data)
number_cols = ['valence', 'year', 'acousticness', 'danceability', 'duration_ms', 'energy', 'explicit',
              'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'popularity', 'speechiness', 'tempo']

def get_song_info():
  """
  Collects song name and year as user input and returns a list of dictionaries.
  """
  song_name = st.text_input("Enter Song Name:")
  song_year = st.number_input("Enter Song Year:", min_value=1900, max_value=2024)
  if not song_name or song_year < 1900 or song_year > 2024:
    return None  # Indicate invalid input
  return [{'name': song_name, 'year': song_year}]

# Streamlit App

st.title("Music Recommender")

# Collect song information
song_info = get_song_info()

# Button to trigger recommendation
if st.button("Recommend Songs"):
  if not song_info:
    st.error("Please enter a valid song name and year (1900-2024)")
  else:
    # Call your recommendation function with list of song info and spotify data
    recommended_songs = recommender.recommend_songs(song_info, spotify_data, n_songs=10)

    # Display recommended songs if any
    if recommended_songs:
      st.subheader("Recommended Songs:")
      for song in recommended_songs:
        st.write(f"{song['name']} ({song['year']}) - {song['artists']}")
    else:
      st.warning("No songs found matching your criteria.")
