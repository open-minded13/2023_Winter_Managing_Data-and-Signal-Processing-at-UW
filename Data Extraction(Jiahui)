import spotipy 
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import csv
# load credentials from .env file
client_id = "97ac54739a3f4be5b27185d88103a733"
client_secret = "55298bb2a27c4c91b3c996440be5f1ee"
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
# Specify the ID of the Spotify playlist you want to extract
playlist_id = '1YC2hYS5awhGQBNaCObjyK'
# Get all the tracks in the playlist
tracks = sp.playlist_tracks(playlist_id)

# Create a new CSV file to store the dataset
with open('classic2.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # Write the header row of the CSV file
    header = ['Track Name', 'Artist', 'Album', 'Duration (ms)', 'danceability', 'energy','key','loudness','mode','speechiness','acousticness','instrumentalness','liveness', 'valence','tempo', 'track_id']
    writer.writerow(header)
    
    # Loop through each track and extract its information
    for track in tracks['items']:
        # Get the track information
        track_name = track['track']['name']
        artist = track['track']['artists'][0]['name']
        album = track['track']['album']['name']
        duration_ms = track['track']['duration_ms']
        track_id = track['track']['id']
        
        # Get the audio features for the track
        audio_features = sp.audio_features(track_id)[0]
        
        # Extract the audio features
        danceability = audio_features['danceability']
        energy = audio_features['energy']
        key = audio_features['key']
        loudness = audio_features['loudness']
        mode = audio_features['mode']
        speechiness = audio_features['speechiness']
        acousticness = audio_features['acousticness']
        Instrumentalness = audio_features['instrumentalness']
        liveness = audio_features['liveness']
        valence = audio_features['valence']
        tempo = audio_features['tempo']
        
        # Write the row to the CSV file
        row = [track_name, artist, album, duration_ms, danceability, energy,key,loudness,mode,speechiness,acousticness,instrumentalness,liveness, valence,tempo, track_id]
        writer.writerow(row)
        
