import os
import random
import pickle
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv

# 1. Load configuration and paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, '.env'))

# 2. Authenticate with Spotify API safely
client_id = os.getenv('SPOTIPY_CLIENT_ID')
client_secret = os.getenv('SPOTIPY_CLIENT_SECRET')

if not client_id or not client_secret:
    print("⚠️ Warning: Spotify credentials missing in .env file. Acoustic fallback will not work.")
else:
    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
        client_id=client_id, client_secret=client_secret
    ))

# 3. Load Datasets and Trained Models
try:
    hot_df = pd.read_csv(os.path.join(BASE_DIR, 'data/hot_songs.csv'))
    clustered_df = pd.read_csv(os.path.join(BASE_DIR, 'data/clustered_songs.csv'))
    
    with open(os.path.join(BASE_DIR, 'models/scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
        
    with open(os.path.join(BASE_DIR, 'models/kmeans_model.pkl'), 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError as e:
    print(f"❌ Error loading files: {e}\nMake sure you ran notebooks 1, 2, and 3 first!")
    exit()


def get_spotify_features(song_title):
    """Searches Spotify for the song, extracts metadata, and pulls audio features."""
    # Search for the track
    results = sp.search(q=f"track:{song_title}", type='track', limit=1)
    tracks = results['tracks']['items']
    
    if not tracks:
        return None, None
    
    track = tracks[0]
    track_id = track['id']
    track_name = track['name']
    artist_name = track['artists'][0]['name']
    
    # Get audio features
    features = sp.audio_features([track_id])[0]
    if not features:
        return None, None
        
    # Extract only the features the K-Means model expects (must match Notebook 3)
    feature_keys = ['danceability', 'energy', 'loudness', 'speechiness', 
                    'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
    
    extracted_features = [features[key] for key in feature_keys]
    
    return extracted_features, f"'{track_name}' by {artist_name}"


def run_recommender():
    print("\n" + "="*50)
    print("Welcome to the GNOD Hybrid Music Recommender System!")
    print("="*50 + "\n")
    
    user_input = input("Enter a song you currently love: ").strip().lower()
    
    if not user_input:
        print("You didn't type anything. Goodbye!")
        return

    # --- PATH A: Check if song is in Billboard Hot 100 ---
    if user_input in hot_df['song'].values:
        print("\n🔥 That's a scorching hot track right now!")
        
        # Pull a random song from the Hot 100 pool (excluding the input if possible)
        recommendation_pool = hot_df[hot_df['song'] != user_input]
        suggestion = recommendation_pool.sample(1).iloc[0]
        
        print(f"Since you like current hits, we recommend: '{suggestion['song'].title()}' by {suggestion['artist'].title()}")
        return

    # --- PATH B: Acoustic Similarity via Spotify and K-Means ---
    print("\n🎵 Not a current top hit. Analyzing acoustic vibe on Spotify...")
    
    features, formal_name = get_spotify_features(user_input)
    
    if features is None:
        print("😔 Sorry, we couldn't find that song or its audio profile on Spotify. Try another track!")
        return
        
    print(f"Found track: {formal_name}")
    
    # Scale features and predict cluster using the saved model artifacts
    features_scaled = scaler.transform([features])
    predicted_cluster = model.predict(features_scaled)[0]
    
    # Get all songs from our pool that share the same cluster
    cluster_pool = clustered_df[clustered_df['cluster'] == predicted_cluster]
    
    if not cluster_pool.empty:
        suggestion = cluster_pool.sample(1).iloc[0]
        print(f"\n🔮 Based on its acoustic profile, we think you'll love this vibe:")
        print(f"👉 '{suggestion['song'].title()}' by {suggestion['artist'].title()}")
    else:
        print("We found the vibe, but our database needs more songs to make a match!")


if __name__ == "__main__":
    run_recommender()