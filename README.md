# Gnod Music Recommender Engine (MVP)

An agile, data-driven Minimum Viable Product (MVP) built to expand Gnod’s core recommendation architecture. This hybrid engine addresses the traditional cold-start limitations of collaborative filtering by incorporating real-time global popularity metrics alongside structural acoustic profiling.

---

## 📈 Business Context & Strategic Vision
Currently, Gnod relies strictly on collaborative filtering. While powerful, this methodology restricts recommendation capabilities to established artist networks and depends heavily on continuous user inputs. 

To secure strategic partnerships with streaming platforms like **Spotify, Apple Music, or Deezer**, Gnod must demonstrate an capability to handle deep granularity (song-level instead of artist-level tracking) and match audio characteristics directly. This engine introduces two parallel recommendation vectors designed to maximize user engagement and diversify revenue streams beyond static advertising.

---

## 🛠️ System Architecture & Logic Flow

The engine maps incoming user inputs across a dual-path decision matrix before serving recommendations:

```text
                  [ User Inputs a Song ]
                            |
             Is it in the Billboard Hot 100?
               /                         \
            (YES)                        (NO)
             /                             \
    [Path A: Pop Vector]          [Path B: Acoustic Profile Vector]
   Recommend another hot song       1. Search & pull tracks from Spotify API
   from the current top charts.     2. Extract quantitative audio features
                                    3. Scale features via StandardScaler
                                    4. Map to structural cluster using K-Means
                                    5. Serve highly similar acoustic match

```

---

## 📂 Repository Structure

The project layout adheres strictly to industry-standard data science file distribution templates:

```text
gnod-audio-recommender/
│
├── .gitignore                  # Prevents committing API keys & data artifacts
├── README.md                   # Project documentation and C-Suite summary
├── requirements.txt            # Python environment dependencies
├── .env                        # Local storage for Spotify Developer credentials
│
├── data/                       # Scraped and engineered target databases
│   ├── hot_songs.csv           # Cleaned Billboard Hot 100 records
│   ├── spotify_songs_pool.csv  # Raw audio features from playlist deep-dives
│   └── clustered_songs.csv     # Final song database mapped with cluster labels
│
├── models/                     # Serialized machine learning artifacts
│   ├── scaler.pkl              # Pre-trained StandardScaler object
│   └── kmeans_model.pkl        # Pre-trained K-Means Clustering model (k=8)
│
├── notebooks/                  # Step-by-step modular experimentation sandboxes
│   ├── 1_web_scraping.ipynb    # Billboard parsing pipeline
│   ├── 2_spotify_api.ipynb     # Spotipy feature collection scripts
│   └── 3_clustering_modeling.ipynb # Feature engineering, Elbow method, Modeling
│
└── src/                        # Production-ready executable applications
    ├── __init__.py
    └── recommender.py          # Operational terminal interface tool

```

---

## 🔬 Machine Learning Pipeline & Data Sources

### 1. Web Scraping (`1_web_scraping.ipynb`)

* **Source:** Billboard Hot 100 Chart.
* **Tech:** `BeautifulSoup4` and `requests`.
* **Output:** Extracts and normalizes current viral songs and artists to serve as the baseline popular recommendation deck.

### 2. Feature Extraction (`2_spotify_api.ipynb`)

* **Source:** Spotify Developer Web API via `Spotipy`.
* **Scope:** Curated extraction across diverse genre buckets (Jazz, Electronic, Rock, Classical, Hip-Hop) to establish a comprehensive data footprint.
* **Metrics Tracked:** `danceability`, `energy`, `loudness`, `speechiness`, `acousticness`, `instrumentalness`, `liveness`, `tempo`, and `valence`.

### 3. Unsupervised Clustering (`3_clustering_modeling.ipynb`)

* **Preprocessing:** Standardized variables utilizing `StandardScaler` to remove distance skew caused by divergent mathematical scales (e.g., `tempo` vs `acousticness`).
* **Modeling:** Built an unsupervised **K-Means Clustering** routine.
* **Hyperparameter Tuning:** Selected an optimal configuration ($K=8$) mapped directly via statistical evaluation using the **Elbow Method** (minimizing inertia changes) and cross-checked against **Silhouette Scores**.

---

## 🚀 Quick Start / How to Run

### Prerequisite Setup

1. Clone this repository onto your machine.
2. Sign in to the [Spotify Developer Dashboard](https://developer.spotify.com/) and register a new application to generate your API credentials.
3. Create a `.env` file in the root directory and paste your credentials explicitly:

```env
   SPOTIPY_CLIENT_ID='your_actual_client_id'
   SPOTIPY_CLIENT_SECRET='your_actual_client_secret'

```

### Installation & Execution

Initialize your local package requirements and trigger the interactive engine environment:

```bash
# Install required computational libraries
pip install -r requirements.txt

# Execute the application interface script
python src/recommender.py

```

---

## 🗺️ Product Roadmap & Agile Next Steps

Moving past this foundational week-one sprint, the technical roadmap scales this engine into production:

1. **Infrastructure Scalability:** Migrate flat CSV processing pipelines over to cloud databases (PostgreSQL/AWS RDS) to scale processing constraints effortlessly past millions of catalog tracks.
2. **Hybrid Scoring Matrix:** Merge existing user collaborative filtering logs with these newly engineered structural acoustic clusters to unlock a comprehensive hybrid scoring equation.
3. **Web Microservice Deployment:** Uncouple the interactive environment from terminal scripts, adapting the engine logic into a lightweight API wrapper (Flask/FastAPI) to update the current live layout found across `gnoosic.com`.

```

```
