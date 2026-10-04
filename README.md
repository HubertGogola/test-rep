# Dynamic Player DNA

Public portfolio demo inspired by my bachelor thesis at AGH University of Krakow.

## Thesis context
- Academy data from **Cracovia**
- Season **2025/2026**
- Main motor dataset: **799 observations / 37 players**
- Detailed technical-tactical dataset: **260 observations / 33 players**
- Main dynamic modelling based on GPS/motor data because of better completeness

## Pipeline
`Data preparation → PCA → K-means → GMM → Dynamic Player DNA → HMM → Kaplan–Meier`

## Selected results
- PC1–PC3: **83.80% explained variance**
- K-means: **2 clusters**, silhouette **0.335**
- GMM: **2 profiles**, ARI **1.000**
- HMM: **3 activity states**
- Median first lower→higher transition: **8 appearances**

## Privacy
The original thesis used real academy data. **No raw Cracovia data is published here.**
The bundled CSV is fully synthetic and exists only to demonstrate the methodology safely.

## Run locally
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Deploy
Upload this repository to GitHub, then deploy `app.py` through Streamlit Community Cloud.

## Author
**Hubert Gogola**  
Football Data Scientist  
Bachelor graduate, Computer Science and Econometrics, AGH University of Krakow  
Continuing with MSc studies in Computer Science and Econometrics at AGH
