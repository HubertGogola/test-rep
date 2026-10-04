# Dynamic Player DNA

Public portfolio demo inspired by my bachelor thesis at AGH University of Krakow.

> **Public demo notice:** All player-level data included in this repository and used in the Streamlit application are fully synthetic. They do not represent real players or real match observations.

## Thesis context
- Academy data from **Cracovia**
- Season **2025/2026**
- Main motor dataset: **799 observations / 37 players**
- Detailed technical-tactical dataset: **260 observations / 33 players**
- Main dynamic modelling based on GPS/motor data because of better completeness

## Pipeline
`Data preparation → PCA → K-means → GMM → Dynamic Player DNA → HMM → Kaplan–Meier`

## Selected results

The following are aggregate results from the original thesis analysis:

- PC1–PC3: **83.80% explained variance**
- K-means: **2 clusters**, silhouette **0.335**
- GMM: **2 profiles**, ARI **1.000**
- HMM: **3 activity states**
- Median first lower→higher transition: **8 appearances**

## Privacy

The original thesis used real academy data from Cracovia.

**No raw player-level, match-level or GPS data from Cracovia are published in this repository.**

All player-level data used in the public Streamlit demo are **fully synthetic** and were generated specifically for demonstration purposes.

The synthetic players and match observations:
- do not represent real players,
- do not represent real matches,
- are not anonymised copies of the original dataset.

Selected aggregate thesis results are presented only to explain the methodology and analytical findings.

## Demo data

The file:

`data/demo_player_data.csv`

contains synthetic player-match observations generated exclusively for this public demo.

For example, player IDs such as `P001`, `P002` or `P003` refer to fictional players created for demonstration purposes.

The purpose of the synthetic dataset is to reproduce the analytical workflow without publishing confidential player data.

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
