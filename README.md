# 🎧 LastFM Network Analysis & Node Classification

This project explores a social network of LastFM users from Asian countries. Using graph theory and machine learning, it extracts insights from user connections and listening preferences, and predicts users’ countries based on graph-based features.

## 📁 Project Structure

```
lastfm-network-ml-analysis/
│
├── part1_analysis.py         # Graph analysis, centrality, communities, link prediction
├── part2_modeling.ipynb      # Node classification using engineered features and XGBoost
├── requirements.txt          # Required Python packages
├── README.md
└── data/
    ├── lastfm_asia_edges.csv
    ├── lastfm_asia_target.csv
    └── lastfm_asia_features.json
```

## 🧠 Part 1 – Graph Analysis

Using NetworkX and community detection techniques:
- Cleaned the graph by removing low-degree nodes
- Computed centrality metrics (Degree, Closeness, Betweenness, Eigenvector, PageRank)
- Analyzed community structure and artist preferences
- Performed link prediction with Jaccard and Adamic-Adar indices

## 🤖 Part 2 – Node Classification

The goal is to predict a user's country using graph-based features:
- Engineered node-level features from:
  - Graph structure (centrality, community)
  - Neighbor country distribution
  - Listened artists
  - Node2Vec embeddings
- Trained an XGBoost model for multi-class classification
- Evaluated accuracy and performed sensitivity analysis

## 📊 Dataset

> Dataset used in this project is derived from the [LastFM Asia Social Network], collected via public API.

- **Nodes:** 7,624 users
- **Edges:** 27,806 mutual friendships
- **Features:** Artists followed, country (target)
- **Source:** [FEATHER Project](https://github.com/benedekrozemberczki/FEATHER)

### Citation

```
@misc{rozemberczki2020characteristic,
  title={Characteristic Functions on Graphs: Birds of a Feather, from Statistical Descriptors to Parametric Models},
  author={Benedek Rozemberczki and Rik Sarkar},
  year={2020},
  eprint={2005.07959},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```

## ⚙️ Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/MoriaCohen8/lastfm-network-ml-analysis.git
cd lastfm-network-ml-analysis
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

3. Run:
- `part1_analysis.py` for network analysis
- `part2_modeling.ipynb` for node classification (open with Jupyter)

## 🧠 Key Insights

- Centrality measures identify influential users and reveal network structure
- Community structure reflects both listening preferences and geography
- Combining structural and semantic features yields strong predictive performance
