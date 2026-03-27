# SKU Demand Forecasting with Clustering

Comparing **K-Means** and **DBSCAN** clustering strategies for segmenting retail SKUs and building cluster-level OLS demand prediction models on real-world weekly sales data.

---

## Overview

When forecasting demand across a large product catalogue, there's a fundamental trade-off:

- **Centralized model** — train one model on all SKUs. Simple, but ignores product heterogeneity.
- **Decentralized model** — train one model per SKU. Captures individuality, but noisy and slow.

This project explores a **middle ground**: group similar SKUs into clusters, then fit a separate OLS regression for each cluster. Two clustering approaches are compared — K-Means and DBSCAN — across different feature sets (averages only vs. averages + standard deviations).

---

## Dataset

Weekly sales data for **44 SKUs** from a tech-gadget e-commerce retailer spanning **98 weeks** (October 2016 – September 2018). Features include price, weekly sales, and various SKU-level attributes.

---

## Methods

### Clustering
| Method | Description |
|--------|-------------|
| **K-Means** | Partitions SKUs into k non-overlapping clusters. k selected via validation R². |
| **DBSCAN** | Density-based clustering. Identifies noise points; no need to specify k upfront. |

Features used for clustering:
- **Baseline:** average price + average weekly sales per SKU (MinMax scaled)
- **Extended:** + standard deviation of price and weekly sales

### Demand Model
For each cluster, an **OLS regression** is fit on all observations from SKUs assigned to that cluster. Performance is evaluated using:
- **Validation R²** — in-distribution fit on held-out validation set (20%)
- **OOS R²** — out-of-sample generalization on test set (20%)

Train / Validation / Test split: **60 / 20 / 20**

---

## Results

### K-Means

| Features | Clusters | Val R² | OOS R² | Time (s) |
|----------|----------|--------|--------|----------|
| Avg only | 10 | 0.5814 | 0.6241 | — |
| Avg + Std Dev | 10 | 0.5988 | **0.6258** | 0.91 |

### DBSCAN (`eps=0.2`, `min_samples=3`)

| Features | Val R² | OOS R² | Time (s) |
|----------|--------|--------|----------|
| Avg only | 0.0896 | 0.0683 | 0.047 |
| Avg + Std Dev | 0.1598 | 0.0824 | 0.157 |

### Takeaway

K-Means outperforms DBSCAN by a large margin on this dataset (OOS R² ~0.63 vs ~0.08). DBSCAN's fixed hyperparameters caused it to collapse most SKUs into a single dense cluster, undermining downstream model quality. K-Means benefits from richer features (adding std dev) and systematic k selection via validation.

---

## Project Structure

```
sku-demand-forecasting/
├── sku_demand_forecasting_clustering.ipynb   # Full analysis notebook
├── results_and_analysis.pdf                  # Results write-up and discussion
├── retail_sales_data.csv                     # Dataset
└── README.md
```

---

## Stack

- **Python 3** — pandas, numpy, scikit-learn, matplotlib, seaborn
- **Clustering** — `sklearn.cluster.KMeans`, `sklearn.cluster.DBSCAN`
- **Scaling** — `sklearn.preprocessing.MinMaxScaler`
- **Regression** — `sklearn.linear_model.LinearRegression`

---

## Usage

```bash
git clone https://github.com/pavelkoya/sku-demand-forecasting.git
cd sku-demand-forecasting
pip install pandas numpy scikit-learn matplotlib seaborn
jupyter notebook sku_demand_forecasting_clustering.ipynb
```

`retail_sales_data.csv`.

---

## Future Work

- Tune DBSCAN hyperparameters using a k-distance graph to find a more appropriate `eps`
- Try **HDBSCAN** for better handling of varying-density clusters
- Experiment with additional clustering features (e.g., seasonality, trend components)
- Compare against tree-based demand models (e.g., Random Forest, XGBoost) per cluster
