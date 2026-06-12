# 🌽 Kenya Maize Price Forecasting — Dashboard

A Streamlit dashboard for visualising and forecasting White Maize wholesale prices
across five Kenyan counties: **Kiambu, Kirinyaga, Mombasa, Nairobi, Uasin-Gishu**.

---

## 📁 Project Folder Structure

```
MaizePrediction/
├── data/
│   ├── kamis_maize_prices.csv
│   ├── agriBORA_maize_prices.csv
│   └── weather_kenya_all_counties_2021_2025 (2).csv
├── notebook/
│   └── Copy_of_Final_one.ipynb
└── dashboard/                  ← you are here
    ├── app.py                  ← main Streamlit app
    ├── requirements.txt
    ├── README.md
    └── .streamlit/
        └── config.toml
```

---

## 🚀 Step-by-Step Deployment Guide

### OPTION A — Run Locally

**Step 1 — Install Python (3.9 or later)**
Download from https://www.python.org/downloads/ if not already installed.

**Step 2 — Create and activate a virtual environment**
```bash
# Create venv
python -m venv venv

# Activate — Windows
venv\Scripts\activate

# Activate — macOS / Linux
source venv/bin/activate
```

**Step 3 — Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 4 — Run the app**
```bash
# From inside the dashboard/ folder:
streamlit run app.py
```

The app opens automatically at **http://localhost:8501**

---

### OPTION B — Deploy on Streamlit Community Cloud (free)

**Step 1 — Push your project to GitHub**

1. Create a new **public** GitHub repository (e.g. `maize-prediction`)
2. Upload the entire `MaizePrediction/` folder structure — your repo must include:
   - `dashboard/app.py`
   - `dashboard/requirements.txt`
   - `dashboard/.streamlit/config.toml`
   - `data/` folder with the three CSV files

```bash
git init
git add .
git commit -m "Initial commit — maize price forecasting dashboard"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/maize-prediction.git
git push -u origin main
```

**Step 2 — Sign up / log in to Streamlit Community Cloud**
Go to https://share.streamlit.io and sign in with GitHub.

**Step 3 — Deploy the app**
1. Click **"New app"**
2. Select your GitHub repository
3. Set **Branch** → `main`
4. Set **Main file path** → `dashboard/app.py`
5. Click **"Deploy!"**

Streamlit installs `requirements.txt` automatically and gives you a public URL
like `https://your-app-name.streamlit.app`.

**Step 4 — (Optional) Make data files private**
If you don't want raw CSVs public, store them in **Google Drive** and read them
via `gdown` or the Google Drive API — add `gdown` to `requirements.txt` and
replace the `load_raw_data()` function paths with download calls.

---

### OPTION C — Deploy on a cloud VM (AWS / GCP / Azure)

**Step 1 — SSH into your VM and clone the repo**
```bash
git clone https://github.com/YOUR_USERNAME/maize-prediction.git
cd maize-prediction/dashboard
```

**Step 2 — Install dependencies**
```bash
pip install -r requirements.txt
```

**Step 3 — Run with nohup (keeps running after you close the terminal)**
```bash
nohup streamlit run app.py --server.port 8501 --server.headless true &
```

**Step 4 — Open firewall port 8501** in your cloud provider's security group/firewall rules.

Access the app at `http://YOUR_VM_IP:8501`

---

## 🛠️ Troubleshooting

| Problem | Fix |
|---|---|
| `FileNotFoundError` for CSVs | Check that `data/` folder is one level above `dashboard/` |
| `ModuleNotFoundError: lightgbm` | Run `pip install lightgbm` — app works without it but uses XGBoost |
| App very slow on first load | Normal — model trains on startup; subsequent loads use cache |
| Streamlit version mismatch | Pin `streamlit==1.35.0` in `requirements.txt` |

---

## 📊 Dashboard Features

| Tab | What it shows |
|---|---|
| 📈 Price Trends | Historical KAMIS & AgriBORA prices with smoothing + seasonality |
| 🔮 Forecast | Iterative multi-county price forecast with confidence band + CSV download |
| 🏆 Model Performance | MAE / RMSE / MAPE / R², per-county errors, feature importances |
| 🌤️ Weather Impact | Temperature & rainfall vs price, correlation heatmap |
