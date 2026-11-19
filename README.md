# Stock Market Recommendations Using Machine Learning

BAI702 Machine Learning 2 mini project, AMC Engineering College, Bengaluru (CSE - AI & ML), 2026-27.

**Team:** Tharun (1AM23CI172), Tharun Gowda (1AM24CI413), Vikram M.H (1AM23CI180), Santhosh (1AM24CI411)
**Guide:** Prof. Bhavya K Bharathan

## What it does
Predicts a **Buy / Not Buy** recommendation for six large-cap stocks (NVDA, MSFT, AAPL, AMZN, GOOGL, TSLA) from technical indicators, and compares five classifiers:

| Model | Key settings |
|---|---|
| Logistic Regression | L2 penalty, scaled features |
| Decision Tree | max_depth 6, min_samples_leaf 20 |
| Random Forest | 200 trees, max_depth 12 |
| Gradient Boosting | 150 estimators, lr 0.08 |
| XGBoost | 300 rounds, lr 0.05, logloss |

**Pipeline:** Yahoo Finance prices -> indicators (SMA spread, RSI-14, MACD, Bollinger position, volatility, 5-day momentum, 20-day trend, volume trend) -> label (Buy if price 20 sessions later is higher) -> chronological 80/20 split -> train -> Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrices, ROC curves, feature importance.

## Run in Google Colab
1. Open [colab.research.google.com](https://colab.research.google.com) -> **File -> Upload notebook** -> choose `stock_market_recommendation.ipynb`
   (or, after pushing to GitHub: **File -> Open notebook -> GitHub** and paste the repository URL).
2. **Runtime -> Run all.** Figures show in the notebook and two zip files download at the end.

## Run locally
```bash
pip install -r requirements.txt
python stock_recommendation.py --start 2020-01-01 --end 2026-10-05
```
Outputs go to `figures/` (report images) and `results/` (metrics, confusion matrix values, feature importance as CSV).

## Upload to GitHub
```bash
git init
git add .
git commit -m "Stock market recommendations using machine learning"
git branch -M main
git remote add origin https://github.com/<your-username>/stock-market-recommendation.git
git push -u origin main
```
Or use the GitHub website: **New repository -> uploading an existing file** and drag in all files.

## Files
- `stock_market_recommendation.ipynb` - self-contained Colab notebook
- `stock_recommendation.py` - same pipeline as a script
- `requirements.txt`, `.gitignore`, `README.md`

## Notes
- Results depend on the date range and the data downloaded, so numbers can differ slightly from the report tables.
- Academic project; not financial advice.
