"""
Stock Market Recommendations Using Machine Learning
BAI702 Mini Project - AMC Engineering College (CSE-AIML)

Pipeline:
  1. Download daily prices of six stocks (Yahoo Finance)
  2. Engineer technical indicators (SMA, RSI, MACD, Bollinger, volatility ...)
  3. Label: Buy (1) if the price 20 sessions later is higher, else Not Buy (0)
  4. Chronological 80/20 split, scale for Logistic Regression only
  5. Train 5 models, evaluate (Accuracy, Precision, Recall, F1, ROC-AUC)
  6. Save all report figures to ./figures and the metrics to ./results

Usage:
    pip install -r requirements.txt
    python stock_recommendation.py --start 2020-01-01 --end 2026-10-05
"""
import argparse
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
sns.set_theme(style="white")

TICKERS = ["NVDA", "MSFT", "AAPL", "AMZN", "GOOGL", "TSLA"]
HORIZON = 20        # sessions ahead
THRESH = 0.0        # Buy if forward return > 0
SEED = 42
FEATURES = ["Daily_Return", "SMA_Spread", "RSI_14", "MACD", "Bollinger_Pos",
            "Volatility", "Momentum_5d", "Trend_20d", "Volume_Trend"]


def load_prices(start, end, tickers=TICKERS):
    import yfinance as yf
    out = {}
    for t in tickers:
        d = yf.download(t, start=start, end=end, auto_adjust=True, progress=False)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        out[t] = d
    return out


def add_features(d, horizon=HORIZON, thresh=THRESH):
    d = d.copy()
    d["Daily_Return"] = d["Close"].pct_change()
    d["SMA_10"] = d["Close"].rolling(10).mean()
    d["SMA_50"] = d["Close"].rolling(50).mean()
    d["SMA_Spread"] = d["SMA_10"] / d["SMA_50"] - 1
    delta = d["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    d["RSI_14"] = 100 - 100 / (1 + gain / loss)
    d["MACD"] = (d["Close"].ewm(span=12, adjust=False).mean()
                 - d["Close"].ewm(span=26, adjust=False).mean())
    mid, sd = d["Close"].rolling(20).mean(), d["Close"].rolling(20).std()
    d["Bollinger_Pos"] = (d["Close"] - (mid - 2 * sd)) / (4 * sd)
    d["Volatility"] = d["Daily_Return"].rolling(10).std()
    d["Momentum_5d"] = d["Close"].pct_change(5)
    d["Trend_20d"] = d["Close"].pct_change(20)
    d["Volume_Trend"] = d["Volume"].pct_change(10)
    d["Forward_Return"] = d["Close"].shift(-horizon) / d["Close"] - 1
    d["Recommendation"] = (d["Forward_Return"] > thresh).astype(int)
    return d.replace([np.inf, -np.inf], np.nan).dropna()


def build_dataset(prices):
    data = pd.concat([add_features(d).assign(Ticker=t) for t, d in prices.items()])
    return data.sort_index()


def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=20,
                                                random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=12,
                                                random_state=SEED),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=150,
                                                        learning_rate=0.08,
                                                        random_state=SEED),
        "XGBoost": XGBClassifier(n_estimators=300, learning_rate=0.05,
                                 eval_metric="logloss", random_state=SEED),
    }


def split_and_scale(data):
    split = int(len(data) * 0.8)                    # chronological, no shuffle
    X_train, X_test = data[FEATURES].iloc[:split], data[FEATURES].iloc[split:]
    y_train, y_test = data["Recommendation"].iloc[:split], data["Recommendation"].iloc[split:]
    scaler = StandardScaler().fit(X_train)          # fit on train only
    return X_train, X_test, y_train, y_test, scaler.transform(X_train), scaler.transform(X_test)


def train_and_evaluate(models, X_train, X_test, y_train, y_test, X_train_s, X_test_s):
    rows, cms, rocs = [], {}, {}
    for name, m in models.items():
        Xtr, Xte = (X_train_s, X_test_s) if name == "Logistic Regression" else (X_train, X_test)
        m.fit(Xtr, y_train)
        pred, prob = m.predict(Xte), m.predict_proba(Xte)[:, 1]
        rows.append([name, accuracy_score(y_test, pred), precision_score(y_test, pred),
                     recall_score(y_test, pred), f1_score(y_test, pred),
                     roc_auc_score(y_test, prob)])
        cms[name] = confusion_matrix(y_test, pred)
        rocs[name] = roc_curve(y_test, prob)
    res = pd.DataFrame(rows, columns=["Model", "Accuracy", "Precision", "Recall",
                                      "F1-Score", "ROC-AUC"]).set_index("Model")
    return res, cms, rocs


def _save(fig_path):
    plt.tight_layout()
    plt.savefig(fig_path, dpi=170, bbox_inches="tight")
    plt.close()


def make_figures(data, res, cms, rocs, xgb_model, out="figures"):
    os.makedirs(out, exist_ok=True)
    # 2.1 target distribution
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(x="Recommendation", data=data, color="#3274a1")
    for p in ax.patches:
        ax.annotate(int(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["0 = Not Buy", "1 = Buy"])
    ax.set_ylabel("Number of Records"); ax.set_title("Recommendation (Target) Distribution")
    _save(f"{out}/fig_2_1_target_distribution.png")
    # 2.2 histograms
    data[FEATURES].hist(bins=30, figsize=(11, 8))
    plt.suptitle("Distribution of Technical Indicators")
    _save(f"{out}/fig_2_2_histograms.png")
    # 2.3 correlation heatmap
    plt.figure(figsize=(9, 7))
    sns.heatmap(data[FEATURES + ["Recommendation"]].corr(), annot=True, fmt=".2f",
                cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap of Features")
    _save(f"{out}/fig_2_3_correlation_heatmap.png")
    # 2.4 boxplots
    cols = ["RSI_14", "MACD", "Volatility", "Volume_Trend", "Daily_Return"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for a, c in zip(axes.ravel(), cols):
        sns.boxplot(x="Recommendation", y=c, data=data, ax=a)
        a.set_title(f"{c} vs Recommendation")
    axes.ravel()[-1].axis("off")
    _save(f"{out}/fig_2_4_boxplots.png")
    # accuracy bar
    d = res.sort_values("Accuracy", ascending=False)
    plt.figure(figsize=(8, 4.6))
    ax = sns.barplot(x=d.index, y=d["Accuracy"], color="#3274a1")
    for p, v in zip(ax.patches, d["Accuracy"]):
        ax.annotate(f"{v*100:.1f}%", (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 1); ax.set_xlabel("Machine Learning Model")
    ax.set_title("Accuracy Comparison of Machine Learning Models")
    plt.xticks(rotation=20)
    _save(f"{out}/fig_accuracy_comparison.png")
    # grouped metrics
    m = res.reset_index().melt(id_vars="Model", var_name="Metric", value_name="Score")
    plt.figure(figsize=(9, 5))
    ax = sns.barplot(data=m, x="Model", y="Score", hue="Metric")
    ax.set_ylim(0, 1); ax.set_title("Comparison of Evaluation Metrics Across ML Models")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=5, frameon=False)
    plt.xticks(rotation=12)
    _save(f"{out}/fig_metrics_comparison.png")
    # confusion matrices
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for a, (n, cm) in zip(axes.ravel(), cms.items()):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=a,
                    xticklabels=["Not Buy", "Buy"], yticklabels=["Not Buy", "Buy"])
        a.set_title(f"Confusion Matrix - {n}"); a.set_xlabel("Predicted"); a.set_ylabel("Actual")
    axes.ravel()[-1].axis("off")
    _save(f"{out}/fig_confusion_matrices.png")
    # ROC curves
    plt.figure(figsize=(7, 6))
    for n, (fpr, tpr, _) in rocs.items():
        plt.plot(fpr, tpr, label=f"{n} (AUC={res.loc[n, 'ROC-AUC']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - All Models"); plt.legend(loc="lower right")
    _save(f"{out}/fig_roc_curves.png")
    # XGBoost feature importance
    imp = pd.Series(xgb_model.feature_importances_, index=FEATURES).sort_values()
    plt.figure(figsize=(7, 4.5)); imp.plot(kind="barh", color="#3274a1")
    plt.xlabel("Importance"); plt.title("XGBoost Feature Importance")
    _save(f"{out}/fig_xgb_feature_importance.png")
    return imp


def run(start, end, figures_dir="figures", results_dir="results", prices=None):
    prices = prices or load_prices(start, end)
    data = build_dataset(prices)
    X_train, X_test, y_train, y_test, X_train_s, X_test_s = split_and_scale(data)
    models = get_models()
    res, cms, rocs = train_and_evaluate(models, X_train, X_test, y_train, y_test,
                                        X_train_s, X_test_s)
    imp = make_figures(data, res, cms, rocs, models["XGBoost"], figures_dir)

    os.makedirs(results_dir, exist_ok=True)
    res.round(4).to_csv(f"{results_dir}/model_metrics.csv")
    pd.DataFrame({n: cm.ravel() for n, cm in cms.items()}, index=["TN", "FP", "FN", "TP"]) \
        .T.to_csv(f"{results_dir}/confusion_matrix_values.csv")
    imp.sort_values(ascending=False).to_csv(f"{results_dir}/xgb_feature_importance.csv")

    print(f"Rows: {len(data)} | train: {len(X_train)} | test: {len(X_test)}")
    print("Class balance (%):\n", (data["Recommendation"].value_counts(normalize=True) * 100).round(2))
    print("\nMetrics (%):\n", (res * 100).round(2))
    print("\nConfusion matrix values (TN, FP, FN, TP):")
    for n, cm in cms.items():
        print(f"  {n}: {cm.ravel().tolist()}")
    print("\nTop XGBoost features:", imp.sort_values(ascending=False).head(4).round(3).to_dict())
    print(f"\nFigures saved in ./{figures_dir}  |  tables saved in ./{results_dir}")
    return data, res, cms, rocs, models


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default="2026-10-05")
    a = ap.parse_args()
    run(a.start, a.end)
