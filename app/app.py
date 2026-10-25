"""
Vessel detection dashboard.

Run from the project folder (Vessel_Detection_DAS) with:
    streamlit run app/app.py
It opens in your browser (Chrome) at http://localhost:8501
"""
import sys
from pathlib import Path

import joblib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "src"))
from models import ANNModel  # noqa: E402

RESULTS = ROOT / "results"
DATA_PATH = r"C:/Users/urvi_/Downloads/dataset.h5"   # only used for the vessel-size tab
DEMO_DAY = "2023-06-24"                               # the day the saved model never saw
THRESHOLD_M = 1000

INK, BLUE, AMBER, GREY = "#12263A", "#1B6CA8", "#E8A33D", "#8A99A8"

st.set_page_config(page_title="Vessel detection on a submarine cable", layout="wide")


# ---------------------------------------------------------------- loading helpers
def find_file(name):
    """Look for a results file in results/ and results/tables/."""
    for folder in (RESULTS, RESULTS / "tables"):
        if (folder / name).exists():
            return folder / name
    return None


@st.cache_data
def load_data():
    d = np.load(RESULTS / "features_all_days_v2.npz")
    times = pd.to_datetime(d["dt"])
    day_str = np.array([str(x) for x in times.date])
    return d["X"], d["y"], times, day_str


@st.cache_resource
def load_model(n_features):
    scaler = joblib.load(RESULTS / "models" / "ann_v2_demo_scaler_test0624.pkl")
    model = ANNModel(input_size=n_features, hidden1=256, hidden2=128, dropout=0.3)
    model.load_state_dict(torch.load(RESULTS / "models" / "ann_v2_demo_test0624.pth", map_location="cpu"))
    model.eval()
    return model, scaler


@st.cache_data
def predict_day(day):
    X, dist, times, day_str = load_data()
    model, scaler = load_model(X.shape[1])
    m = day_str == day
    with torch.no_grad():
        logits = model(torch.tensor(scaler.transform(X[m]), dtype=torch.float32))
        prob = torch.sigmoid(logits).numpy()
    return times[m], dist[m], prob


@st.cache_data
def vessel_lengths():
    import h5py
    with h5py.File(DATA_PATH, "r") as f:
        return f["ship_info"]["length"][:].astype(float)


def read_table(name):
    path = find_file(name)
    return pd.read_csv(path) if path else None


# ---------------------------------------------------------------- page header
st.title("Detecting nearby ships from a submarine cable")
st.write(
    "The cable itself works as a microphone. A neural network reads 10 seconds of its sound at a time "
    f"and says whether a vessel is closer than {THRESHOLD_M} metres."
)

tab_demo, tab_results, tab_errors = st.tabs(["Run the model", "Results across 10 days", "Where it struggles"])

# ---------------------------------------------------------------- tab 1: live model
with tab_demo:
    try:
        X, dist_all, times_all, day_str = load_data()
        days = sorted(set(day_str))
    except FileNotFoundError:
        st.error("results/features_all_days_v2.npz is missing. Build the feature file first (notebook 06).")
        st.stop()

    c1, c2, c3 = st.columns([1.2, 1.5, 1.2])
    day = c1.selectbox("Recording day", days, index=days.index(DEMO_DAY))
    minutes_window = c3.selectbox("Window shown", [60, 120, 240], index=1,
                                  format_func=lambda m: f"{m} minutes")

    if day != DEMO_DAY:
        st.warning(f"The saved model was trained on {day}, so scores for this day look better than they would "
                   f"on new data. {DEMO_DAY} is the only day it has never seen.")

    times, dist, prob = predict_day(day)
    truth = (dist < THRESHOLD_M).astype(int)
    pred = (prob >= 0.5).astype(int)

    n_win = minutes_window * 6
    max_start = max(len(times) - n_win, 0)
    start_min = c2.slider("Start of the window (minutes into the day)", 0, max_start // 6,
                          min(300, max_start // 6), step=10)
    sl = slice(start_min * 6, start_min * 6 + n_win)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{accuracy_score(truth, pred):.3f}")
    m2.metric("Near-vessel F1", f"{f1_score(truth, pred):.3f}")
    m3.metric("AUC", f"{roc_auc_score(truth, prob):.3f}")
    m4.metric("Recordings on this day", f"{len(times):,}")

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 5.8), sharex=True)
    a1.plot(times[sl], dist[sl], color=INK, lw=1.1)
    a1.axhline(THRESHOLD_M, ls="--", color=GREY, lw=1)
    a1.text(times[sl][0], THRESHOLD_M + 60, f"{THRESHOLD_M} m threshold", color=GREY, fontsize=9)
    a1.set_ylim(0, 3000)
    a1.set_ylabel("Distance to nearest vessel (m)")
    a1.set_title("What the ship tracking (AIS) says", loc="left", fontsize=11)

    a2.fill_between(times[sl], 0, 1, where=truth[sl] == 1, color=AMBER, alpha=0.22,
                    label=f"A vessel really is within {THRESHOLD_M} m")
    a2.plot(times[sl], prob[sl], color=BLUE, lw=1.3, label="Model: probability that a vessel is near")
    a2.axhline(0.5, ls="--", color=GREY, lw=1)
    a2.set_ylim(0, 1)
    a2.set_ylabel("Probability")
    a2.set_title("What the model says from the cable sound alone", loc="left", fontsize=11)
    a2.legend(loc="upper right", fontsize=9, frameon=False)
    a2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    for a in (a1, a2):
        a.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)

    left, right = st.columns([1, 1])
    cm = confusion_matrix(truth, pred)
    right.write("Counts for the whole day")
    right.dataframe(
        pd.DataFrame(cm, index=["Really far", "Really near"], columns=["Model said far", "Model said near"]),
    )
    left.write(
        "Where the blue line rises above 0.5 while the amber band is present, the model has found a nearby "
        "vessel. A blue peak outside the band is a false alarm, and an amber band with no peak is a miss."
    )

# ---------------------------------------------------------------- tab 2: results
with tab_results:
    st.subheader("Three models, tested on each day they had never seen")
    sources = {
        "ANN, 100 features": "ann_10day_results.csv",
        "ANN, 300 features": "ann_v2_10day_results.csv",
        "LSTM, 5 recordings": "lstm_10day_results.csv",
    }
    frames, rows = {}, []
    for label, fname in sources.items():
        df = read_table(fname)
        if df is None:
            st.info(f"{fname} not found, so '{label}' is left out.")
            continue
        frames[label] = df
        rows.append({
            "Model": label,
            "Accuracy": df["accuracy"].mean(),
            "Near F1": df["near_f1"].mean(),
            "AUC": df["auc"].mean(),
            "Near F1 spread (std)": df["near_f1"].std(),
        })
    if rows:
        st.dataframe(pd.DataFrame(rows).round(3), hide_index=True)
        st.caption("Each number is the mean over 10 folds. In each fold, one day is held out for testing.")

        fig2, ax = plt.subplots(figsize=(10, 3.6))
        colours = [GREY, BLUE, AMBER]
        for (label, df), colour in zip(frames.items(), colours):
            ax.plot(df["test_day"].astype(str).str[-5:], df["near_f1"], marker="o", lw=1.4,
                    color=colour, label=label)
        ax.set_ylabel("Near-vessel F1")
        ax.set_xlabel("Test day (June 2023)")
        ax.set_ylim(0.65, 0.9)
        ax.legend(frameon=False, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig2)
        st.write("The models are close on every day, and 22 June is the hardest day for all of them.")

# ---------------------------------------------------------------- tab 3: errors by size
with tab_errors:
    st.subheader("Small vessels are detected far less often")
    pred_path = find_file("ann_v2_10day_predictions.csv")
    if pred_path is None:
        st.info("ann_v2_10day_predictions.csv not found, so this tab cannot be drawn.")
    else:
        try:
            lengths = vessel_lengths()
        except Exception:
            st.info("The vessel lengths come from the original dataset file. Check DATA_PATH at the top of app.py.")
            st.stop()
        _, _, _, day_str = load_data()
        order = np.concatenate([np.where(day_str == d)[0] for d in sorted(set(day_str))])
        pp = pd.read_csv(pred_path)
        pp["length"] = lengths[order]
        near = pp[pp["true"] == 1].copy()
        near["size"] = pd.cut(near["length"], [0, 50, 100, 150, 200, 1000],
                              labels=["under 50 m", "50-100 m", "100-150 m", "150-200 m", "over 200 m"])
        s = near.groupby("size", observed=True).agg(n=("true", "size"), recall=("pred", "mean"))

        fig3, ax = plt.subplots(figsize=(8, 3.8))
        ax.bar(s.index.astype(str), s["recall"], color=BLUE)
        for i, (r, n) in enumerate(zip(s["recall"], s["n"])):
            ax.text(i, r + 0.02, f"{r:.2f}\n(n={n:,})", ha="center", fontsize=9)
        ax.set_ylim(0, 1.2)
        ax.set_ylabel("Share of nearby vessels detected")
        ax.set_xlabel("Vessel length")
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig3)
        st.write(
            "Detection rises steadily with vessel length. Ships over 150 m are found about 93% of the time, "
            "while ships under 100 m are found only about 55% of the time. Smaller vessels are quieter, but "
            "length is also mixed up with vessel type, so this shows a pattern and does not prove the cause."
        )