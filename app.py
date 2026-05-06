#Run the `app.py` is fully self-contained. It does everything the notebook does,
#  just inside the Streamlit UI

#- Loads `data/train.csv` and `data/test.csv` itself
#- Cleans the data, builds the pipeline, trains the LogisticRegression 
# model in-browser when the student clicks **"Train the model"**
#- Predicts on `test.csv` and shows the `account #N → bot/human` lines
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

st.set_page_config(page_title="Bot Account Classifier", layout="wide")
st.title("Bot Account Classification")
st.caption("Live demo — the model trains in the browser and predicts on test.csv")

DATA_DIR = Path(__file__).parent / "data"


def show(title, fn):
    """Run fn() and also display its source code so the student can read it."""
    st.subheader(title)
    code_col, out_col = st.columns([1, 1])
    with code_col:
        st.code(inspect.getsource(fn), language="python")
    result = fn()
    return result, out_col


def step_load():
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    return train, test


def step_clean(train, test):
    train = train.replace("Unknown", np.nan)
    test = test.replace("Unknown", np.nan)

    for col in ["friends_count", "posts_count"]:
        train[col] = pd.to_numeric(train[col], errors="coerce")
        test[col] = pd.to_numeric(test[col], errors="coerce")

    lo = train["friends_count"].quantile(0.01)
    hi = train["friends_count"].quantile(0.99)
    train["friends_count"] = train["friends_count"].clip(lo, hi)
    test["friends_count"] = test["friends_count"].clip(lo, hi)
    return train, test


def step_build_model(X):
    num_cols = X.select_dtypes(include="number").columns
    cat_cols = X.select_dtypes(exclude="number").columns

    preprocess = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), cat_cols),
    ])

    model = Pipeline([
        ("prep", preprocess),
        ("clf", LogisticRegression(max_iter=5000)),
    ])
    return model


def step_train(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    score = f1_score(y_val, preds)
    return model, score


# ---------- 1. Load ----------
st.header("1. Load the datasets")
st.code(inspect.getsource(step_load), language="python")
train, test = step_load()
c1, c2 = st.columns(2)
c1.write(f"**train.csv** — {train.shape[0]} rows × {train.shape[1]} cols")
c1.dataframe(train.head())
c2.write(f"**test.csv** — {test.shape[0]} rows × {test.shape[1]} cols")
c2.dataframe(test.head())

# ---------- 2. Clean ----------
st.header("2. Clean the data")
st.code(inspect.getsource(step_clean), language="python")
train, test = step_clean(train, test)
st.success("Replaced 'Unknown' with NaN, converted counts to numeric, clipped outliers.")

# ---------- 3. Build pipeline ----------
st.header("3. Build the preprocessing + model pipeline")
X = train.drop(columns=["outcome", "id"])
y = train["outcome"]
st.code(inspect.getsource(step_build_model), language="python")
model = step_build_model(X)

# ---------- 4. Train ----------
st.header("4. Train and evaluate")
st.code(inspect.getsource(step_train), language="python")
if st.button("Train the model", type="primary"):
    with st.spinner("Training..."):
        model, score = step_train(model, X, y)
    st.metric("F1 score on validation set", f"{score:.4f}")
    st.session_state["model"] = model

# ---------- 5. Predict ----------
st.header("5. Predict on test.csv")
st.code(
    'test_features = test.drop(columns=["id"])\n'
    'test_preds = model.predict(test_features)',
    language="python",
)

if "model" in st.session_state:
    model = st.session_state["model"]
    test_features = test.drop(columns=["id"])
    test_preds = model.predict(test_features)

    label = {0: "human", 1: "bot"}
    n = st.slider("How many accounts to display?", 5, min(50, len(test)), 5)

    st.subheader("Predictions")
    for i in range(n):
        acc_id = int(test["id"].iloc[i])
        pred = int(test_preds[i])
        emoji = "🤖" if pred == 1 else "🧑"
        st.markdown(f"{emoji}  **account #{acc_id}** → **{label[pred]}**")

    submissions = pd.DataFrame({"id": test["id"], "outcome": test_preds})
    st.download_button(
        "Download submissions.csv",
        submissions.to_csv(index=False).encode("utf-8"),
        file_name="submissions.csv",
        mime="text/csv",
    )
else:
    st.info("Train the model first (step 4) to see predictions here.")
