# streamlit_knapsack_app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import random

st.set_page_config(layout="wide", page_title="Knapsack Greedy + ML Demo")

st.title("🎒 Knapsack Greedy Strategies + ML Profit Predictor")

# Load dataset
DATA_PATH = "items_dataset.csv"
try:
    items_df = pd.read_csv(DATA_PATH)
except Exception as e:
    st.error(f"Could not load '{DATA_PATH}': {e}")
    st.stop()

st.sidebar.header("Settings")
num_items = st.sidebar.slider("Number of items to use", min_value=1, max_value=len(items_df), value=len(items_df))
strategy = st.sidebar.selectbox("Greedy strategy", ("Profit/Weight (ratio)", "Profit (highest first)", "Weight (lightest first)"))
capacity = st.sidebar.slider("Knapsack capacity", min_value=1, max_value=200, value=60)

st.markdown("## Items (you can edit weights & profits below)")
# copy relevant rows
items = items_df.head(num_items).copy().reset_index(drop=True)

# Create sliders for each item
edited_weights = []
edited_profits = []

cols = st.columns((1,1,1,1))
for i, row in items.iterrows():
    with st.container():
        c1, c2, c3, c4 = st.columns([2,2,2,3])
        with c1:
            st.markdown(f"**{row['Item']}**")
        with c2:
            w = st.slider(f"Weight_{i}", min_value=1, max_value=100, value=int(row["Weight"]))
            edited_weights.append(w)
        with c3:
            p = st.slider(f"Profit_{i}", min_value=1, max_value=500, value=int(row["Profit"]))
            edited_profits.append(p)
        with c4:
            ratio = round(p / w, 3)
            st.markdown(f"Profit/Weight: **{ratio}**")

# Build current items table
current = pd.DataFrame({
    "Item": items["Item"],
    "Weight": edited_weights,
    "Profit": edited_profits
})
current["ProfitPerWeight"] = (current["Profit"] / current["Weight"]).round(4)

st.dataframe(current)

# ----------------------------
# Greedy selection functions
# ----------------------------
def greedy_select(df, cap, strategy_name):
    df2 = df.copy().reset_index(drop=True)
    if strategy_name == "Profit/Weight (ratio)":
        df2["key"] = df2["Profit"] / df2["Weight"]
        df2 = df2.sort_values("key", ascending=False)
    elif strategy_name == "Profit (highest first)":
        df2 = df2.sort_values("Profit", ascending=False)
    elif strategy_name == "Weight (lightest first)":
        df2 = df2.sort_values("Weight", ascending=True)
    else:
        raise ValueError("Unknown strategy")

    selected = []
    total_w = 0
    total_p = 0
    for idx, r in df2.iterrows():
        if total_w + r["Weight"] <= cap:
            selected.append(r["Item"])
            total_w += r["Weight"]
            total_p += r["Profit"]
    return selected, total_w, total_p, df2

# Run greedy
selected_items, total_w, total_p, ordering = greedy_select(current, capacity, strategy)

st.markdown("## Greedy Selection Result")
st.write(f"**Strategy:** {strategy}")
st.write(f"**Capacity:** {capacity}")
st.write(f"**Selected items:** {selected_items}")
st.write(f"**Total weight used:** {total_w}")
st.write(f"**Net profit (greedy):** {total_p}")

# Visuals: bar charts of profits and weights and the ordering
st.markdown("### Visuals")
fig, ax = plt.subplots(1, 2, figsize=(12,4))
ordering.plot.bar(x="Item", y="Profit", ax=ax[0])
ax[0].set_title("Profit (items ordered by strategy)")
ordering.plot.bar(x="Item", y="Weight", ax=ax[1])
ax[1].set_title("Weight (items ordered by strategy)")
plt.tight_layout()
st.pyplot(fig)

# ----------------------------
# ML Model: Train a regressor to predict greedy net profit
# We'll generate synthetic training data by randomizing item features
# ----------------------------
@st.cache_data
def generate_training_data(n_samples=3000, n_items= num_items):
    X = []
    y = []
    for _ in range(n_samples):
        # random item features
        ws = np.random.randint(1, 101, size=n_items)
        ps = np.random.randint(1, 501, size=n_items)
        cap = int(np.random.randint(1, max(1, int(ws.sum()*0.8)+1)))
        strat = random.choice(["ratio", "profit", "weight"])
        # build df
        df_inst = pd.DataFrame({"Item": [f"I{i}" for i in range(n_items)], "Weight": ws, "Profit": ps})
        # compute greedy result
        if strat == "ratio":
            key = ps / ws
            order = np.argsort(-key)
        elif strat == "profit":
            order = np.argsort(-ps)
        else:
            order = np.argsort(ws)
        total_w = 0
        total_p = 0
        for idx in order:
            if total_w + ws[idx] <= cap:
                total_w += ws[idx]
                total_p += ps[idx]
        # features: flattened ws and ps, capacity, strategy encoded
        feat = np.concatenate([ws, ps, [cap, 0 if strat=="ratio" else (1 if strat=="profit" else 2)]])
        X.append(feat)
        y.append(total_p)
    X = np.array(X)
    y = np.array(y)
    return X, y

# generate & train
with st.spinner("Generating synthetic data and training ML predictor..."):
    X, y = generate_training_data(n_samples=3000, n_items=num_items)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds_test = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds_test, squared=False)

st.write(f"### ML Predictor (RandomForestRegressor)")
st.write(f"Trained on synthetic data for {num_items} items. Test RMSE: **{rmse:.2f}**")

# Prepare current instance feature vector for prediction
def make_feature_vector(df_current, cap, strat_name):
    n = len(df_current)
    ws = df_current["Weight"].to_numpy()
    ps = df_current["Profit"].to_numpy()
    s = 0 if strat_name=="Profit/Weight (ratio)" else (1 if strat_name=="Profit (highest first)" else 2)
    feat = np.concatenate([ws, ps, [cap, s]])
    return feat.reshape(1, -1)

feat_vec = make_feature_vector(current, capacity, strategy)
predicted_profit = model.predict(feat_vec)[0]

st.write(f"**ML predicted net profit:** {predicted_profit:.2f}")
st.write(f"**Greedy actual net profit (computed):** {total_p}")

# Compare predicted vs actual visually
st.markdown("### Predicted vs Actual")
fig2, ax2 = plt.subplots(1,1, figsize=(6,4))
ax2.bar(["Greedy Actual", "ML Predicted"], [total_p, predicted_profit])
ax2.set_ylabel("Net Profit")
st.pyplot(fig2)

# Download current configuration and results
output = current.copy()
output["Selected"] = output["Item"].apply(lambda x: 1 if x in selected_items else 0)
output["Capacity"] = capacity
output["Strategy"] = strategy
output["GreedyNetProfit"] = total_p
output["MLPredictedProfit"] = round(predicted_profit, 2)

csv = output.to_csv(index=False)
st.download_button("Download current config + results (CSV)", csv, file_name="knapsack_current_results.csv")
