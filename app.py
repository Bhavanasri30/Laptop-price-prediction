import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import altair as alt

# --- Config / mapping (must match training) ---
BRAND_MAPPING = {"Acer": 0, "Apple": 1, "Asus": 2, "Dell": 3, "HP": 4, "Lenovo": 5}
FEATURE_COLS = ["Brand", "Processor_Speed", "RAM_Size", "Storage_Capacity", "Screen_Size", "Weight"]
TARGET_COL = "Price"

# --- UI: Header ---
st.set_page_config(page_title="Laptop Price ML App", layout="centered")
st.title("💻 Laptop Price — from Notebook to Streamlit")
st.write("Quickly explore data, train a model and predict laptop price.")

# --- Load dataset ---
@st.cache_data
def load_data(path="Laptop_price.csv"):
    return pd.read_csv(path)

df = load_data()
st.subheader("Dataset preview")
st.dataframe(df.head())
st.markdown(f"**Rows:** {len(df)} — **Columns:** {', '.join(df.columns)}")

# --- EDA ---
st.subheader("Exploratory Data Analysis")
col1, col2 = st.columns(2)
with col1:
    hist_col = st.selectbox("Histogram column", FEATURE_COLS + [TARGET_COL], index=1)
    st.write(alt.Chart(df).mark_bar().encode(
        alt.X(hist_col, bin=alt.Bin(maxbins=40)),
        y='count()'
    ).interactive().properties(height=250))
with col2:
    x_col = st.selectbox("Scatter X", FEATURE_COLS, index=1)
    y_col = st.selectbox("Scatter Y", [TARGET_COL], index=0)
    st.write(alt.Chart(df).mark_circle(size=60).encode(
        x=x_col, y=y_col, color='Brand'
    ).interactive().properties(height=250))

# --- Preprocess helpers ---
def encode_brand(series):
    return series.map(BRAND_MAPPING)

def make_X_y(df):
    X = df[FEATURE_COLS].copy()
    X["Brand"] = encode_brand(X["Brand"])
    y = df[TARGET_COL].values
    return X, y

# --- Model training controls ---
st.subheader("Train a model")
model_type = st.selectbox("Choose model", ["Linear Regression", "Random Forest", "SVR"])
test_size = st.slider("Test set fraction", 0.1, 0.5, 0.2)
random_state = st.number_input("Random state", value=42, step=1)

# hyperparams
if model_type == "Random Forest":
    n_estimators = st.slider("n_estimators", 10, 500, 100)
    max_depth = st.slider("max_depth (None=0)", 0, 50, 0)
elif model_type == "SVR":
    svr_C = st.number_input("C", value=1.0, format="%.3f")
    svr_gamma = st.selectbox("gamma", ["scale", "auto"])
else:
    # Linear regression has no main hyperparams
    pass

if st.button("Train model"):
    X, y = make_X_y(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if model_type == "Linear Regression":
        model = LinearRegression()
    elif model_type == "Random Forest":
        md = None if max_depth == 0 else max_depth
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=md, random_state=random_state)
    else:
        model = SVR(C=svr_C, gamma=svr_gamma)

    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    st.success("Training completed ✅")
    st.metric("R2", f"{r2:.4f}")
    st.metric("MAE", f"{mae:.2f}")
    st.metric("RMSE", f"{rmse:.2f}")

    # scatter actual vs predicted
    results = pd.DataFrame({"actual": y_test, "pred": y_pred})
    chart = alt.Chart(results).mark_circle().encode(x="actual", y="pred").properties(width=500, height=350)
    st.altair_chart(chart, use_container_width=True)

    # Save model & scaler
    if st.button("Save model & scaler to disk"):
        joblib.dump(model, "laptop_price_model.pkl")
        joblib.dump(scaler, "laptop_price_scaler.pkl")
        st.success("Model and scaler saved as .pkl files")

# --- Prediction UI (reuse same preprocessing) ---
st.subheader("Make a prediction")
colA, colB = st.columns(2)
with colA:
    p_brand = st.selectbox("Brand", list(BRAND_MAPPING.keys()))
    p_speed = st.slider("Processor Speed (GHz)", 1.0, 5.0, 3.0)
    p_ram = st.selectbox("RAM (GB)", [4, 8, 16, 32, 64])
with colB:
    p_storage = st.selectbox("Storage (GB)", [128, 256, 512, 1024, 2048])
    p_screen = st.slider("Screen Size (inches)", 11.0, 18.0, 15.6)
    p_weight = st.slider("Weight (kg)", 1.0, 5.0, 2.5)

# Load saved model for prediction
use_saved = st.checkbox("Use existing saved model `laptop_price_model.pkl` (if present)", value=False)
predict_button = st.button("Predict price")

if predict_button:
    row = pd.DataFrame([[p_brand, p_speed, p_ram, p_storage, p_screen, p_weight]], columns=FEATURE_COLS)
    row["Brand"] = encode_brand(row["Brand"])
    try:
        if use_saved:
            model = joblib.load("laptop_price_model.pkl")  # your existing file
            scaler = joblib.load("laptop_price_scaler.pkl")
        else:
            # fallback to model saved by this app (if present)
            model = joblib.load("laptop_price_model.pkl")
            scaler = joblib.load("laptop_price_scaler.pkl")

        x_scaled = scaler.transform(row)
        pred = model.predict(x_scaled)[0]
        st.success(f"💰 Predicted Price: ₹ {pred:.2f}")
    except FileNotFoundError:
        st.error("Saved model/scaler not found. Train and save a model first, or uncheck 'Use existing saved model' and run training above.")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
