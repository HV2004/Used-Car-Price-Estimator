import streamlit as st

# ✅ MUST be the first Streamlit command
st.set_page_config(page_title="Used Car Price Estimator", layout="centered")

import pandas as pd
import joblib

# Load model, scaler, and reference columns
model = joblib.load("models/model.pkl")
scaler = joblib.load("models/scaler.pkl")
reference_X = joblib.load("models/reference_columns.pkl")

# Load dataset to build dropdowns
df = pd.read_csv("used_cars_dataset_v2.csv")
brand_model_map = df.groupby("Brand")["model"].unique().apply(list).to_dict()
all_brands = sorted(df["Brand"].dropna().unique())
all_transmissions = sorted(df["Transmission"].dropna().unique())
all_owners = sorted(df["Owner"].dropna().unique())
all_fuel_types = sorted(df["FuelType"].dropna().unique())

# Apply custom CSS AFTER set_page_config
st.markdown("""
    <style>
        .main-title {
            text-align: center;
            font-size: 42px;
            color: #0d6efd;
            font-weight: bold;
            margin-bottom: 0.5em;
        }
        .section {
            background-color: #ffffff;
            border: 1px solid #dee2e6;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 0 10px rgba(0,0,0,0.05);
        }
        .result-box {
            background-color: #e6f4ea;
            padding: 20px;
            border-radius: 10px;
            border: 2px solid #198754;
            font-size: 24px;
            font-weight: bold;
            text-align: center;
            color: #198754;
        }
    </style>
""", unsafe_allow_html=True)

# 🚗 Title
st.markdown("<div class='main-title'>🚗 Used Car Price Estimator</div>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# 🧾 Car Input Section
st.markdown("### 📝 Enter Car Details")
with st.container():
    st.markdown("<div class='section'>", unsafe_allow_html=True)

    brand = st.selectbox("Brand", all_brands)
    model_list = sorted(brand_model_map.get(brand, []))
    model_name = st.selectbox("Model", model_list)

    col1, col2 = st.columns(2)
    with col1:
        year = st.number_input("Year of Manufacture", min_value=1995, max_value=2025, value=2019)
    with col2:
        age = st.number_input("Car Age (in years)", min_value=0, max_value=30, value=2025 - year)

    km_driven = st.slider("Kilometers Driven", 0, 300000, 50000, step=1000)

    col3, col4 = st.columns(2)
    with col3:
        transmission = st.selectbox("Transmission", all_transmissions)
    with col4:
        owner = st.selectbox("Owner Type", all_owners)

    fuel_type = st.selectbox("Fuel Type", all_fuel_types)

    st.markdown("</div>", unsafe_allow_html=True)

# 📈 Prediction Section
if st.button("📈 Predict Price"):
    input_data = {
        'Brand': brand,
        'model': model_name,
        'Year': year,
        'Age': age,
        'kmDriven': km_driven,
        'Transmission': transmission,
        'Owner': owner,
        'FuelType': fuel_type
    }

    input_df = pd.DataFrame([input_data])
    input_encoded = pd.get_dummies(input_df)

    # Add missing columns
    for col in reference_X:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded[reference_X]

    # Scale and predict
    input_scaled = scaler.transform(input_encoded)
    predicted_price = model.predict(input_scaled)[0]

    # 💰 Result
    st.markdown("<div class='result-box'>", unsafe_allow_html=True)
    st.markdown(f"💰 Estimated Ask Price: ₹ {int(predicted_price):,}")
    st.markdown("</div>", unsafe_allow_html=True)
