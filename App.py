import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# 2. CUSTOM CSS
# --------------------------------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(
        135deg, #edf2ff 0%, #dce6ff 50%, #eeeaff 100%
    );
    color: #172554;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #172554, #3730a3);
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span {
    color: white !important;
}
[data-testid="stSidebar"] input {
    color: #111827 !important;
    background: white !important;
}
h1, h2, h3 {
    color: #172554 !important;
}
.hero {
    padding: 28px;
    border-radius: 18px;
    background: linear-gradient(120deg, #f4b86a, #f59e55);
    margin-bottom: 20px;
}
.hero h1, .hero p {
    color: white !important;
}
.prediction-card {
    background: linear-gradient(120deg, #3730a3, #2563eb);
    padding: 28px;
    border-radius: 18px;
    color: white;
    text-align: center;
}
.prediction-card h3, .prediction-card p {
    color: white !important;
}
.prediction-price {
    font-size: 2.4rem;
    font-weight: 800;
    margin: 12px 0;
}
[data-testid="stMetric"] {
    background: white;
    padding: 18px;
    border-radius: 14px;
    border: 1px solid #c7d2fe;
}
[data-testid="stMetricLabel"] p {
    color: #475569 !important;
}
[data-testid="stMetricValue"] {
    color: #172554 !important;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# 3. GENERATE SYNTHETIC DATASET
# --------------------------------------------------
@st.cache_data
def create_dataset():
    rng = np.random.default_rng(42)
    n = 600

    area = rng.integers(600, 3501, size=n)

    bedrooms = np.clip(
        np.rint(area / 650 + rng.normal(0, 0.55, n)),
        1, 6
    ).astype(int)

    bathrooms = np.clip(
        np.rint(bedrooms * 0.65 + rng.normal(0.35, 0.5, n)),
        1, 5
    ).astype(int)

    age = rng.integers(0, 41, size=n)

    parking = np.clip(
        np.rint((area - 400) / 1000 + rng.normal(0, 0.65, n)),
        0, 4
    ).astype(int)

    location_factor = rng.normal(0, 5, size=n)
    noise = rng.normal(0, 3, size=n)

    # Price in lakhs
    price = (
        12
        + 0.045 * area
        + 3.2 * bedrooms
        + 2.0 * bathrooms
        - 0.55 * age
        + 2.5 * parking
        + location_factor
        + noise
    )

    price = np.maximum(price, 15)

    return pd.DataFrame({
        "Area_sqft": area,
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "Age_years": age,
        "Parking": parking,
        "Price_lakhs": np.round(price, 2)
    })


df = create_dataset()

features = [
    "Area_sqft",
    "Bedrooms",
    "Bathrooms",
    "Age_years",
    "Parking"
]

target = "Price_lakhs"

# --------------------------------------------------
# 4. PREPARE FEATURES
# --------------------------------------------------
def transform_features(data):
    transformed = data[features].copy()

    # Invert age so a positive coefficient means
    # older houses have a lower predicted price.
    transformed["Age_years"] = -transformed["Age_years"]

    return transformed


X = transform_features(df)
y = df[target]

# --------------------------------------------------
# 5. TRAIN AND EVALUATE MODEL
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Constrain coefficients to be non-negative in transformed
# feature space. This makes age's original coefficient
# non-positive and the other feature coefficients non-negative.
model = LinearRegression(positive=True)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Convert age coefficient back to its original meaning.
original_coefs = model.coef_.copy()
original_coefs[features.index("Age_years")] *= -1

intercept = float(model.intercept_)

# --------------------------------------------------
# 6. HEADER
# --------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>🏠 House Price Prediction</h1>
    <p>
        Multiple Linear Regression | Explore data,
        evaluate model performance, and predict house prices.
    </p>
</div>
""", unsafe_allow_html=True)

st.caption("Machine Learning Dashboard | Target: Price in ₹ lakhs")

st.warning(
    "This application uses 600 synthetic demonstration records, "
    "not actual property-sale data. Predictions are for learning only."
)

# --------------------------------------------------
# 7. SIDEBAR INPUTS
# --------------------------------------------------
st.sidebar.title("🏡 Predict House Price")
st.sidebar.write("Enter the house details.")

area = st.sidebar.number_input(
    "Area (square feet)",
    min_value=300,
    max_value=10000,
    value=2100,
    step=100
)

bedrooms = st.sidebar.number_input(
    "Number of Bedrooms",
    min_value=1,
    max_value=10,
    value=3,
    step=1
)

bathrooms = st.sidebar.number_input(
    "Number of Bathrooms",
    min_value=1,
    max_value=10,
    value=2,
    step=1
)

age = st.sidebar.number_input(
    "House Age (years)",
    min_value=0,
    max_value=100,
    value=5,
    step=1
)

parking = st.sidebar.number_input(
    "Parking Spaces",
    min_value=0,
    max_value=10,
    value=2,
    step=1
)

input_original = pd.DataFrame([{
    "Area_sqft": area,
    "Bedrooms": bedrooms,
    "Bathrooms": bathrooms,
    "Age_years": age,
    "Parking": parking
}])

input_transformed = transform_features(input_original)

predicted_price = float(
    model.predict(input_transformed)[0]
)

st.sidebar.divider()
st.sidebar.subheader("Selected Features")
st.sidebar.dataframe(
    input_original,
    hide_index=True,
    use_container_width=True
)

# --------------------------------------------------
# 8. PREDICTION RESULT
# --------------------------------------------------
st.markdown("## 🎯 House Price Prediction")

col1, col2 = st.columns([1.5, 1], gap="large")

with col1:
    st.markdown(f"""
    <div class="prediction-card">
        <h3>Estimated House Price</h3>
        <div class="prediction-price">
            ₹ {predicted_price:.2f} Lakhs
        </div>
        <p>Based on the selected property features</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.metric("Input Features", len(features))
    st.metric("Training Records", len(X_train))
    st.metric("Testing Records", len(X_test))

# --------------------------------------------------
# 9. MODEL PERFORMANCE
# --------------------------------------------------
st.markdown("## 📊 Model Performance")

m1, m2, m3, m4 = st.columns(4)

m1.metric("MAE", f"{mae:.2f} Lakhs")
m2.metric("MSE", f"{mse:.2f}")
m3.metric("RMSE", f"{rmse:.2f} Lakhs")
m4.metric("R² Score", f"{r2:.3f}")

st.caption(
    "Lower MAE, MSE and RMSE generally indicate smaller errors. "
    "R² measures how well the model explains price variation "
    "in the test set."
)

# --------------------------------------------------
# 10. DATASET PREVIEW AND DOWNLOAD
# --------------------------------------------------
st.markdown("## 🧾 Dataset Preview")

st.download_button(
    label="⬇️ Download Dataset as CSV",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="house_prices_dataset.csv",
    mime="text/csv"
)

with st.expander("View dataset records"):
    st.dataframe(
        df,
        hide_index=True,
        use_container_width=True
    )

    st.write(
        f"Dataset size: {df.shape[0]} rows × {df.shape[1]} columns"
    )

# --------------------------------------------------
# 11. VISUALIZATIONS
# --------------------------------------------------
st.markdown("## 📈 Visualizations")

tab1, tab2, tab3 = st.tabs([
    "Area vs Price",
    "Actual vs Predicted",
    "Feature Coefficients"
])

# Area vs Price
with tab1:
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.scatter(
        df["Area_sqft"],
        df["Price_lakhs"],
        alpha=0.65,
        label="House records"
    )

    slope, line_intercept = np.polyfit(
        df["Area_sqft"],
        df["Price_lakhs"],
        1
    )

    x_values = np.linspace(
        df["Area_sqft"].min(),
        df["Area_sqft"].max(),
        100
    )

    ax.plot(
        x_values,
        slope * x_values + line_intercept,
        "--",
        linewidth=2,
        label="Area-only trend"
    )

    ax.set_title("House Area vs Price")
    ax.set_xlabel("Area (square feet)")
    ax.set_ylabel("Price (₹ lakhs)")
    ax.grid(alpha=0.25)
    ax.legend()

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# Actual vs Predicted
with tab2:
    comparison = pd.DataFrame({
        "Actual Price": y_test.to_numpy(),
        "Predicted Price": y_pred
    })

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(
        comparison["Actual Price"],
        comparison["Predicted Price"],
        s=45,
        alpha=0.75
    )

    low = min(
        comparison["Actual Price"].min(),
        comparison["Predicted Price"].min()
    )

    high = max(
        comparison["Actual Price"].max(),
        comparison["Predicted Price"].max()
    )

    ax.plot(
        [low, high],
        [low, high],
        "--",
        linewidth=2,
        label="Perfect prediction"
    )

    ax.set_title("Actual vs Predicted Prices")
    ax.set_xlabel("Actual Price (₹ lakhs)")
    ax.set_ylabel("Predicted Price (₹ lakhs)")
    ax.grid(alpha=0.25)
    ax.legend()

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.dataframe(
        comparison.round(2),
        hide_index=True,
        use_container_width=True
    )

# Feature Coefficients
with tab3:
    coefficient_df = pd.DataFrame({
        "Feature": features,
        "Coefficient": original_coefs
    }).sort_values("Coefficient")

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.barh(
        coefficient_df["Feature"],
        coefficient_df["Coefficient"]
    )

    ax.axvline(0, linewidth=1)
    ax.set_title("Feature Coefficients")
    ax.set_xlabel("Coefficient (₹ lakhs per feature unit)")
    ax.set_ylabel("Feature")
    ax.grid(axis="x", alpha=0.25)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.dataframe(
        coefficient_df.round(4),
        hide_index=True,
        use_container_width=True
    )

    st.caption(
        "The model constrains coefficient signs for this "
        "educational demonstration. This does not prove "
        "real-world causal effects."
    )

# --------------------------------------------------
# 12. REGRESSION EQUATION
# --------------------------------------------------
st.markdown("## 🧮 Regression Equation")

equation_parts = [
    f"({coef:+.4f} × {feature})"
    for feature, coef in zip(features, original_coefs)
]

equation = (
    f"Price = {intercept:.4f} "
    + " ".join(equation_parts)
)

st.code(equation, language="text")

# --------------------------------------------------
# 13. METRIC EXPLANATION
# --------------------------------------------------
with st.expander("Understand the metrics and coefficients"):
    st.markdown("""
    - **MAE:** Average absolute prediction error, in lakhs.
    - **MSE:** Average squared prediction error.
    - **RMSE:** Square root of MSE, in lakhs.
    - **R²:** How much target variation is explained by the model
      on the test split.
    - **Coefficient:** Estimated price change when one feature
      increases by one unit, holding the others constant.

    **Important:** The dataset is synthetic and the model uses
    sign-constrained regression for educational purposes.
    Replace it with verified property-sale data for real estimates.
    """)

# --------------------------------------------------
# 14. FOOTER
# --------------------------------------------------
st.divider()

st.caption(
    "Built with Streamlit, pandas, NumPy, Matplotlib "
    "and scikit-learn."
)