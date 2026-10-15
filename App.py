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
# 2. CUSTOM CSS - IMPROVED COLOR CONTRAST
# --------------------------------------------------
st.markdown("""
<style>
    /* Main page */
    .stApp {
        background: linear-gradient(
            135deg, #edf2ff 0%, #dce6ff 50%, #eeeaff 100%
        );
        color: #172554;
    }
 
    /* Default text colors */
    .stApp,
    .stApp p,
    .stApp label,
    .stApp li,
    .stApp span {
        color: #1e293b;
    }
 
    [data-testid="stHeader"] {
        background: transparent;
    }
 
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(
            180deg, #172554 0%, #3730a3 100%
        );
    }
 
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span {
        color: #ffffff !important;
    }
 
    /* Sidebar input text stays dark inside white input fields */
    [data-testid="stSidebar"] input {
        color: #111827 !important;
        background-color: #ffffff !important;
    }
 
    /* Headings */
    h1, h2, h3, h4 {
        color: #172554 !important;
        font-weight: 750 !important;
    }
 
    /* Hero banner */
    .hero {
        padding: 28px 32px;
        border-radius: 18px;
        background: linear-gradient(
            120deg, #f4b86a, #f59e55
        );
        color: #ffffff !important;
        margin-bottom: 20px;
        box-shadow: 0 8px 24px rgba(30, 41, 59, 0.12);
    }
 
    .hero h1 {
        color: #ffffff !important;
        font-size: 2.2rem;
        margin-bottom: 8px;
    }
 
    .hero p {
        color: #ffffff !important;
        margin-bottom: 0;
    }
 
    /* Prediction card */
    .prediction-card {
        background: linear-gradient(
            120deg, #3730a3, #2563eb
        );
        padding: 28px 20px;
        border-radius: 18px;
        color: #ffffff !important;
        text-align: center;
        box-shadow: 0 8px 20px rgba(37, 99, 235, 0.18);
    }
 
    .prediction-card h3 {
        color: #ffffff !important;
        margin-bottom: 16px;
    }
 
    .prediction-card .prediction-price {
        color: #ffffff !important;
        font-size: 2.4rem;
        font-weight: 800;
        margin: 12px 0;
    }
 
    .prediction-card p {
        color: #e0e7ff !important;
    }
 
    /* Metric cards - FIXED INVISIBLE TEXT */
    [data-testid="stMetric"] {
        background: #ffffff !important;
        padding: 18px !important;
        border-radius: 14px !important;
        border: 1px solid #c7d2fe !important;
        box-shadow: 0 4px 12px rgba(30, 41, 59, 0.06);
    }
 
    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] p,
    [data-testid="stMetricLabel"] label {
        color: #475569 !important;
        opacity: 1 !important;
        font-size: 0.92rem !important;
    }
 
    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] div,
    [data-testid="stMetricValue"] span {
        color: #172554 !important;
        opacity: 1 !important;
        font-weight: 750 !important;
    }
 
    [data-testid="stMetricDelta"] {
        opacity: 1 !important;
    }
 
    /* Input widgets */
    .stNumberInput label,
    .stTextInput label,
    .stSelectbox label {
        color: #1e293b !important;
        font-weight: 600 !important;
    }
 
    .stNumberInput input,
    .stTextInput input {
        color: #111827 !important;
        background: #ffffff !important;
    }
 
    /* Buttons */
    div.stButton > button {
        background: #3730a3 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
    }
 
    div.stButton > button:hover {
        background: #1d4ed8 !important;
        color: #ffffff !important;
    }
 
    /* Dataframe */
    [data-testid="stDataFrame"] {
        border: 1px solid #cbd5e1;
        border-radius: 10px;
        overflow: hidden;
    }
 
    /* Tabs */
    .stTabs [data-baseweb="tab"] {
        color: #334155 !important;
        font-weight: 600;
    }
 
    .stTabs [aria-selected="true"] {
        color: #3730a3 !important;
    }
 
    /* Alerts */
    [data-testid="stAlert"] p {
        color: #1e3a8a !important;
    }
 
    /* Footer and captions */
    .stCaption,
    [data-testid="stCaptionContainer"] {
        color: #475569 !important;
    }
 
    /* Code block */
    .stCode pre {
        background: #f8fafc !important;
        color: #0f172a !important;
    }
</style>
""", unsafe_allow_html=True)
 
 
# --------------------------------------------------
# 3. SAMPLE DATASET
# --------------------------------------------------
data = {
    "Area_sqft": [
        800, 1000, 1200, 1500, 1800,
        2000, 2200, 2500, 2800, 3000
    ],
    "Bedrooms": [2, 2, 2, 3, 3, 3, 4, 4, 4, 5],
    "Bathrooms": [1, 2, 2, 2, 3, 3, 3, 3, 4, 4],
    "Age_years": [20, 15, 12, 10, 8, 7, 5, 4, 3, 2],
    "Parking": [0, 1, 1, 1, 1, 2, 2, 2, 2, 2],
    "Price_lakhs": [
        30, 40, 48, 60, 72, 82, 95, 110, 128, 145
    ]
}
 
df = pd.DataFrame(data)
 
features = [
    "Area_sqft",
    "Bedrooms",
    "Bathrooms",
    "Age_years",
    "Parking"
]
 
X = df[features]
y = df["Price_lakhs"]
 
 
# --------------------------------------------------
# 4. TRAIN MODEL AND EVALUATE
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
 
model = LinearRegression()
model.fit(X_train, y_train)
 
y_pred = model.predict(X_test)
 
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
 
 
# --------------------------------------------------
# 5. HEADER
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
 
 
# --------------------------------------------------
# 6. SIDEBAR - FEATURE INPUTS
# --------------------------------------------------
st.sidebar.title("🏡 Predict House Price")
st.sidebar.write("Enter the house details.")
 
area = st.sidebar.number_input(
    "Area (square feet)",
    min_value=300, max_value=10000,
    value=2100, step=100
)
 
bedrooms = st.sidebar.number_input(
    "Number of Bedrooms",
    min_value=1, max_value=10,
    value=3, step=1
)
 
bathrooms = st.sidebar.number_input(
    "Number of Bathrooms",
    min_value=1, max_value=10,
    value=3, step=1
)
 
age = st.sidebar.number_input(
    "House Age (years)",
    min_value=0, max_value=100,
    value=5, step=1
)
 
parking = st.sidebar.number_input(
    "Parking Spaces",
    min_value=0, max_value=10,
    value=2, step=1
)
 
input_data = pd.DataFrame([{
    "Area_sqft": area,
    "Bedrooms": bedrooms,
    "Bathrooms": bathrooms,
    "Age_years": age,
    "Parking": parking
}])
 
predicted_price = model.predict(input_data)[0]
 
st.sidebar.divider()
st.sidebar.subheader("Selected Features")
st.sidebar.dataframe(input_data, hide_index=True, use_container_width=True)
 
 
# --------------------------------------------------
# 7. PREDICTION RESULT
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
 
st.info(
    "Demonstration dataset: only 10 illustrative records. "
    "Use a larger, representative dataset for real-world estimates."
)
 
 
# --------------------------------------------------
# 8. MODEL PERFORMANCE
# --------------------------------------------------
st.markdown("## 📊 Model Performance")
 
m1, m2, m3, m4 = st.columns(4)
 
m1.metric("MAE", f"{mae:.2f} Lakhs")
m2.metric("MSE", f"{mse:.2f}")
m3.metric("RMSE", f"{rmse:.2f} Lakhs")
m4.metric("R² Score", f"{r2:.3f}")
 
st.caption(
    "Lower MAE, MSE and RMSE generally indicate smaller errors. "
    "R² measures how well the model explains price variation."
)
 
 
# --------------------------------------------------
# 9. DATASET PREVIEW
# --------------------------------------------------
st.markdown("## 🧾 Dataset Preview")
 
with st.expander("View dataset records", expanded=True):
    st.dataframe(df, hide_index=True, use_container_width=True)
    st.write(f"Dataset size: {df.shape[0]} rows × {df.shape[1]} columns")
 
 
# --------------------------------------------------
# 10. VISUALIZATIONS
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
        alpha=0.85,
        label="House records"
    )
 
    slope, intercept = np.polyfit(
        df["Area_sqft"], df["Price_lakhs"], 1
    )
 
    x_values = np.linspace(
        df["Area_sqft"].min(),
        df["Area_sqft"].max(),
        100
    )
 
    ax.plot(
        x_values,
        slope * x_values + intercept,
        "--",
        linewidth=2,
        label="Area-only trend"
    )
 
    ax.set_title("House Area vs Price", color="#172554")
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
        s=80,
        alpha=0.85
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
        [low, high], [low, high],
        "--", linewidth=2,
        label="Perfect prediction"
    )
 
    ax.set_title("Actual vs Predicted Prices", color="#172554")
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
        "Coefficient": model.coef_
    }).sort_values("Coefficient")
 
    fig, ax = plt.subplots(figsize=(9, 5))
 
    ax.barh(
        coefficient_df["Feature"],
        coefficient_df["Coefficient"],
        color="#4f46e5"
    )
 
    ax.axvline(0, linewidth=1, color="#334155")
    ax.set_title("Feature Coefficients", color="#172554")
    ax.set_xlabel("Coefficient value")
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
        "Coefficients estimate the change in predicted price "
        "when a feature increases by one unit, holding others "
        "constant. Results can be unstable with very little data."
    )
 
 
# --------------------------------------------------
# 11. MODEL EQUATION
# --------------------------------------------------
st.markdown("## 🧮 Regression Equation")
 
equation_parts = [
    f"({coef:.4f} × {feature})"
    for feature, coef in zip(features, model.coef_)
]
 
equation = (
    f"Price = {model.intercept_:.4f} + "
    " + ".join(equation_parts)
)
 
st.code(equation, language="text")
 
with st.expander("Understand the metrics"):
    st.markdown("""
    - **MAE:** Average absolute prediction error.
    - **MSE:** Average squared prediction error.
    - **RMSE:** Square root of MSE, in price units.
    - **R²:** Fraction of target variation explained by the model.
    """)
 
 
# --------------------------------------------------
# 12. FOOTER
# --------------------------------------------------
st.divider()
st.caption(
    "Built with Streamlit, pandas, Matplotlib and scikit-learn."
)
 