import streamlit as st
import pandas as pd
import joblib
import os

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide")

MODEL_DIR = "models"
VIS_DIR = "charts"

@st.cache_data
def load_csvs():
    comparison = pd.read_csv("model_comparison.csv")
    anomalies = pd.read_csv("anomaly_report.csv")
    clusters = pd.read_csv("cluster_assignments.csv")
    segment_metrics = pd.read_csv("segment_model_metrics.csv")
    return comparison, anomalies, clusters, segment_metrics

@st.cache_data
def load_raw_data():
    df = pd.read_csv(os.path.join("Dataset", "train.csv"), encoding="latin1")
    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)
    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month
    return df

@st.cache_resource
def load_models():
    sarima = joblib.load(os.path.join(MODEL_DIR, "sarima_model.pkl"))
    prophet = joblib.load(os.path.join(MODEL_DIR, "prophet_model.pkl"))
    xgb = joblib.load(os.path.join(MODEL_DIR, "xgboost_model.pkl"))
    return sarima, prophet, xgb

comparison_df, anomaly_df, cluster_df, segment_metrics_df = load_csvs()
raw_df = load_raw_data()
sarima_model, prophet_model, xgb_model = load_models()

st.title("Sales Forecasting & Demand Intelligence Dashboard")

page = st.sidebar.radio("Navigate", ["Sales Overview", "Forecast Explorer", "Anomaly Report", "Product Segments"])

if page == "Sales Overview":
    st.header("Sales Overview Dashboard")

    yearly_sales = raw_df.groupby("Year")["Sales"].sum().reset_index()
    st.subheader("Total Sales by Year")
    st.bar_chart(yearly_sales.set_index("Year"))

    monthly_sales_page = raw_df.groupby(["Year", "Month"])["Sales"].sum().reset_index()
    monthly_sales_page["Period"] = pd.to_datetime(monthly_sales_page["Year"].astype(str) + "-" + monthly_sales_page["Month"].astype(str))
    monthly_sales_page = monthly_sales_page.sort_values("Period")
    st.subheader("Monthly Sales Trend")
    st.line_chart(monthly_sales_page.set_index("Period")["Sales"])

    st.subheader("Sales by Region and Category")
    col1, col2 = st.columns(2)
    with col1:
        selected_region = st.selectbox("Region", ["All"] + sorted(raw_df["Region"].unique().tolist()))
    with col2:
        selected_category = st.selectbox("Category", ["All"] + sorted(raw_df["Category"].unique().tolist()))

    filtered = raw_df.copy()
    if selected_region != "All":
        filtered = filtered[filtered["Region"] == selected_region]
    if selected_category != "All":
        filtered = filtered[filtered["Category"] == selected_category]

    filtered_summary = filtered.groupby(["Region", "Category"])["Sales"].sum().reset_index()
    st.dataframe(filtered_summary)
    st.write(f"Filtered total: {filtered['Sales'].sum():,.2f}")

elif page == "Forecast Explorer":
    st.header("Forecast Explorer")

    tab1, tab2 = st.tabs(["Overall Models", "Category / Region Forecast"])

    with tab1:
        model_choice = st.selectbox("Select Model", ["SARIMA", "Prophet", "XGBoost"])
        horizon = st.slider("Forecast Horizon (months)", 1, 3, 3, key="overall_horizon")

        model_row = comparison_df[comparison_df["Model"] == model_choice].iloc[0]
        forecast_cols = ["Forecast_Month1", "Forecast_Month2", "Forecast_Month3"][:horizon]
        forecast_values = model_row[forecast_cols].values

        st.subheader(f"{model_choice} Forecast — Next {horizon} Month(s)")
        st.bar_chart(pd.Series(forecast_values, index=[f"Month {i+1}" for i in range(horizon)]))

        col1, col2, col3 = st.columns(3)
        col1.metric("MAE", f"{model_row['MAE']:,.2f}")
        col2.metric("RMSE", f"{model_row['RMSE']:,.2f}")
        col3.metric("MAPE", f"{model_row['MAPE']:.2f}%")

        best_model_row = comparison_df.loc[comparison_df["RMSE"].idxmin()]
        best_model_name = best_model_row["Model"]

        if model_choice == best_model_name:
            st.success(f"{best_model_name} has the lowest forecast error (RMSE) of the three models tested — this is the recommended model for production use.")
        else:
            st.info(f"{best_model_name} has a lower forecast error (RMSE {best_model_row['RMSE']:,.0f}) than {model_choice} (RMSE {model_row['RMSE']:,.0f}) and is the recommended model for production use.")

    with tab2:
        segment_choice = st.selectbox("Select Category or Region",
                                       segment_metrics_df["Segment"].tolist())
        seg_horizon = st.slider("Forecast Horizon (months)", 1, 3, 3, key="segment_horizon")

        seg_row = segment_metrics_df[segment_metrics_df["Segment"] == segment_choice].iloc[0]
        seg_forecast_cols = ["Forecast_Month1", "Forecast_Month2", "Forecast_Month3"][:seg_horizon]
        seg_forecast_values = seg_row[seg_forecast_cols].values.astype(float)

        st.subheader(f"{segment_choice} Forecast — Next {seg_horizon} Month(s)")
        st.bar_chart(pd.Series(seg_forecast_values, index=[f"Month {i+1}" for i in range(seg_horizon)]))

        avg_forecast = seg_forecast_values.mean()
        relative_error = (seg_row["MAE"] / avg_forecast) * 100

        col1, col2, col3 = st.columns(3)
        col1.metric("MAE", f"{seg_row['MAE']:,.2f}")
        col2.metric("RMSE", f"{seg_row['RMSE']:,.2f}")

        if relative_error > 30:
            col3.metric("Confidence", "Low", delta=f"{relative_error:.1f}% error", delta_color="inverse")
            st.warning(f"This forecast has a relative error of {relative_error:.1f}% — treat as a directional estimate, not a precise figure.")
        else:
            col3.metric("Confidence", "Higher", delta=f"{relative_error:.1f}% error", delta_color="off")
            st.caption(f"Relative error: {relative_error:.1f}% — reasonably reliable for planning.")

elif page == "Anomaly Report":
    st.header("Anomaly Report")

    st.subheader("Weekly Sales Anomaly Detection")
    st.image(os.path.join(VIS_DIR, "anomaly_detection.png"))

    st.subheader("Detected Anomaly Weeks")
    st.dataframe(anomaly_df[anomaly_df["IsoAnomaly"] == 1][["Date", "Sales"]].reset_index(drop=True))

elif page == "Product Segments":
    st.header("Product Demand Segments")

    st.subheader("Sub-Category Demand Clusters")
    st.image(os.path.join(VIS_DIR, "cluster_scatter.png"))

    st.subheader("Sub-Category Cluster Assignments")
    st.dataframe(cluster_df[["Sub-Category", "Cluster"]].reset_index(drop=True))
