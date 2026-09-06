from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from src.clustering_pipeline import CustomerSegmentationModel  # noqa: E402
from config.recommendations import RECOMMENDATIONS  # noqa: E402

DATA_PATH = ROOT / "data" / "customer_segments.csv"
MODEL_PATH = ROOT / "models" / "clustering_model.joblib"

st.set_page_config(
    page_title="Customer Personality Analysis",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded",
)

FEATURES = [
    "Customer_Age",
    "Income",
    "Total_Spending",
    "Recency",
    "Customer_Tenure",
    "Family_Size",
    "Total_Campaign_Acceptance",
    "NumWebPurchases",
    "NumStorePurchases",
    "NumCatalogPurchases",
]

SEGMENTS = [
    "Budget Shoppers",
    "Emerging / Mid-Tier Customers",
    "High-Value Customers",
    "Premium & Campaign-Responsive Customers",
]


def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource(show_spinner="Loading saved segmentation model...")
def load_model():
    # The exact production artifact is loaded; no new model is trained here.
    return CustomerSegmentationModel.load(r"C:\Users\os\Desktop\customer-personality-app\models\clustering_model.joblib")


def money(value):
    return f"${value:,.0f}"


def segment_profile(df, segment):
    return df[df["Segment_Name"] == segment]


def show_profile(df, segment):
    rows = segment_profile(df, segment)
    if rows.empty:
        st.warning("No customers found for this segment.")
        return

    cols = st.columns(5)
    metrics = [
        ("Customers", f"{len(rows):,}"),
        ("Share", f"{len(rows) / len(df) * 100:.1f}%"),
        ("Avg Income", money(rows["Income"].mean())),
        ("Avg Spending", money(rows["Total_Spending"].mean())),
        ("Avg Recency", f"{rows['Recency'].mean():.0f} days"),
    ]
    for col, (label, value) in zip(cols, metrics):
        col.metric(label, value)


def dashboard(df):
    st.title("🏠 Customer Personality Dashboard")
    st.caption("Customer overview powered by the final segmentation dataset and saved ML model.")

    cols = st.columns(5)
    kpis = [
        ("Total Customers", f"{len(df):,}"),
        ("Segments", f"{df['Segment_Name'].nunique()}"),
        ("Avg Income", money(df["Income"].mean())),
        ("Avg Spending", money(df["Total_Spending"].mean())),
        ("Avg Recency", f"{df['Recency'].mean():.0f} days"),
    ]
    for col, (label, value) in zip(cols, kpis):
        col.metric(label, value)

    st.divider()
    c1, c2 = st.columns(2)
    counts = df["Segment_Name"].value_counts().reindex(SEGMENTS).fillna(0).reset_index()
    counts.columns = ["Segment_Name", "Customers"]
    with c1:
        fig = px.bar(counts, x="Customers", y="Segment_Name", orientation="h", title="Customer Distribution by Segment")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        spending = df.groupby("Segment_Name", as_index=False)["Total_Spending"].mean()
        spending["Segment_Name"] = pd.Categorical(spending["Segment_Name"], SEGMENTS, ordered=True)
        spending = spending.sort_values("Segment_Name")
        fig = px.bar(spending, x="Segment_Name", y="Total_Spending", title="Average Spending by Segment")
        fig.update_xaxes(tickangle=-25)
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        income = df.groupby("Segment_Name", as_index=False)["Income"].mean()
        fig = px.bar(income, x="Segment_Name", y="Income", title="Average Income by Segment")
        fig.update_xaxes(tickangle=-25)
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        channel = df.groupby("Segment_Name", as_index=False)[
            ["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"]
        ].mean().melt(id_vars="Segment_Name", var_name="Channel", value_name="Average Purchases")
        fig = px.bar(channel, x="Segment_Name", y="Average Purchases", color="Channel", barmode="group", title="Average Purchases by Channel")
        fig.update_xaxes(tickangle=-25)
        st.plotly_chart(fig, use_container_width=True)


def segmentation_page(df):
    st.title("👥 Customer Segmentation")
    segment = st.selectbox("Select segment", ["All Segments"] + SEGMENTS)
    view = df if segment == "All Segments" else segment_profile(df, segment)

    if segment != "All Segments":
        show_profile(df, segment)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(
            view,
            x="Income",
            y="Total_Spending",
            color="Segment_Name",
            hover_data=["ID", "Recency", "Total_Purchases"],
            title="Income vs Total Spending",
        )
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.box(view, x="Segment_Name", y="Total_Spending", title="Spending Distribution")
        fig.update_xaxes(tickangle=-25)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Segment Profile")
    profile_cols = [
        "Income", "Total_Spending", "Recency", "Customer_Age", "Customer_Tenure",
        "Family_Size", "Total_Campaign_Acceptance", "NumWebPurchases",
        "NumStorePurchases", "NumCatalogPurchases",
    ]
    profile = view.groupby("Segment_Name")[profile_cols].mean().round(2)
    st.dataframe(profile, use_container_width=True)


def prediction_page(df, model):
    st.title("🔍 Customer Prediction")
    st.write("Enter the final engineered features required by the production segmentation model.")
    st.info("The prediction below calls the saved K-Means model and its saved StandardScaler; no cluster label is manually assigned.")

    mode = st.radio("Prediction mode", ["Existing Customer", "New Customer"], horizontal=True)

    if mode == "Existing Customer":
        customer_id = st.selectbox("Customer ID", sorted(df["ID"].dropna().unique().tolist()))
        row = df[df["ID"] == customer_id].iloc[[0]]
        if st.button("Predict Segment", type="primary"):
            predicted_cluster = int(model.predict(row[FEATURES])[0])
            predicted_name = model.predict_segment_names(row[FEATURES])[0]
            display_prediction(df, row.iloc[0], predicted_cluster, predicted_name)
    else:
        with st.form("new_customer"):
            c1, c2 = st.columns(2)
            with c1:
                age = st.number_input("Customer Age", min_value=18, max_value=100, value=35)
                income = st.number_input("Income", min_value=0.0, value=50000.0, step=1000.0)
                spending = st.number_input("Total Spending", min_value=0.0, value=500.0, step=50.0)
                recency = st.number_input("Recency (days)", min_value=0, value=30, step=1)
                tenure = st.number_input("Customer Tenure (days)", min_value=1, value=365, step=1)
            with c2:
                family = st.number_input("Family Size", min_value=1, value=2, step=1)
                campaigns = st.number_input("Accepted Campaigns", min_value=0, value=0, step=1)
                web = st.number_input("Web Purchases", min_value=0, value=5, step=1)
                store = st.number_input("Store Purchases", min_value=0, value=5, step=1)
                catalog = st.number_input("Catalog Purchases", min_value=0, value=2, step=1)
            submitted = st.form_submit_button("Predict Segment", type="primary")

        if submitted:
            new_customer = pd.DataFrame([{
                "Customer_Age": age,
                "Income": income,
                "Total_Spending": spending,
                "Recency": recency,
                "Customer_Tenure": tenure,
                "Family_Size": family,
                "Total_Campaign_Acceptance": campaigns,
                "NumWebPurchases": web,
                "NumStorePurchases": store,
                "NumCatalogPurchases": catalog,
            }])
            predicted_cluster = int(model.predict(new_customer)[0])
            predicted_name = model.predict_segment_names(new_customer)[0]
            display_prediction(df, new_customer.iloc[0], predicted_cluster, predicted_name)


def display_prediction(df, customer, cluster, segment):
    st.success(f"Predicted Segment: **{segment}**")
    cols = st.columns(4)
    profile = segment_profile(df, segment)
    values = [
        ("Cluster", str(cluster)),
        ("Segment Size", f"{len(profile):,}"),
        ("Avg Spending", money(profile["Total_Spending"].mean())),
        ("Avg Income", money(profile["Income"].mean())),
    ]
    for col, (label, value) in zip(cols, values):
        col.metric(label, value)

    st.subheader("Customer Inputs")
    st.dataframe(pd.DataFrame([customer]), use_container_width=True)

    rec = RECOMMENDATIONS.get(segment, {})
    st.subheader("Marketing Recommendation")
    st.warning("Module 8 recommendations are placeholders until the original recommendations file is restored.")
    st.write(rec.get("summary", "Add the Module 8 recommendation here."))
    for action in rec.get("actions", []):
        st.write(f"- {action}")


def recommendations_page(df):
    st.title("💡 Marketing Recommendations")
    segment = st.selectbox("Select segment", SEGMENTS)
    rows = segment_profile(df, segment)
    show_profile(df, segment)
    rec = RECOMMENDATIONS[segment]
    st.subheader(rec["priority"])
    st.write(rec["summary"])
    for action in rec["actions"]:
        st.write(f"- {action}")


def downloads_page(df):
    st.title("📥 Downloads")
    st.write("Download the segmentation results or a filtered subset.")
    segment = st.selectbox("Filter segment", ["All Segments"] + SEGMENTS)
    view = df if segment == "All Segments" else segment_profile(df, segment)
    csv = view.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, "customer_segments_filtered.csv", "text/csv")
    st.dataframe(view.head(100), use_container_width=True)


def main():
    try:
        df = load_data()
        model = load_model()
    except Exception as exc:
        st.error("The application could not load the saved production model/data.")
        st.exception(exc)
        st.stop()

    st.sidebar.title("👥 Customer Personality")
    st.sidebar.caption("ML Customer Segmentation Application")
    page = st.sidebar.radio(
        "Navigation",
        ["🏠 Dashboard", "👥 Customer Segmentation", "🔍 Customer Prediction", "💡 Marketing Recommendations", "📥 Downloads"],
    )

    if page == "🏠 Dashboard":
        dashboard(df)
    elif page == "👥 Customer Segmentation":
        segmentation_page(df)
    elif page == "🔍 Customer Prediction":
        prediction_page(df, model)
    elif page == "💡 Marketing Recommendations":
        recommendations_page(df)
    else:
        downloads_page(df)


if __name__ == "__main__":
    main()
