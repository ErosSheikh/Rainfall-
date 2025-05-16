import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans, DBSCAN

st.set_page_config(layout="wide", page_title="Indian Rainfall Analysis", page_icon="🌧️")

@st.cache_data
def load_data():
    df = pd.read_csv("DataSets/rainfaLLIndia.csv")
    df.drop_duplicates(inplace=True)
    df['Avg_Jun_Sep'] = df[['JUN', 'JUL', 'AUG', 'SEP']].mean(axis=1)
    df.sort_values(by=['subdivision', 'YEAR'], inplace=True)
    df['YoY_Change'] = df.groupby('subdivision')['Avg_Jun_Sep'].diff()
    df['Lag1_Avg'] = df.groupby('subdivision')['Avg_Jun_Sep'].shift(1)
    df['subdivision_encoded'] = df['subdivision'].astype('category').cat.codes
    df = df.fillna(0)
    return df

df = load_data()

st.title("🌧️ Indian Rainfall Dashboard (JUN–SEP Analysis)")

# Sidebar filters
st.sidebar.header("📊 Filter Data")
selected_sub = st.sidebar.selectbox("Choose Subdivision", sorted(df['subdivision'].unique()))
years = df['YEAR'].unique()
start_year, end_year = st.sidebar.select_slider("Select Year Range", options=years, value=(years.min(), years.max()))

filtered_df = df[(df['subdivision'] == selected_sub) & 
                 (df['YEAR'] >= start_year) & 
                 (df['YEAR'] <= end_year)]

tab1, tab2, tab3, tab4 = st.tabs(["📈 Trends", "📊 EDA", "🤖 ML Models", "📍 Clustering"])

# ======================== TAB 1 ========================
with tab1:
    st.subheader(f"📈 Trend of Rainfall in {selected_sub} ({start_year}–{end_year})")
    
    st.plotly_chart(
        px.line(filtered_df, x="YEAR", y="Avg_Jun_Sep", markers=True,
                title=f"Average Rainfall (Jun–Sep) in {selected_sub}",
                labels={"Avg_Jun_Sep": "Avg Rainfall (mm)", "YEAR": "Year"}))

    if not filtered_df.empty:
        st.subheader("🗓️ Monthly Rainfall Heatmap")
        monthly = filtered_df[['YEAR', 'JUN', 'JUL', 'AUG', 'SEP']].set_index('YEAR')
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(monthly, annot=True, fmt=".1f", cmap='YlGnBu', ax=ax, cbar_kws={'label': 'Rainfall (mm)'})
        st.pyplot(fig)

# ======================== TAB 2 ========================
with tab2:
    st.subheader("📊 Exploratory Data Analysis")
    
    st.markdown("#### 🧪 Distribution of Rainfall by Subdivision")
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.boxplot(data=df, x='subdivision', y='Avg_Jun_Sep', ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    st.pyplot(fig)

    st.markdown("#### 📉 Histogram of Average Rainfall (JUN–SEP)")
    fig2, ax2 = plt.subplots()
    sns.histplot(df['Avg_Jun_Sep'], bins=30, kde=True, ax=ax2)
    st.pyplot(fig2)

# ======================== TAB 3 ========================
with tab3:
    st.subheader("🤖 Machine Learning Models")

    le = LabelEncoder()
    df['subdivision_enc'] = le.fit_transform(df['subdivision'])
    X = df.drop(columns=['YoY_Change', 'subdivision', 'YEAR'])
    y = df['YoY_Change']

    xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

    model_choice = st.selectbox("Choose Model", ["Linear Regression", "Random Forest", "AdaBoost"])

    if model_choice == "Linear Regression":
        model = LinearRegression()
    elif model_choice == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        model = AdaBoostRegressor(n_estimators=30, random_state=42)

    model.fit(xtrain, ytrain)
    ypred = model.predict(xtest)

    st.write("**Model Performance:**")
    st.metric("R² Score", round(r2_score(ytest, ypred), 3))
    st.metric("RMSE", round(mean_squared_error(ytest, ypred, squared=False), 3))
    st.metric("MAE", round(mean_absolute_error(ytest, ypred), 3))

# ======================== TAB 4 ========================
with tab4:
    st.subheader("📍 Clustering Analysis")

    cluster_method = st.radio("Choose Clustering Method", ["KMeans", "DBSCAN"])
    cluster_df = df[['Avg_Jun_Sep', 'YoY_Change']]

    if cluster_method == "KMeans":
        k = st.slider("Number of Clusters", 2, 10, 5)
        km = KMeans(n_clusters=k, random_state=42)
        df['Cluster'] = km.fit_predict(cluster_df)
        st.plotly_chart(
            px.scatter(df, x='Avg_Jun_Sep', y='YoY_Change', color='Cluster', 
                       title='KMeans Clustering: Rainfall vs YoY Change'))
    else:
        eps = st.slider("DBSCAN eps", 0.1, 5.0, 1.0)
        min_samples = st.slider("min_samples", 2, 10, 5)
        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(cluster_df)
        df['DBSCAN_Cluster'] = labels
        st.plotly_chart(
            px.scatter(df, x='Avg_Jun_Sep', y='YoY_Change', color=df['DBSCAN_Cluster'].astype(str),
                       title='DBSCAN Clustering: Rainfall vs YoY Change'))

st.markdown("---")
st.caption("Developed with ❤️ using Streamlit | Data: IMD Rainfall Dataset")
