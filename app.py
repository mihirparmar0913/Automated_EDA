import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_option('deprecation.showPyplotGlobalUse', False)

st.title("📊 Automated EDA Tool")

# Upload file
uploaded_file = st.file_uploader("Upload your dataset (CSV/Excel)", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Read data
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # --- Basic Info ---
    st.subheader("🔍 Data Preview")
    st.write(df.head())

    st.subheader("📏 Dataset Info")
    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    st.subheader("🧾 Column Details")
    st.write(df.dtypes)

    st.subheader("❌ Missing Values")
    st.write(df.isnull().sum())

    st.subheader("📊 Summary Statistics")
    st.write(df.describe(include="all"))

    # --- Visualizations ---
    st.subheader("📈 Data Visualizations")

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

    # Histogram
    if len(numeric_cols) > 0:
        st.write("### Histogram")
        col = st.selectbox("Select column for histogram", numeric_cols)
        plt.figure(figsize=(6,4))
        sns.histplot(df[col].dropna(), kde=True, bins=30)
        st.pyplot()

    # Boxplot
    if len(numeric_cols) > 0:
        st.write("### Boxplot")
        col = st.selectbox("Select column for boxplot", numeric_cols, key="boxplot")
        plt.figure(figsize=(6,4))
        sns.boxplot(x=df[col].dropna())
        st.pyplot()

    # Correlation heatmap
    if len(numeric_cols) > 1:
        st.write("### Correlation Heatmap")
        plt.figure(figsize=(8,6))
        sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm")
        st.pyplot()
