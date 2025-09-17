import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
from sklearn.preprocessing import StandardScaler, MinMaxScaler

st.title("Automated EDA & Preprocessing App")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.write(df.head())
    st.write(f"Shape: {df.shape}")
    
    # Separate numerical and categorical columns
    num_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
    cat_cols = df.select_dtypes(exclude=['int64','float64']).columns.tolist()
    
    # ----- Basic Info -----
    st.subheader("Missing Values")
    st.write(df.isnull().sum())
    
    st.subheader("Data Types")
    st.write(df.dtypes)

    st.subheader("Summary Statistics")
    st.write(df.describe(include="all"))
    
    # ----- Univariate -----
    st.subheader("Univariate Analysis (Numerical Only)")
    if num_cols:
        col = st.selectbox("Select Numeric Column", num_cols)
        fig, ax = plt.subplots()
        sns.histplot(df[col].dropna(), kde=True, ax=ax)
        st.pyplot(fig)
    else:
        st.warning("No numerical columns found for univariate analysis.")
    
    # ----- Custom Relationship Analysis -----
    st.subheader("Custom Relationship Analysis")
    analysis_type = st.radio("Choose Analysis Type", ["Categorical vs Categorical", "Categorical vs Continuous", "Continuous vs Continuous"])
    
    if analysis_type == "Categorical vs Categorical":
        if len(cat_cols) >= 2:
            col1 = st.selectbox("Select First Categorical Column", cat_cols, key="cat1")
            col2 = st.selectbox("Select Second Categorical Column", cat_cols, key="cat2")
            if col1 != col2:
                fig, ax = plt.subplots()
                ct = pd.crosstab(df[col1], df[col2])
                sns.heatmap(ct, annot=True, fmt="d", cmap="Blues", ax=ax)
                st.pyplot(fig)
        else:
            st.warning("Not enough categorical columns for this analysis.")
    
    elif analysis_type == "Categorical vs Continuous":
        if cat_cols and num_cols:
            cat_col = st.selectbox("Select Categorical Column", cat_cols, key="cat_cont")
            num_col = st.selectbox("Select Numeric Column", num_cols, key="num_cont")
            fig, ax = plt.subplots()
            sns.boxplot(x=df[cat_col], y=df[num_col], ax=ax)
            st.pyplot(fig)
        else:
            st.warning("Need at least one categorical and one numerical column.")
    
    elif analysis_type == "Continuous vs Continuous":
        if len(num_cols) >= 2:
            col1 = st.selectbox("Select First Numeric Column", num_cols, key="num1")
            col2 = st.selectbox("Select Second Numeric Column", num_cols, key="num2")
            if col1 != col2:
                fig, ax = plt.subplots()
                sns.scatterplot(x=df[col1], y=df[col2], ax=ax)
                st.pyplot(fig)
        else:
            st.warning("Not enough numerical columns for this analysis.")
    
    # ----- Correlation Heatmap -----
    st.subheader("Correlation Heatmap (Numerical Columns)")
    if len(num_cols) > 1:
        fig, ax = plt.subplots()
        sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    
    # ----- Pairplot -----
    st.subheader("Multivariate Pairplot (Numerical Columns)")
    if len(num_cols) > 1:
        sampled_df = df.sample(min(200, len(df)))
        fig = sns.pairplot(sampled_df[num_cols])
        st.pyplot(fig)
    
    # ----- Preprocessing -----
    st.subheader("Preprocessing Options")
    option = st.radio("Choose scaling method:", ["None", "Standardization (Z-score)", "Normalization (Min-Max)"])
    
    processed_df = df.copy()
    if option != "None" and num_cols:
        scaler = StandardScaler() if option == "Standardization (Z-score)" else MinMaxScaler()
        processed_df[num_cols] = scaler.fit_transform(processed_df[num_cols])
        st.success(f"{option} applied on numerical columns!")
        st.write(processed_df.head())
    
    # ----- Export Clean Data -----
    st.subheader("Download Processed Data")
    buffer = BytesIO()
    processed_df.to_csv(buffer, index=False)
    buffer.seek(0)
    st.download_button("Download CSV", buffer, file_name="processed_data.csv", mime="text/csv")
