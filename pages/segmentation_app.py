import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import streamlit as st

st.title("Segmentation App")
st.write("This is the Segmentation app.")
# Your existing code for Segmentation App goes here

st.set_page_config(layout="wide")
st.title("Customer Segmentation Dashboard")

# Upload segmented data

# Upload segmented data
uploaded_file = st.file_uploader("Upload your segmented customer file (CSV)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")

    # Sidebar filters
    st.sidebar.header("Filter Customers")

    # Segment filter
    segments = df['Segment_Label'].unique().tolist()
    selected_segments = st.sidebar.multiselect("Select Segment(s)", segments, default=segments)

    # Age filter
    min_age = int(df['Age'].min())
    max_age = int(df['Age'].max())
    age_range = st.sidebar.slider("Select Age Range", min_age, max_age, (min_age, max_age))

    # Income filter
    if 'Monthly_Income_(GHS)' in df.columns:
        min_income = float(df['Monthly_Income_(GHS)'].min())
        max_income = float(df['Monthly_Income_(GHS)'].max())
        income_range = st.sidebar.slider("Select Monthly Income (GHS)", min_income, max_income, (min_income, max_income))

    # Employment sector filter (if available)
    if 'Employment_Sector' in df.columns:
        sectors = df['Employment_Sector'].dropna().unique().tolist()
        selected_sectors = st.sidebar.multiselect("Employment Sector", sectors, default=sectors)
    else:
        selected_sectors = None

    # Apply filters
    filtered_df = df[
        (df['Segment_Label'].isin(selected_segments)) &
        (df['Age'].between(*age_range)) &
        (df['Monthly_Income_(GHS)'].between(*income_range))
    ]
    if selected_sectors:
        filtered_df = filtered_df[filtered_df['Employment_Sector'].isin(selected_sectors)]

    st.subheader("Filtered Data Preview")
    st.dataframe(filtered_df.head())

    # Segment distribution
    st.subheader("Filtered Segment Distribution")
    seg_counts = filtered_df['Segment_Label'].value_counts().reset_index()
    seg_counts.columns = ['Segment', 'Count']
    fig = px.pie(seg_counts, values='Count', names='Segment', title="Segment Distribution (Filtered)")
    st.plotly_chart(fig)

    # Summary stats
    st.subheader("Filtered Segment Summary")
    summary_metrics = filtered_df.groupby('Segment_Label')[[
        'Current_Balance', 'Transaction_Frequency', 'Age', 'Default_Count'
    ]].agg(['mean', 'median', 'sum'])
    st.dataframe(summary_metrics)

    # PCA scatter
    st.subheader("Filtered PCA Scatter Plot")
    if 'PCA1' in filtered_df.columns and 'PCA2' in filtered_df.columns:
        fig_pca = px.scatter(
            filtered_df, x='PCA1', y='PCA2', color='Segment_Label',
            title='PCA Plot (Filtered)',
            opacity=0.7
        )
        st.plotly_chart(fig_pca)

    # Feature Explorer
    st.subheader("Feature Explorer")
    selected_feature = st.selectbox("Choose a feature", filtered_df.select_dtypes(include=['number']).columns)
    fig_box = px.box(filtered_df, x='Segment_Label', y=selected_feature, color='Segment_Label')
    st.plotly_chart(fig_box)

    # Download filtered data
    st.download_button("Download Filtered Data", data=filtered_df.to_csv(index=False), file_name="filtered_customers.csv")

else:
    st.info("Please upload a file to proceed.")

