import streamlit as st
import pandas as pd

# Title
st.title("COVID-19 Dashboard (Malaysia)")

# Load dataset
df = pd.read_csv("covid.csv")

# Sidebar: filter by state
states = df['state'].unique()
selected_state = st.sidebar.selectbox("Select a State", states)

# Filter data
filtered_df = df[df['state'] == selected_state]

# Show filtered table
st.subheader(f"Data for {selected_state}")
st.dataframe(filtered_df)

# Summary
st.subheader("Summary Statistics")
st.write(f"Total New Cases: {filtered_df['cases_new'].sum():,.0f}")
st.write(f"Total Recovered: {filtered_df['cases_recovered'].sum():,.0f}")

# Line chart
st.subheader("New Cases Over Time")
filtered_df['date'] = pd.to_datetime(filtered_df['date'])
filtered_df = filtered_df.sort_values('date')
st.line_chart(filtered_df.set_index('date')['cases_new'])
