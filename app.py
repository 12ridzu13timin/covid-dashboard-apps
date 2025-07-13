import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load data
df = pd.read_csv("data,covid.csv")

# Basic cleaning and ensuring correct data types
df['date'] = df['date'].astype(int)
df['cases_new'] = pd.to_numeric(df['cases_new'], errors='coerce')
df['deaths_new'] = pd.to_numeric(df['deaths_new'], errors='coerce')

# Sidebar filters
st.sidebar.title("Filters")
states = df['state'].unique()
selected_state = st.sidebar.selectbox("Select a state", states)
selected_year = st.sidebar.selectbox("Select a year", sorted(df['year'].unique()))

# Filtered data for selected state
state_df = df[df['state'] == selected_state]

# Forecasting
X = state_df[['date']]
y = state_df['cases_new']
model = LinearRegression()
model.fit(X, y)

# Predict future cases (next 5 years)
future_years = pd.DataFrame({'year': list(range(df['year'].max() + 1, df['year'].max() + 6))})
future_cases = model.predict(future_years)

# Layout
st.title("📊 COVID-19 Dashboard with Forecasting")
st.subheader(f"New Cases in {selected_state}")

# Line chart of cases
st.line_chart(state_df.set_index('year')['cases_new'])

# Forecast vs actual chart
st.subheader("📈 Forecast: New Cases (Next 5 Years)")
fig, ax = plt.subplots()
ax.plot(X, y, label="Actual")
ax.plot(future_years, future_cases, linestyle='--', label="Forecast", color='orange')
ax.set_xlabel("Year")
ax.set_ylabel("New Cases")
ax.legend()
st.pyplot(fig)

# Pie chart of % cases per state for selected year
st.subheader(f"🥧 Case Distribution by State in {selected_year}")
year_df = df[df['date'] == selected_year]
cases_per_state = year_df.groupby('state')['cases_new'].sum()
fig2, ax2 = plt.subplots()
ax2.pie(cases_per_state, labels=cases_per_state.index, autopct='%1.1f%%', startangle=90)
ax2.axis('equal')
st.pyplot(fig2)

# Bar chart of deaths per year for selected state
st.subheader(f"📊 Deaths in {selected_state} by Year")
deaths_by_year = state_df.groupby('date')['deaths_new'].sum()
fig3, ax3 = plt.subplots()
ax3.bar(deaths_by_year.index, deaths_by_year.values, color='red')
ax3.set_xlabel("date")
ax3.set_ylabel("Deaths")
st.pyplot(fig3)

# Optional: download filtered data
st.subheader("⬇️ Download Filtered Data")
st.download_button(
    label="Download CSV",
    data=state_df.to_csv(index=False).encode('utf-8'),
    file_name=f"{selected_state}_covid_data.csv",
    mime='text/csv'
)
