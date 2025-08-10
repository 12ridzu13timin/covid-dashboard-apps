import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ===========================
# Load & clean data
# ===========================
df = pd.read_csv("data_covid.csv")  # ✅ fixed filename
df.columns = df.columns.str.strip()

# Convert date column to datetime
df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y', errors='coerce')

# Extract year
df['year'] = df['date'].dt.year

# Ensure numeric columns
for col in ['cases_new', 'deaths_new']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with missing values in main columns
df = df.dropna(subset=['cases_new', 'deaths_new', 'year'])

# ===========================
# Sidebar filters
# ===========================
st.sidebar.title("Filters")

# State selection
states = sorted(df['state'].dropna().unique())
selected_state = st.sidebar.selectbox("Select a state", states)

# Filtered state data
state_df = df[df['state'] == selected_state]

# ===========================
# Group yearly data
# ===========================
yearly_data = state_df.groupby('year', as_index=False).agg({
    'cases_new': 'sum',
    'deaths_new': 'sum'
})

# ===========================
# Forecasting Cases
# ===========================
X = yearly_data[['year']]
y = yearly_data['cases_new']

model = LinearRegression()
model.fit(X, y)

# Predict for next 5 years
future_years = pd.DataFrame({
    'year': list(range(yearly_data['year'].max() + 1, yearly_data['year'].max() + 6))
})
future_cases = model.predict(future_years)

# ===========================
# Dashboard Title
# ===========================
st.title("COVID-19 Dashboard with Forecasting")
st.subheader(f"State: {selected_state}")

# ===========================
# Line Chart: Actual Cases
# ===========================
st.subheader("Yearly New Cases")
st.line_chart(yearly_data.set_index('year')['cases_new'])

# ===========================
# Forecast Chart
# ===========================
fig_forecast, ax_forecast = plt.subplots()
ax_forecast.plot(X, y, label="Actual", marker='o')
ax_forecast.plot(future_years, future_cases, '--', label="Forecast", marker='x')
ax_forecast.set_xlabel("Year")
ax_forecast.set_ylabel("New Cases")
ax_forecast.set_title("Forecasted New Cases")
ax_forecast.legend()
st.pyplot(fig_forecast)

# ===========================
# Pie Chart: % of Cases by State
# ===========================
years = sorted(df['year'].unique())
selected_year = st.sidebar.selectbox("Select a year for case distribution", years)

yearly_state = df[df['year'] == selected_year].groupby('state')['cases_new'].sum()

st.subheader(f"Case Distribution by State in {selected_year}")
fig_pie, ax_pie = plt.subplots()
ax_pie.pie(yearly_state, labels=yearly_state.index, autopct='%1.1f%%', startangle=90)
ax_pie.axis('equal')
st.pyplot(fig_pie)

# ===========================
# Line Chart: Deaths per Year
# ===========================
st.subheader(f"Deaths per Year in {selected_state}")
fig_deaths, ax_deaths = plt.subplots()
ax_deaths.plot(yearly_data['year'], yearly_data['deaths_new'], marker='o', color='red')
ax_deaths.set_xlabel("Year")
ax_deaths.set_ylabel("Deaths")
ax_deaths.set_title("Deaths Over the Years")
st.pyplot(fig_deaths)
