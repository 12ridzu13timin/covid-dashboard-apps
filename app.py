import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load and clean data
df = pd.read_csv("data,covid.csv")
df.columns = df.columns.str.strip()
df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y')
df['year'] = df['date'].dt.year

# Ensure numeric columns
for col in ['cases_new', 'deaths_new']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna(subset=['cases_new', 'deaths_new'])

# Sidebar selection
st.sidebar.title("Filters")
states = df['state'].unique()
selected_state = st.sidebar.selectbox("Select a state", states)

# Filtered state data
state_df = df[df['state'] == selected_state]

yearly_data = state_df.groupby('year').agg({
    'cases_new': 'sum',
    'deaths_new': 'sum'
}).reset_index()

# Forecasting cases
X = yearly_data[['year']]
y = yearly_data['cases_new']
model = LinearRegression()
model.fit(X, y)
future_years = pd.DataFrame({'year': list(range(yearly_data['year'].max()+1, yearly_data['year'].max()+6))})
future_cases = model.predict(future_years)

# Main dashboard
st.title("COVID-19 Dashboard with Forecasting")
st.subheader(f"State: {selected_state}")

# Line chart for actual cases
st.line_chart(yearly_data.set_index('year')['cases_new'])

# Forecast chart
plt.figure()
plt.plot(X, y, label="Actual")
plt.plot(future_years, future_cases, '--', label="Forecast")
plt.xlabel("Year")
plt.ylabel("New Cases")
plt.title("Forecasted New Cases")
plt.legend()
st.pyplot(plt)

# Pie chart: % of cases by state for a selected year
years = sorted(df['year'].unique())
selected_year = st.sidebar.selectbox("Select a year for state case distribution", years)
yearly_state = df[df['year'] == selected_year].groupby('state')['cases_new'].sum()

st.subheader(f"Case Distribution by State in {selected_year}")
fig1, ax1 = plt.subplots()
ax1.pie(yearly_state, labels=yearly_state.index, autopct='%1.1f%%', startangle=90)
ax1.axis('equal')
st.pyplot(fig1)

# Line chart: deaths vs. year
st.subheader(f"Deaths per Year in {selected_state}")
plt.figure()
plt.plot(yearly_data['year'], yearly_data['deaths_new'], marker='o', color='red')
plt.xlabel("Year")
plt.ylabel("Deaths")
plt.title("Deaths Over the Years")
st.pyplot(plt)
