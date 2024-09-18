import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Title of the app
st.title('Energy Consumption Prediction')

# Load the data
data = pd.read_csv('Consumption.csv')
st.write("Sample Data:")
st.write(data)

# Prepare features and target variable
data = pd.get_dummies(data, columns=['season', 'heating_system', 'region'], drop_first=True)
X = data.drop('energy_consumption', axis=1)
y = data['energy_consumption']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Display model parameters
st.write(f"Model Coefficients: {model.coef_}")
st.write(f"Intercept: {model.intercept_}")

# Make predictions
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
st.write(f"Mean Squared Error: {mse}")

# Visualize predictions vs true values
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, color='blue')
ax.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
ax.set_xlabel('True Energy Consumption (kWh)')
ax.set_ylabel('Predicted Energy Consumption (kWh)')
ax.set_title('Predictions vs True Values')
st.pyplot(fig)

# User input for prediction
st.header('Input House Details for Prediction')

num_residents_input = st.number_input('Number of Residents', min_value=1, value=3, key='num_residents')
house_size_input = st.number_input('House Size (sq ft)', min_value=0, value=1500, key='house_size')
avg_temperature_input = st.number_input('Average Temperature (°C)', min_value=-30, value=22, key='avg_temperature')
num_appliances_input = st.number_input('Number of Appliances', min_value=0, value=10, key='num_appliances')
energy_efficiency_rating_input = st.selectbox('Energy Efficiency Rating (1-5)', options=[1, 2, 3, 4, 5], key='energy_efficiency')
insulation_quality_input = st.selectbox('Insulation Quality (1-5)', options=[1, 2, 3, 4, 5], key='insulation_quality')
num_floors_input = st.number_input('Number of Floors', min_value=1, value=1, key='num_floors')
monthly_income_input = st.number_input('Monthly Income ($)', min_value=0, value=3000, key='monthly_income')

# Get season input
season_input = st.selectbox('Season', options=['Spring', 'Summer', 'Fall', 'Winter'], key='season')

# Get heating system input
heating_system_input = st.selectbox('Type of Heating System', options=['Gas', 'Electric', 'None'], key='heating_system')

# Get region input
region_input = st.selectbox('Region', options=['Urban', 'Suburban', 'Rural'], key='region')

# Prepare input data for prediction
input_dict = {
    'num_residents': num_residents_input,
    'house_size': house_size_input,
    'avg_temperature': avg_temperature_input,
    'num_appliances': num_appliances_input,
    'energy_efficiency_rating': energy_efficiency_rating_input,
    'insulation_quality': insulation_quality_input,
    'num_floors': num_floors_input,
    'monthly_income': monthly_income_input,
}

# Create one-hot encoded features for categorical inputs
season_encoded = pd.get_dummies([season_input], prefix='season', drop_first=True)
heating_system_encoded = pd.get_dummies([heating_system_input], prefix='heating_system', drop_first=True)
region_encoded = pd.get_dummies([region_input], prefix='region', drop_first=True)

# Combine all input features into a single DataFrame
input_df = pd.DataFrame({
    **input_dict,
    **season_encoded.iloc[0].to_dict(),
    **heating_system_encoded.iloc[0].to_dict(),
    **region_encoded.iloc[0].to_dict()
}, index=[0])  # Add index to create a DataFrame with a single row

# Ensure the input data matches the model's expected input
input_df = input_df.reindex(columns=X.columns, fill_value=0)

# Make the prediction
predicted_energy_consumption = model.predict(input_df)

# Display the prediction result
st.write(f'Predicted Energy Consumption: {predicted_energy_consumption[0]:.2f} kWh')
