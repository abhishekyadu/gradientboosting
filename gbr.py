import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# Load dataset
data = pd.read_csv('solarpowergeneration.csv')

# Sidebar for navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Go to", ["EDA", "Model Training", "Prediction"])

# EDA Section
if options == "EDA":
    st.title("Exploratory Data Analysis")
    
    st.write("### Dataset Preview")
    st.dataframe(data.head())

    st.write("### Basic Statistics")
    st.write(data.describe())

    st.write("### Correlation Matrix")
    st.write(data.corr())

# Model Training Section
elif options == "Model Training":
    st.title("Model Training")
    
    # Feature selection
    X = data.drop("power-generated", axis=1)
    y = data["power-generated"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # RandomForestRegressor model
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save the model
    with open('gbr.pkl', 'wb') as file:
        pickle.dump(model, file)

    # Predictions and evaluation
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    st.write("### Model Performance")
    st.write(f"Mean Squared Error: {mse}")
    st.write(f"R^2 Score: {r2}")

    st.success("Model training completed and saved successfully.")

# Prediction Section
elif options == "Prediction":
    st.title("Make a Prediction")
    
    # Load the trained model
    with open('gbr.pkl', 'rb') as file:
        model = pickle.load(file)
    
    st.write("### Input Environmental Variables")

    # User inputs for prediction
    distance_to_solar_noon = st.number_input("Distance to Solar Noon", min_value=0.0, max_value=1.0, value=0.5)
    temperature = st.number_input("Temperature", min_value=-50, max_value=150, value=70)
    wind_direction = st.number_input("Wind Direction", min_value=0, max_value=360, value=180)
    wind_speed = st.number_input("Wind Speed", min_value=0.0, max_value=100.0, value=5.0)
    sky_cover = st.number_input("Sky Cover", min_value=0, max_value=100, value=0)
    visibility = st.number_input("Visibility", min_value=0.0, max_value=100.0, value=10.0)
    humidity = st.number_input("Humidity", min_value=0, max_value=100, value=50)
    average_wind_speed = st.number_input("Average Wind Speed (Period)", min_value=0.0, max_value=100.0, value=5.0)
    average_pressure = st.number_input("Average Pressure (Period)", min_value=20.0, max_value=40.0, value=30.0)

    # Make prediction
    prediction = model.predict([[distance_to_solar_noon, temperature, wind_direction, wind_speed,
                                 sky_cover, visibility, humidity, average_wind_speed, average_pressure]])

    st.write(f"### Predicted Power Generated: {prediction[0]:.2f} watts")
