import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import requests
from geopy.geocoders import Nominatim

USE_API = "geopy"  
API_KEY = "YOUR_OPENWEATHER_API_KEY"  

# Function to get latitude and longitude
def get_lat_lon(country):
    if USE_API == "geopy":
        geolocator = Nominatim(user_agent="geoapi")
        location = geolocator.geocode(country)
        if location:
            return location.latitude, location.longitude
        else:
            raise ValueError(f"Invalid country name: {country}. Please check spelling.")
    elif USE_API == "openweather":
        url = f"https://api.openweathermap.org/geo/1.0/direct?q={country}&limit=1&appid={API_KEY}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if data:
                return data[0]["lat"], data[0]["lon"]
            else:
                raise ValueError(f"No data found for country: {country}. Please check spelling.")
        else:
            raise ValueError(f"API request failed with status code {response.status_code}. Check API key or network.")

# Function to generate climate data
def load_data():
    print("Loading climate data...")
    dates = pd.date_range(start='1929-01-01', end='2025-02-12', freq='D')
    temperatures = np.random.normal(15, 5, size=len(dates)) + np.linspace(0, 2, len(dates))
    rainfalls = np.random.normal(50, 20, size=len(dates))
    cyclones = np.random.choice([0, 1], size=len(dates), p=[0.98, 0.02])
    tsunamis = np.random.choice([0, 1], size=len(dates), p=[0.99, 0.01])
    return pd.DataFrame({'Date': dates, 'Temperature': temperatures, 'Rainfall': rainfalls, 'Cyclone': cyclones, 'Tsunami': tsunamis})

# Preprocessing function
def preprocess_data(df):
    print("Preprocessing data...")
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    return df

# Data visualization
def visualize_data(df):
    print("Visualizing data...")
    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['Temperature'], label='Historical Temperature')
    plt.xlabel('Year')
    plt.ylabel('Temperature (°C)')
    plt.title('Global Temperature Trend')
    plt.legend()
    plt.show()

# Prepare data for ML
def prepare_for_ml(df):
    print("Preparing data for machine learning...")
    X = df[['Year', 'Month', 'Day']]
    y_temp = df['Temperature']
    y_rain = df['Rainfall']
    y_cyclone = df['Cyclone']
    y_tsunami = df['Tsunami']
    return train_test_split(X, y_temp, test_size=0.2, random_state=42), train_test_split(X, y_rain, test_size=0.2, random_state=42), train_test_split(X, y_cyclone, test_size=0.2, random_state=42), train_test_split(X, y_tsunami, test_size=0.2, random_state=42)

# Train model
def train_model(X_train, y_train):
    print("Training the model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Predict future climate
def predict_future(model, years_to_predict=10):
    print("Making future predictions...")
    last_year = 2023
    future_dates = pd.date_range(start=f'{last_year+1}-01-01', periods=years_to_predict * 365, freq='D')
    future_data = pd.DataFrame({'Year': future_dates.year, 'Month': future_dates.month, 'Day': future_dates.day})
    predictions = model.predict(future_data)
    return pd.DataFrame({'Date': future_dates, 'Prediction': predictions})

# Evaluate conditions
def evaluate_future_conditions(future_df, threshold, metric):
    print(f"Evaluating future {metric} conditions...")
    future_df['Condition'] = future_df['Prediction'].apply(lambda x: 'Bad' if x > threshold else 'Good')
    return future_df

# Plot predictions
def plot_predictions(future_df, metric):
    print(f"Visualizing future {metric} predictions...")
    plt.figure(figsize=(12, 6))
    plt.bar(future_df['Date'], future_df['Prediction'], alpha=0.7, label=f'Predicted {metric}')
    plt.xlabel('Year')
    plt.ylabel(metric)
    plt.title(f'Predicted {metric} Trend')
    plt.legend()
    plt.show()

# Notify severe weather
def notify_severe_weather(future_df):
    print("Checking for severe weather conditions...")
    severe_conditions = future_df[future_df['Condition'] == 'Bad']
    if not severe_conditions.empty:
        alert_dates = severe_conditions['Date'].iloc[:3]
        print(f"Severe weather expected on: {alert_dates.tolist()} - Take Precautions!")

# Main execution
if __name__ == "__main__":
    try:
        country = input("Enter country name: ")
        lat, lon = get_lat_lon(country)
        print(f"Coordinates for {country}: Latitude {lat}, Longitude {lon}")
    except ValueError as e:
        print(e)
        exit()

    df = load_data()
    df = preprocess_data(df)
    visualize_data(df)
    
    (X_train_temp, X_test_temp, y_train_temp, y_test_temp), (X_train_rain, X_test_rain, y_train_rain, y_test_rain), (X_train_cyclone, X_test_cyclone, y_train_cyclone, y_test_cyclone), (X_train_tsunami, X_test_tsunami, y_train_tsunami, y_test_tsunami) = prepare_for_ml(df)
    
    temp_model = train_model(X_train_temp, y_train_temp)
    rain_model = train_model(X_train_rain, y_train_rain)
    cyclone_model = train_model(X_train_cyclone, y_train_cyclone)
    tsunami_model = train_model(X_train_tsunami, y_train_tsunami)
    
    future_temp = predict_future(temp_model, years_to_predict=10)
    future_rain = predict_future(rain_model, years_to_predict=10)
    future_cyclone = predict_future(cyclone_model, years_to_predict=10)
    future_tsunami = predict_future(tsunami_model, years_to_predict=10)
    
    future_temp = evaluate_future_conditions(future_temp, threshold=18, metric='Temperature')
    future_rain = evaluate_future_conditions(future_rain, threshold=80, metric='Rainfall')
    future_cyclone = evaluate_future_conditions(future_cyclone, threshold=0.5, metric='Cyclone')
    future_tsunami = evaluate_future_conditions(future_tsunami, threshold=0.2, metric='Tsunami')
    
    plot_predictions(future_temp, 'Temperature')
    plot_predictions(future_rain, 'Rainfall')
    plot_predictions(future_cyclone, 'Cyclone')
    plot_predictions(future_tsunami, 'Tsunami')
    
    notify_severe_weather(future_temp)
    notify_severe_weather(future_rain)
    notify_severe_weather(future_cyclone)
    notify_severe_weather(future_tsunami)
    
    print("Analysis and prediction completed.")
