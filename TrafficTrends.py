#Identifying trends in traffic patterns to reduce congestion using publicly available data sets

# Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt #Creation of visulization
import seaborn as sns
from sklearn.cluster import KMeans
from statsmodels.tsa.seasonal import seasonal_decompose #
from statsmodels.tsa.arima.model import ARIMA   #
import geopandas as gpd  # For spati3al data

#Load Datasets
def load_data():
    traffic_data = pd.read_csv("traffic_data.csv")
    weather_data = pd.read_csv("weather_data.csv")
    accident_data = pd.read_csv("accident_data.csv")
    return traffic_data, weather_data, accident_data

#Preprocess Data
def preprocess_data(traffic_data, weather_data, accident_data):
    # to_datetime converts pandas date and time format
    traffic_data['timestamp'] = pd.to_datetime(traffic_data['timestamp'])
    weather_data['timestamp'] = pd.to_datetime(weather_data['timestamp'])
    accident_data['timestamp'] = pd.to_datetime(accident_data['timestamp'])
    #combines 2 dataframe with common columns or indecies
    merged_data = pd.merge(traffic_data, weather_data, on='timestamp', how='left')
    merged_data = pd.merge(merged_data, accident_data, on='timestamp', how='left')
    #all matching rows of left dataframe to right dataframe(how)

    # Handle missing values
    merged_data.fillna(0, inplace=True)
    #0 replaces all NaN values and inplace is used to modify the original data frame or not if true modify else no
    return merged_data

#Analyze Traffic Trends
def analyze_traffic(merged_data):
    # Aggregate traffic counts by hour and day
    # Extracting hour from timestamp column
    merged_data['hour'] = merged_data['timestamp'].dt.hour
    # Extracts days form timestamp column
    merged_data['day_of_week'] = merged_data['timestamp'].dt.day_name()

    hourly_traffic = merged_data.groupby('hour')['traffic_count'].mean()
    daily_traffic = merged_data.groupby('day_of_week')['traffic_count'].mean()

    return hourly_traffic, daily_traffic

#Visualize Trends
def visualize_trends(hourly_traffic, daily_traffic):
    # Hourly Traffic Trends
    plt.figure(figsize=(10, 6))
    plt.plot(hourly_traffic, marker='o')    # Used to generate line plot
    plt.title("Hourly Traffic Trends")      
    plt.xlabel("Hour of Day")
    plt.ylabel("Average Traffic Count")
    plt.grid(True)
    plt.show()

    # Daily Traffic Trends
    plt.figure(figsize=(10, 6))
    sns.barplot(x=daily_traffic.index, y=daily_traffic.values, palette="viridis")   # Used to generate bar graph
    plt.title("Daily Traffic Trends")
    plt.xlabel("Day of Week")
    plt.ylabel("Average Traffic Count")
    plt.xticks(rotation=45)
    plt.show()

#Seasonal Decomposition
def decompose_time_series(merged_data):
    # Decompose traffic time series
    traffic_series = merged_data.set_index('timestamp')['traffic_count']
    decomposition = seasonal_decompose(traffic_series, model='additive', period=24)
    decomposition.plot()
    plt.show()

# Step 6: Clustering Congestion Hotspots
def cluster_congestion(merged_data):
    # Assume coordinates for clustering
    coords = merged_data[['latitude', 'longitude']]
    kmeans = KMeans(n_clusters=5, random_state=42).fit(coords)
    merged_data['cluster'] = kmeans.labels_
    
    # Plot clusters on a map
    gdf = gpd.GeoDataFrame(merged_data, geometry=gpd.points_from_xy(merged_data.longitude, merged_data.latitude))
    gdf.plot(column='cluster', cmap='viridis', legend=True, figsize=(10, 6))
    plt.title("Congestion Hotspots")
    plt.show()

# Step 7: Predict Traffic Patterns
def predict_traffic(merged_data):
    # Fit ARIMA model on traffic time series
    traffic_series = merged_data.set_index('timestamp')['traffic_count']
    model = ARIMA(traffic_series, order=(1, 1, 1))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=24)
    plt.figure(figsize=(10, 6))
    plt.plot(traffic_series[-100:], label="Historical Data")
    plt.plot(forecast, label="Forecast", color='red')
    plt.title("Traffic Pattern Forecast")
    plt.legend()
    plt.show()

# Main Function
if __name__ == "__main__":
    # Load and preprocess data
    traffic_data, weather_data, accident_data = load_data()
    merged_data = preprocess_data(traffic_data, weather_data, accident_data)
   # Analyze trends
    hourly_traffic, daily_traffic = analyze_traffic(merged_data)
    visualize_trends(hourly_traffic, daily_traffic)
    
    # Decompose time series
    decompose_time_series(merged_data)
    
    # Cluster congestion hotspots
    cluster_congestion(merged_data)
    
    # Predict future traffic patterns
    predict_traffic(merged_data)
