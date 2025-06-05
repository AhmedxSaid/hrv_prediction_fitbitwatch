import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os
from datetime import datetime, timedelta

def analyze_hrv_data(target_date=None):
    # Read the combined data
    df = pd.read_csv('data/processed/combined_data_1min.csv', delimiter=';')
    time_column = 'minutes__minute'
    if time_column not in df.columns:
        raise ValueError(f"Time column '{time_column}' not found in data")
    # Convert the time column to datetime objects, specifying the correct format
    df[time_column] = pd.to_datetime(df[time_column], format='%d/%m/%Y %H:%M')
    
    # Determine the target variable
    target = 'night_hrv' if 'night_hrv' in df.columns else 'minutes__value__rmssd'

    # Filter for nighttime data
    if 'night_hrv' in df.columns:
        # Use actual HRV values for nighttime (assuming night_hrv > 0 indicates nighttime data)
        df_night = df[df['night_hrv'] > 0].copy()
    else:
        # Fallback to time-based filtering if night_hrv column is not available
        df['hour'] = df['minutes__minute'].dt.hour
        df_night = df[((df['hour'] >= 22) | (df['hour'] < 6))].copy()

    print(f"\nTotal data points: {len(df)}")
    print(f"Nighttime data points: {len(df_night)}")

    # If no target date provided, use the first day with nighttime data
    if target_date is None:
        if not df_night.empty:
            target_date = df_night['minutes__minute'].dt.date.min()
        else:
            print("No nighttime data available to select a target date.")
            return # Exit if no nighttime data
    else:
        target_date = pd.to_datetime(target_date).date()
    
    # Filter data for the target date within the nighttime data
    df_night_day = df_night[df_night['minutes__minute'].dt.date == target_date].copy()
    
    # Check if there is data for the target date
    if df_night_day.empty:
        print(f"No nighttime data available for the target date: {target_date}")
        return # Exit if no data for the target date

    # Feature descriptions (Keeping relevant ones, added some based on the attached file content)
    feature_descriptions = {
        'minutes__value__rmssd': 'Current HRV (RMSSD) value',
        'prev_day_hrv_mean': "Previous day's average HRV",
        'prev_day_hrv_std': "Previous day's HRV standard deviation",
        'minutes__value': 'Current heart rate',
        'hr_diff_1min': '1-minute heart rate difference',
        'hr_diff_5min': '5-minute heart rate difference',
        'hr_rolling_mean_5min': '5-minute rolling average heart rate',
        'hr_rolling_std_5min': '5-minute heart rate standard deviation', # Added based on attached file
        'hr_rolling_mean_15min': '15-minute rolling average heart rate',
        'hr_rolling_std_15min': '15-minute heart rate standard deviation', # Added based on attached file
        'hr_rolling_mean_30min': '30-minute rolling average heart rate',
        'hr_rolling_std_30min': '30-minute heart rate standard deviation', # Added based on attached file
        'hr_rolling_mean_60min': '60-minute rolling average heart rate',
        'hr_rolling_std_60min': '60-minute heart rate standard deviation', # Added based on attached file
        'hr_t trend': 'Heart rate trend over 30 minutes',
        'hr_stability': 'Heart rate stability over 30 minutes',
        'hour': 'Hour of day', # Keep hour and minute as potential features even in nighttime data
        'minute': 'Minute of hour', # Keep hour and minute as potential features even in nighttime data
        'deep_sleep_minutes': "Previous night's deep sleep duration",
        'light_sleep_minutes': "Previous night's light sleep duration",
        'rem_sleep_minutes': "Previous night's REM sleep duration",
        'wake_minutes': "Previous night's wake duration",
        'sleep_efficiency': "Previous night's sleep efficiency",
        'sleep_duration': "Previous night's sleep duration",
        'active_minutes': "Previous day's active minutes", # Keep as they might influence recovery
        'steps': "Previous day's step count", # Keep as they might influence recovery
        'activity_calories': "Previous day's activity calories", # Keep as they might influence recovery
        'breathing_rate_avg': 'Current breathing rate', # Added based on attached file
        'skin_temperature': 'Current skin temperature' # Added based on attached file
    }
    
    # Print feature descriptions
    print("\nFeature Descriptions:")
    for feature, description in feature_descriptions.items():
        print(f"{feature}: {description}")

    # Prepare data for modeling
    # Select only numeric columns for features
    numeric_columns = df_night.select_dtypes(include=[np.number]).columns
    # Exclude the target variable, 'id', and 'minutes__value__rmssd' from features
    features_to_exclude = ['id', target, 'minutes__value__rmssd']
    features = [col for col in numeric_columns if col not in features_to_exclude]
    
    # Handle missing values
    print("\nMissing values before imputation:")
    print(df_night[features + [target]].isnull().sum()) # Include target in missing value check
    
    # Impute missing values with median for numeric columns only
    for col in features + [target]:
         if col in df_night.columns: # Added check if column exists
            df_night[col] = df_night[col].fillna(df_night[col].median())
    
    # Drop rows where target is still missing after imputation
    df_night = df_night.dropna(subset=[target])
    
    print("\nMissing values after imputation:")
    print(df_night[features + [target]].isnull().sum()) # Include target in missing value check

    # Check if there are enough data points after handling missing values
    if len(df_night) < 2: # Need at least 2 data points for train/test split
        print("Not enough data points for modeling after handling missing values.")
        return # Exit if not enough data

    # Split data into training and testing sets
    # Ensure X and y are created from the filtered and imputed df_night
    X = df_night[features]
    y = df_night[target]
    
    # Adjust test_size if the dataset is very small
    test_size_val = 0.2 if len(df_night) * 0.2 >= 1 else (1 if len(df_night) > 1 else 0)
    if test_size_val == 0:
         print("Not enough data points for train/test split.")
         return # Exit if not enough data for split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size_val, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train a Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\nModel Performance (Nighttime Only):")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R² Score: {r2:.2f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features (Nighttime Only):")
    print(feature_importance.head(10))
    
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Plot feature importance
    plt.figure(figsize=(12, 6))
    sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
    plt.title('Top 10 Most Important Features (Nighttime Only)')
    plt.tight_layout()
    plt.savefig('plots/feature_importance_night.png')
    plt.close()
    
    # Plot actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual ' + target)
    plt.ylabel('Predicted ' + target)
    plt.title('Actual vs Predicted ' + target + ' Values (Nighttime Only)')
    plt.tight_layout()
    plt.savefig('plots/actual_vs_predicted_night.png')
    plt.close()
    
    # Plot residuals
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted ' + target)
    plt.ylabel('Residuals')
    plt.title('Residual Plot (Nighttime Only)')
    plt.tight_layout()
    plt.savefig('plots/residuals_night.png')
    plt.close()
    
    # Plot HRV over time for the selected night
    plt.figure(figsize=(15, 6))
    plt.plot(df_night_day['minutes__minute'], df_night_day[target], label='Actual ' + target)
    plt.xlabel('Time')
    plt.ylabel(target + ' (RMSSD)')
    plt.title(f'{target} Over Time - {target_date} (Nighttime Only)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/hrv_over_time_night.png')
    plt.close()
    
    # Plot hourly HRV statistics for the selected night
    # Ensure the column exists before grouping
    if target in df_night_day.columns:
        hourly_stats = df_night_day.groupby(df_night_day['minutes__minute'].dt.hour)[target].agg(['mean', 'std']).reset_index()
        
        plt.figure(figsize=(15, 6))
        plt.plot(hourly_stats['minutes__minute'], hourly_stats['mean'], label='Hourly Mean ' + target)
        # Check if std is not NaN before plotting fill_between
        if 'std' in hourly_stats.columns and not hourly_stats['std'].isnull().all():
            plt.fill_between(hourly_stats['minutes__minute'], 
                            hourly_stats['mean'] - hourly_stats['std'],
                            hourly_stats['mean'] + hourly_stats['std'],
                            alpha=0.2)
        plt.xlabel('Hour of Night')
        plt.ylabel(target + ' (RMSSD)')
        plt.title(f'Hourly {target} Statistics - {target_date} (Nighttime Only)')
        plt.legend()
        plt.tight_layout()
        plt.savefig('plots/hourly_hrv_stats_night.png')
        plt.close()
    else:
        print(f"Target column '{target}' not found in df_night_day. Cannot plot hourly stats.")

if __name__ == "__main__":
    # You can specify a target date in YYYY-MM-DD format
    analyze_hrv_data()  # This will use the first night with data