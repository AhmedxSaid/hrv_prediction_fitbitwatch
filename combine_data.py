import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Read the data files
def read_data():
    # Read HRV data (5-minute intervals)
    hrv_df = pd.read_csv('data/fitbit_data_ahmed_said/csv files/combined_data/hrv_data.csv', sep=';')
    hrv_df['minutes__minute'] = pd.to_datetime(hrv_df['minutes__minute'])
    
    # Read heart rate data (1-minute intervals)
    hr_df = pd.read_csv('data/fitbit_data_ahmed_said/csv files/combined_data/heart_rate_data.csv', sep=';')
    
    # Read sleep data with proper encoding
    sleep_df = pd.read_csv('data/fitbit_data_ahmed_said/csv files/combined_data/sleep_data.csv', sep=';', encoding='utf-8-sig')
    print("Original sleep data columns:", sleep_df.columns.tolist())
    
    # Clean up column names - handle BOM and quotes
    sleep_df.columns = [col.encode('utf-8').decode('utf-8-sig').strip().strip('"') for col in sleep_df.columns]
    print("Cleaned sleep data columns:", sleep_df.columns.tolist())
    
    # Convert dateOfSleep to datetime
    sleep_df['dateOfSleep'] = pd.to_datetime(sleep_df['dateOfSleep'], format='%d/%m/%Y', errors='coerce')
    # Convert startTime and endTime to datetime, handling missing values
    sleep_df['startTime'] = pd.to_datetime(sleep_df['startTime'], format='%Y-%m-%dT%H:%M:%S.%f', errors='coerce')
    sleep_df['endTime'] = pd.to_datetime(sleep_df['endTime'], format='%Y-%m-%dT%H:%M:%S.%f', errors='coerce')
    
    # Read activity data
    activity_df = pd.read_csv('data/fitbit_data_ahmed_said/csv files/combined_data/activity_data.csv', sep=';', encoding='utf-8-sig')
    # Clean up column names
    activity_df.columns = [col.encode('utf-8').decode('utf-8-sig').strip().strip('"') for col in activity_df.columns]
    # Convert startDate to datetime
    activity_df['activities__startDate'] = pd.to_datetime(activity_df['activities__startDate'], format='%d/%m/%Y', errors='coerce')
    
    # Read breathing rate data
    breathing_df = pd.read_csv('data/fitbit_data_ahmed_said/csv files/combined_data/breating_rate_data.csv', sep=';', encoding='utf-8-sig')
    breathing_df.columns = [col.encode('utf-8').decode('utf-8-sig').strip().strip('"') for col in breathing_df.columns]
    # Print sample of breathing data for debugging
    print("\nSample of breathing data:")
    print(breathing_df.head())
    print("\nBreathing data columns:", breathing_df.columns.tolist())
    # Convert dateTime to datetime using the correct format
    breathing_df['dateTime'] = pd.to_datetime(breathing_df['dateTime'], format='%d/%m/%Y', errors='coerce')
    
    # Read skin temperature data
    temp_df = pd.read_csv('data/fitbit_data_ahmed_said/csv files/combined_data/skintemp_data.csv', sep=';', encoding='utf-8-sig')
    temp_df.columns = [col.encode('utf-8').decode('utf-8-sig').strip().strip('"') for col in temp_df.columns]
    # Print sample of temperature data for debugging
    print("\nSample of temperature data:")
    print(temp_df.head())
    print("\nTemperature data columns:", temp_df.columns.tolist())
    # Convert dateTime to datetime using the correct format
    temp_df['dateTime'] = pd.to_datetime(temp_df['dateTime'], format='%d/%m/%Y', errors='coerce')
    
    # Process heart rate data
    hr_df['date'] = hr_df['activities-heart__dateTime'].ffill()
    hr_df['datetime_str'] = hr_df.apply(
        lambda row: f"{row['date']} {row['activities-heart-intraday__dataset__time']}" if pd.notna(row['activities-heart-intraday__dataset__time']) else None,
        axis=1
    )
    hr_df['minutes__minute'] = pd.to_datetime(hr_df['datetime_str'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
    hr_df = hr_df.dropna(subset=['minutes__minute'])
    hr_df = hr_df[['minutes__minute', 'activities-heart-intraday__dataset__value']]
    hr_df.columns = ['minutes__minute', 'minutes__value']
    # Convert heart rate values to numeric
    hr_df['minutes__value'] = pd.to_numeric(hr_df['minutes__value'], errors='coerce')
    
    return hrv_df, hr_df, sleep_df, activity_df, breathing_df, temp_df

def create_minute_intervals(start_time, end_time):
    """Create a DataFrame with 1-minute intervals between start and end time"""
    minutes = pd.date_range(start=start_time, end=end_time, freq='1min')
    return pd.DataFrame({'minutes__minute': minutes})

def forward_fill_data(df, time_col, value_cols):
    """Forward fill data for 5-minute measurements to 1-minute intervals"""
    # Create 1-minute intervals
    time_range = create_minute_intervals(df[time_col].min(), df[time_col].max())
    
    # Merge with original data
    merged = pd.merge_asof(time_range, df, on=time_col, direction='backward')
    
    # Keep only the specified columns
    columns_to_keep = [time_col] + value_cols
    merged = merged[columns_to_keep]
    
    return merged

def calculate_heart_rate_features(df):
    # Ensure heart rate values are numeric
    df['minutes__value'] = pd.to_numeric(df['minutes__value'], errors='coerce')
    
    # Heart rate differences
    df['hr_diff_1min'] = df['minutes__value'].diff()
    df['hr_diff_5min'] = df['minutes__value'].diff(5)
    
    # Rolling statistics
    for window in [5, 15, 30, 60]:
        df[f'hr_rolling_mean_{window}min'] = df['minutes__value'].rolling(window=window, min_periods=1).mean()
        df[f'hr_rolling_std_{window}min'] = df['minutes__value'].rolling(window=window, min_periods=1).std()
    
    # Heart rate trend and stability
    df['hr_trend'] = df['minutes__value'].rolling(window=30, min_periods=2).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else np.nan
    )
    df['hr_stability'] = df['minutes__value'].rolling(window=30, min_periods=1).std()
    
    return df

def calculate_hrv_features(df):
    # Calculate daily HRV statistics
    df['date'] = df['minutes__minute'].dt.date
    daily_hrv = df.groupby('date')['minutes__value__rmssd'].agg(['mean', 'std']).reset_index()
    daily_hrv.columns = ['date', 'daily_hrv_mean', 'daily_hrv_std']
    
    # Merge daily statistics back to the main dataframe
    df = pd.merge(df, daily_hrv, on='date', how='left')
    
    # Shift daily statistics to get previous day's values
    df['prev_day_hrv_mean'] = df.groupby('date')['daily_hrv_mean'].transform('first').shift(1)
    df['prev_day_hrv_std'] = df.groupby('date')['daily_hrv_std'].transform('first').shift(1)
    
    # Drop temporary columns
    df = df.drop(['date', 'daily_hrv_mean', 'daily_hrv_std'], axis=1)
    
    return df

def add_sleep_features(df, sleep_df):
    # Initialize sleep feature columns with NaN
    sleep_features = ['deep_sleep_minutes', 'light_sleep_minutes', 'rem_sleep_minutes', 
                     'wake_minutes', 'sleep_efficiency', 'sleep_duration']
    for feature in sleep_features:
        df[feature] = np.nan
    
    # Process sleep data
    for idx, row in sleep_df.iterrows():
        if pd.notna(row['startTime']) and pd.notna(row['endTime']) and pd.notna(row['dateOfSleep']):
            # Get the date for the next day (since sleep affects the next day)
            next_day = row['dateOfSleep'].date() + pd.Timedelta(days=1)
            mask = (df['minutes__minute'].dt.date == next_day)
            
            # Add basic sleep metrics
            df.loc[mask, 'sleep_efficiency'] = float(row['efficiency'])
            df.loc[mask, 'sleep_duration'] = float((row['endTime'] - row['startTime']).total_seconds() / 3600)
            
            # Process sleep stages from summary data first
            if pd.notna(row['levels__summary__|__minutes']):
                stage = row['levels__|'].lower() if pd.notna(row['levels__|']) else None
                minutes = float(row['levels__summary__|__minutes'])
                
                if stage == 'deep':
                    df.loc[mask, 'deep_sleep_minutes'] = minutes
                elif stage == 'light':
                    df.loc[mask, 'light_sleep_minutes'] = minutes
                elif stage == 'rem':
                    df.loc[mask, 'rem_sleep_minutes'] = minutes
                elif stage == 'wake':
                    df.loc[mask, 'wake_minutes'] = minutes
            
            # Process second level if it exists
            if pd.notna(row['levels__||__minutes']):
                stage = row['levels__||'].lower() if pd.notna(row['levels__||']) else None
                minutes = float(row['levels__||__minutes'])
                
                if stage == 'deep':
                    df.loc[mask, 'deep_sleep_minutes'] = minutes
                elif stage == 'light':
                    df.loc[mask, 'light_sleep_minutes'] = minutes
                elif stage == 'rem':
                    df.loc[mask, 'rem_sleep_minutes'] = minutes
                elif stage == 'wake':
                    df.loc[mask, 'wake_minutes'] = minutes
            
            # Process raw sleep stage data if summary data is missing
            if pd.notna(row['levels__data__level']) and pd.notna(row['levels__data__seconds']):
                # Convert seconds to minutes
                minutes = float(row['levels__data__seconds']) / 60
                stage = row['levels__data__level'].lower()
                
                # Only update if the summary data is missing
                if pd.isna(df.loc[mask, 'deep_sleep_minutes']).all() and stage == 'deep':
                    df.loc[mask, 'deep_sleep_minutes'] = minutes
                elif pd.isna(df.loc[mask, 'light_sleep_minutes']).all() and stage == 'light':
                    df.loc[mask, 'light_sleep_minutes'] = minutes
                elif pd.isna(df.loc[mask, 'rem_sleep_minutes']).all() and stage == 'rem':
                    df.loc[mask, 'rem_sleep_minutes'] = minutes
                elif pd.isna(df.loc[mask, 'wake_minutes']).all() and stage == 'wake':
                    df.loc[mask, 'wake_minutes'] = minutes
            
            # Use minutesAsleep and minutesAwake as a fallback
            if pd.notna(row['minutesAsleep']) and pd.notna(row['minutesAwake']):
                # If we still don't have sleep stage data, use the total minutes
                if pd.isna(df.loc[mask, 'deep_sleep_minutes']).all():
                    # Assume 20% of sleep time is deep sleep
                    df.loc[mask, 'deep_sleep_minutes'] = float(row['minutesAsleep']) * 0.2
                if pd.isna(df.loc[mask, 'light_sleep_minutes']).all():
                    # Assume 60% of sleep time is light sleep
                    df.loc[mask, 'light_sleep_minutes'] = float(row['minutesAsleep']) * 0.6
                if pd.isna(df.loc[mask, 'rem_sleep_minutes']).all():
                    # Assume 20% of sleep time is REM sleep
                    df.loc[mask, 'rem_sleep_minutes'] = float(row['minutesAsleep']) * 0.2
                if pd.isna(df.loc[mask, 'wake_minutes']).all():
                    df.loc[mask, 'wake_minutes'] = float(row['minutesAwake'])
    
    # Forward fill sleep features for the entire day
    for feature in sleep_features:
        df[feature] = df.groupby(df['minutes__minute'].dt.date)[feature].transform('first')
    
    return df

def add_activity_features(df, activity_df):
    # Initialize activity feature columns with NaN
    activity_features = ['active_minutes', 'steps', 'activity_calories']
    for feature in activity_features:
        df[feature] = np.nan
    
    # Convert activity date to datetime if it's not already
    activity_df['activities__startDate'] = pd.to_datetime(activity_df['activities__startDate'])
    
    # Add activity features from previous day
    for idx, row in activity_df.iterrows():
        if pd.notna(row['activities__startDate']):
            # Get the date for the next day (since activity affects the next day)
            next_day = row['activities__startDate'].date() + pd.Timedelta(days=1)
            mask = (df['minutes__minute'].dt.date == next_day)
            
            # Calculate total active minutes from different activity levels
            active_minutes = float(
                row['summary__lightlyActiveMinutes'] +
                row['summary__fairlyActiveMinutes'] +
                row['summary__veryActiveMinutes']
            )
            df.loc[mask, 'active_minutes'] = active_minutes
            df.loc[mask, 'steps'] = float(row['summary__steps'])
            df.loc[mask, 'activity_calories'] = float(row['summary__activityCalories'])
    
    # Forward fill activity features for the entire day
    for feature in activity_features:
        df[feature] = df.groupby(df['minutes__minute'].dt.date)[feature].transform('first')
    
    return df

def add_physiological_features(df, breathing_df, temp_df):
    # Initialize physiological feature columns
    df['breathing_rate_avg'] = np.nan
    df['skin_temperature'] = np.nan
    
    # Print some diagnostic information
    print("\nPhysiological data date ranges:")
    print(f"Breathing data: {breathing_df['dateTime'].min()} to {breathing_df['dateTime'].max()}")
    print(f"Temperature data: {temp_df['dateTime'].min()} to {temp_df['dateTime'].max()}")
    
    # Add breathing rate and skin temperature from previous night
    for idx, row in breathing_df.iterrows():
        if pd.notna(row['dateTime']):
            # Get the date for the next day (since these measurements affect the next day)
            next_day = row['dateTime'].date() + pd.Timedelta(days=1)
            mask = (df['minutes__minute'].dt.date == next_day)
            if mask.any():  # Only assign if we have matching dates
                # Use the breathing rate value
                df.loc[mask, 'breathing_rate_avg'] = float(row['value__|__breathingRate'])
    
    for idx, row in temp_df.iterrows():
        if pd.notna(row['dateTime']):
            # Get the date for the next day
            next_day = row['dateTime'].date() + pd.Timedelta(days=1)
            mask = (df['minutes__minute'].dt.date == next_day)
            if mask.any():  # Only assign if we have matching dates
                # Use the nightly relative temperature value
                df.loc[mask, 'skin_temperature'] = float(row['value__nightlyRelative'])
    
    # Forward fill physiological features for the entire day
    df['breathing_rate_avg'] = df.groupby(df['minutes__minute'].dt.date)['breathing_rate_avg'].transform('first')
    df['skin_temperature'] = df.groupby(df['minutes__minute'].dt.date)['skin_temperature'].transform('first')
    
    return df

def combine_datasets():
    # Read all data files
    hrv_df, hr_df, sleep_df, activity_df, breathing_df, temp_df = read_data()
    
    # Forward fill HRV data to 1-minute intervals
    hrv_1min = forward_fill_data(hrv_df, 'minutes__minute', ['minutes__value__rmssd'])
    
    # Calculate heart rate features
    hr_features = calculate_heart_rate_features(hr_df)
    
    # Merge HRV and heart rate data
    combined = pd.merge(hr_features, hrv_1min, on='minutes__minute', how='left')
    
    # Remove hour and minute columns (keeping just the original timestamp)
    combined = combined.drop(['hour', 'minute'], axis=1, errors='ignore')
    
    # First identify the correct HRV column name
    hrv_col = [col for col in combined.columns if 'hrv' in col.lower() or 'rmssd' in col.lower()][0]
    
    # Create night_hrv column with actual HRV values during nighttime
    night_mask = (combined['minutes__minute'].dt.hour >= 23) | (combined['minutes__minute'].dt.hour < 7)
    combined['night_hrv'] = combined[hrv_col].where(night_mask, 0)
    
    # Reorder columns
    cols = [col for col in combined.columns if col != 'night_hrv'] + ['night_hrv']
    combined = combined[cols]
    
    # Save the final combined data
    combined.to_csv('data/processed/combined_data_1min.csv', index=False)
    
    # Add sleep features
    combined = add_sleep_features(combined, sleep_df)
    
    # Add activity features
    combined = add_activity_features(combined, activity_df)
    
    # Add breathing rate features
    def add_breathing_features(df, breathing_df):
        """
        Add breathing rate features to the combined dataframe
        """
        # Convert breathing_df dateTime to datetime if not already
        breathing_df['dateTime'] = pd.to_datetime(breathing_df['dateTime'], format='%d/%m/%Y', errors='coerce')
        
        # Create a date column for merging
        breathing_df['date'] = breathing_df['dateTime'].dt.date
        df['date'] = df['minutes__minute'].dt.date
        
        # Merge breathing data
        merged = pd.merge(df, breathing_df, on='date', how='left')
        
        # Clean up columns
        merged = merged.drop(['dateTime', 'date'], axis=1)
        
        # Forward fill breathing rate values
        merged['value__|__breathingRate'] = merged['value__|__breathingRate'].ffill()
        
        return merged
    
    # Add temperature features
    def add_temperature_features(df, temp_df):
        """
        Add skin temperature features to the combined dataframe
        """
        # Convert temp_df dateTime to datetime if not already
        temp_df['dateTime'] = pd.to_datetime(temp_df['dateTime'], format='%d/%m/%Y', errors='coerce')
        
        # Create a date column for merging
        temp_df['date'] = temp_df['dateTime'].dt.date
        df['date'] = df['minutes__minute'].dt.date
        
        # Merge temperature data
        merged = pd.merge(df, temp_df, on='date', how='left')
        
        # Clean up columns
        merged = merged.drop(['dateTime', 'date', 'logType'], axis=1)
        
        # Forward fill temperature values
        merged['value__nightlyRelative'] = merged['value__nightlyRelative'].ffill()
        
        return merged
    
    # Calculate HRV features
    combined = calculate_hrv_features(combined)
    
    return combined

if __name__ == "__main__":
    combined_data = combine_datasets()
    print("Data combined successfully!")
    print(f"Total rows: {len(combined_data)}")
    print("\nFirst few rows:")
    print(combined_data.head())
    print("\nColumns in the dataset:")
    print(combined_data.columns.tolist())