<<<<<<< HEAD
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta

def plot_predictions(actual_df, predicted_df, date_str, filename):
    """
    Helper function to plot actual vs predicted HRV for a given day.
    
    Args:
        actual_df (pd.DataFrame): DataFrame with actual HRV values (must have 'time' and 'actual' columns).
        predicted_df (pd.DataFrame): DataFrame with predicted HRV values (must have 'time' and 'predicted' columns).
        date_str (str): String representation of the date for the plot title.
        filename (str): The name of the file to save the plot to.
    """
    plt.figure(figsize=(15, 6)) # Set figure size
    # Plot actual HRV
    plt.plot(actual_df['time'], actual_df['actual'], 
             label='Actual HRV', color='blue')
    # Plot predicted HRV
    plt.plot(predicted_df['time'], predicted_df['predicted'], 
             label='Predicted HRV', color='red', linestyle='--')
    plt.xlabel('Time') # X-axis label
    plt.ylabel('HRV (RMSSD)') # Y-axis label
    plt.title(f'HRV Predictions vs Actual Values - {date_str}') # Plot title
    plt.legend() # Show legend
    plt.tight_layout() # Adjust layout
    plt.savefig(f'plots/ffs-30day-plots/{filename}') # Save the plot
    plt.close() # Close the figure to free up memory

def plot_residuals(actual_df, predicted_df, date_str, filename):
    """
    Helper function to plot residuals (actual - predicted) for a given day.
    
    Args:
        actual_df (pd.DataFrame): DataFrame with actual HRV values.
        predicted_df (pd.DataFrame): DataFrame with predicted HRV values.
        date_str (str): String representation of the date for the plot title.
        filename (str): The name of the file to save the plot to.
    """
    residuals = actual_df['actual'] - predicted_df['predicted'] # Calculate residuals
    plt.figure(figsize=(15, 6)) # Set figure size
    plt.scatter(predicted_df['time'], residuals, alpha=0.5) # Scatter plot of residuals over time
    plt.axhline(y=0, color='r', linestyle='--') # Add a horizontal line at y=0
    plt.xlabel('Time') # X-axis label
    plt.ylabel('Residuals (Actual - Predicted)') # Y-axis label
    plt.title(f'Residuals - {date_str}') # Plot title
    plt.tight_layout() # Adjust layout
    plt.savefig(f'plots/ffs-30day-plots/{filename}') # Save the plot
    plt.close() # Close the figure

def plot_all_predictions(all_predictions, method):
    """
    Helper function to plot all predictions vs actual values across all predicted days.
    
    Args:
        all_predictions (dict): Dictionary where keys are dates and values are dicts with 'time', 'actual', 'predicted'.
        method (str): Feature selection method used (e.g., 'FFS').
    """
    plt.figure(figsize=(20, 10)) # Set figure size
    
    # Sort predictions by date to ensure chronological plotting
    dates = sorted(all_predictions.keys())
    
    # Plot actual and predicted for each date
    for i, date in enumerate(dates):
        pred_data = all_predictions[date]
        # Plot actual HRV (lighter color)
        plt.plot(pred_data['time'], pred_data['actual'], 
                color='blue', alpha=0.3, label='Actual' if i == 0 else "") # Add label only for the first date
        # Plot predicted HRV (lighter color)
        plt.plot(pred_data['time'], pred_data['predicted'], 
                color='red', alpha=0.3, label='Predicted' if i == 0 else "") # Add label only for the first date
    
    plt.xlabel('Time') # X-axis label
    plt.ylabel('HRV (RMSSD)') # Y-axis label
    plt.title(f'All HRV Predictions vs Actual Values ({method})') # Plot title
    plt.legend() # Show legend
    plt.tight_layout() # Adjust layout
    plt.savefig(f'plots/ffs-30day-plots/all_predictions_{method}.png') # Save the plot
    plt.close() # Close the figure

def plot_all_residuals(all_predictions, method):
    """
    Helper function to plot all residuals across all predicted days.
    
    Args:
        all_predictions (dict): Dictionary where keys are dates and values are dicts with 'time', 'actual', 'predicted'.
        method (str): Feature selection method used (e.g., 'FFS').
    """
    plt.figure(figsize=(20, 10)) # Set figure size
    
    # Sort predictions by date
    dates = sorted(all_predictions.keys())
    
    # Plot residuals for each date as scattered points
    for i, date in enumerate(dates):
        pred_data = all_predictions[date]
        residuals = pred_data['actual'] - pred_data['predicted'] # Calculate residuals
        plt.scatter(pred_data['time'], residuals, alpha=0.3, 
                   label=date.strftime('%Y-%m-%d') if i == 0 else "") # Add label only for the first date
    
    plt.axhline(y=0, color='r', linestyle='--') # Add a horizontal line at y=0
    plt.xlabel('Time') # X-axis label
    plt.ylabel('Residuals (Actual - Predicted)') # Y-axis label
    plt.title(f'All Residuals ({method})') # Plot title
    plt.legend() # Show legend
    plt.tight_layout() # Adjust layout
    plt.savefig(f'plots/ffs-30day-plots/all_residuals_{method}.png') # Save the plot
    plt.close() # Close the figure

def plot_model_performance_metrics(all_predictions, method):
    """
    Plot comprehensive model performance metrics (R², RMSE, MAE, MAPE) over time.
    
    Args:
        all_predictions (dict): Dictionary where keys are dates and values are dicts with 'time', 'actual', 'predicted'.
        method (str): Feature selection method used (e.g., 'FFS').
    """
    if not all_predictions: # Check if there are any predictions
        print(f"No predictions available for {method} to plot performance metrics.")
        return
        
    plt.figure(figsize=(15, 10)) # Set figure size
    gs = plt.GridSpec(2, 2) # Create a 2x2 grid for subplots
    
    # Initialize lists to store metrics per day
    dates = []
    r2_scores = []
    rmse_scores = []
    mae_scores = []
    mape_scores = []
    
    # Calculate metrics for each predicted day
    for date, pred_data in all_predictions.items():
        actual = pred_data['actual']
        predicted = pred_data['predicted']
        
        # Calculate metrics using scikit-learn and numpy
        r2 = r2_score(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mae = np.mean(np.abs(actual - predicted))
        # Avoid division by zero for MAPE
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100 if np.mean(actual) != 0 else np.nan
        
        # Append metrics and date
        dates.append(date)
        r2_scores.append(r2)
        rmse_scores.append(rmse)
        mae_scores.append(mae)
        mape_scores.append(mape)
    
    # Plot R² Score over time
    ax1 = plt.subplot(gs[0, 0]) # Select top-left subplot
    ax1.plot(dates, r2_scores, marker='o') # Plot R² scores
    ax1.set_title('R² Score Over Time') # Subplot title
    ax1.set_xlabel('Date') # X-axis label
    ax1.set_ylabel('R² Score') # Y-axis label
    ax1.grid(True) # Show grid
    
    # Plot RMSE over time
    ax2 = plt.subplot(gs[0, 1]) # Select top-right subplot
    ax2.plot(dates, rmse_scores, marker='o', color='red') # Plot RMSE scores
    ax2.set_title('RMSE Over Time') # Subplot title
    ax2.set_xlabel('Date') # X-axis label
    ax2.set_ylabel('RMSE') # Y-axis label
    ax2.grid(True) # Show grid
    
    # Plot MAE over time
    ax3 = plt.subplot(gs[1, 0]) # Select bottom-left subplot
    ax3.plot(dates, mae_scores, marker='o', color='green') # Plot MAE scores
    ax3.set_title('MAE Over Time') # Subplot title
    ax3.set_xlabel('Date') # X-axis label
    ax3.set_ylabel('MAE') # Y-axis label
    ax3.grid(True) # Show grid
    
    # Plot MAPE over time
    ax4 = plt.subplot(gs[1, 1]) # Select bottom-right subplot
    ax4.plot(dates, mape_scores, marker='o', color='purple') # Plot MAPE scores
    ax4.set_title('MAPE Over Time') # Subplot title
    ax4.set_xlabel('Date') # X-axis label
    ax4.set_ylabel('MAPE (%)') # Y-axis label
    ax4.grid(True) # Show grid
    
    plt.tight_layout() # Adjust layout
    plt.savefig(f'plots/ffs-30day-plots/performance_metrics_{method}.png') # Save the plot
    plt.close() # Close the figure

def plot_prediction_error_analysis(all_predictions, method):
    """
    Plot detailed prediction error analysis, including scatter plots and distributions.
    
    Args:
        all_predictions (dict): Dictionary where keys are dates and values are dicts with 'time', 'actual', 'predicted'.
        method (str): Feature selection method used (e.g., 'FFS').
    """
    if not all_predictions: # Check if there are any predictions
        print(f"No predictions available for {method} to plot error analysis.")
        return
        
    plt.figure(figsize=(15, 10)) # Set figure size
    gs = plt.GridSpec(2, 2) # Create a 2x2 grid for subplots
    
    # Collect all actual, predicted, and error values from all days
    all_actual = []
    all_predicted = []
    all_errors = []
    
    for pred_data in all_predictions.values():
        all_actual.extend(pred_data['actual'])
        all_predicted.extend(pred_data['predicted'])
        all_errors.extend(pred_data['actual'] - pred_data['predicted']) # Calculate errors
    
    # Convert lists to numpy arrays for easier handling
    all_actual = np.array(all_actual)
    all_predicted = np.array(all_predicted)
    all_errors = np.array(all_errors)

    # Prediction vs Actual scatter plot
    ax1 = plt.subplot(gs[0, 0]) # Select top-left subplot
    ax1.scatter(all_actual, all_predicted, alpha=0.5) # Scatter plot
    # Add a line representing perfect prediction (actual == predicted)
    ax1.plot([min(all_actual), max(all_actual)], 
             [min(all_actual), max(all_actual)], 
             'r--', label='Perfect Prediction')
    ax1.set_title('Predicted vs Actual Values') # Subplot title
    ax1.set_xlabel('Actual HRV') # X-axis label
    ax1.set_ylabel('Predicted HRV') # Y-axis label
    ax1.legend() # Show legend
    ax1.grid(True) # Show grid
    
    # Error distribution
    ax2 = plt.subplot(gs[0, 1]) # Select top-right subplot
    sns.histplot(all_errors, kde=True, ax=ax2) # Histogram with KDE
    ax2.set_title('Error Distribution') # Subplot title
    ax2.set_xlabel('Prediction Error') # X-axis label
    ax2.set_ylabel('Frequency') # Y-axis label
    
    # Error vs Predicted value scatter plot
    ax3 = plt.subplot(gs[1, 0]) # Select bottom-left subplot
    ax3.scatter(all_predicted, all_errors, alpha=0.5) # Scatter plot
    ax3.axhline(y=0, color='r', linestyle='--') # Add a horizontal line at y=0
    ax3.set_title('Error vs Predicted Value') # Subplot title
    ax3.set_xlabel('Predicted HRV') # X-axis label
    ax3.set_ylabel('Prediction Error') # Y-axis label
    ax3.grid(True) # Show grid
    
    # Error vs Actual value scatter plot
    ax4 = plt.subplot(gs[1, 1]) # Select bottom-right subplot
    ax4.scatter(all_actual, all_errors, alpha=0.5) # Scatter plot
    ax4.axhline(y=0, color='r', linestyle='--') # Add a horizontal line at y=0
    ax4.set_title('Error vs Actual Value') # Subplot title
    ax4.set_xlabel('Actual HRV') # X-axis label
    ax4.set_ylabel('Prediction Error') # Y-axis label
    ax4.grid(True) # Show grid
    
    plt.tight_layout() # Adjust layout
    plt.savefig(f'plots/ffs-30day-plots/error_analysis_{method}.png') # Save the plot
    plt.close() # Close the figure

def plot_time_series_analysis(all_predictions, method):
    """
    Plot time series analysis of actual and predicted HRV values, including rolling means.
    
    Args:
        all_predictions (dict): Dictionary where keys are dates and values are dicts with 'time', 'actual', 'predicted'.
        method (str): Feature selection method used (e.g., 'FFS').
    """
    if not all_predictions: # Check if there are any predictions
        print(f"No predictions available for {method} to plot time series analysis.")
        return
        
    plt.figure(figsize=(15, 10)) # Set figure size
    gs = plt.GridSpec(2, 1, height_ratios=[2, 1]) # Create a 2x1 grid for subplots with specified height ratios
    
    # Collect all data points from all days
    all_times = []
    all_actual = []
    all_predicted = []
    
    for pred_data in all_predictions.values():
        all_times.extend(pred_data['time'])
        all_actual.extend(pred_data['actual'])
        all_predicted.extend(pred_data['predicted'])
    
    # Sort data chronologically by time
    sorted_indices = np.argsort(all_times)
    all_times = np.array(all_times)[sorted_indices]
    all_actual = np.array(all_actual)[sorted_indices]
    all_predicted = np.array(all_predicted)[sorted_indices]
    
    # Main time series plot (Actual vs Predicted)
    ax1 = plt.subplot(gs[0]) # Select the top subplot
    ax1.plot(all_times, all_actual, label='Actual HRV', alpha=0.5) # Plot actual HRV
    ax1.plot(all_times, all_predicted, label='Predicted HRV', alpha=0.5) # Plot predicted HRV
    ax1.set_title(f'HRV Time Series - {method}') # Subplot title
    ax1.set_xlabel('Time') # X-axis label
    ax1.set_ylabel('HRV (RMSSD)') # Y-axis label
    ax1.legend() # Show legend
    ax1.grid(True) # Show grid
    
    # Rolling statistics plot (Rolling Mean)
    window_size = 100  # Define the window size for rolling mean (adjust as needed)
    # Calculate rolling mean for actual and predicted values
    rolling_actual = pd.Series(all_actual).rolling(window=window_size).mean()
    rolling_predicted = pd.Series(all_predicted).rolling(window=window_size).mean()
    
    ax2 = plt.subplot(gs[1]) # Select the bottom subplot
    ax2.plot(all_times, rolling_actual, label='Actual HRV (Rolling Mean)', alpha=0.7) # Plot rolling mean of actual
    ax2.plot(all_times, rolling_predicted, label='Predicted HRV (Rolling Mean)', alpha=0.7) # Plot rolling mean of predicted
    ax2.set_title(f'Rolling Mean HRV (Window Size: {window_size})') # Subplot title
    ax2.set_xlabel('Time') # X-axis label
    ax2.set_ylabel('HRV (RMSSD)') # Y-axis label
    ax2.legend() # Show legend
    ax2.grid(True) # Show grid
    
    plt.tight_layout() # Adjust layout
    plt.savefig(f'plots/ffs-30day-plots/time_series_analysis_{method}.png') # Save the plot
    plt.close() # Close the figure

def forward_feature_selection(X_train, y_train, X_val, y_val, model):
    """
    Performs Iterative Forward Feature Selection based on R² score on a validation set.
    
    Starts with no features and adds the feature that provides the greatest improvement
    in R² score on the validation set at each step. Stops when adding no longer improves R².
    
    Args:
        X_train (pd.DataFrame): Training features data.
        y_train (pd.Series): Training target variable data.
        X_val (pd.DataFrame): Validation features data.
        y_val (pd.Series): Validation target variable data.
        model: The machine learning model instance (e.g., RandomForestRegressor).
        
    Returns:
        list: A list of the selected feature names.
    """
    selected_features = [] # Initialize with an empty list of selected features
    remaining_features = list(X_train.columns) # All features are initially remaining
    best_r2 = -np.inf # Initialize best R² with negative infinity
    
    print("Starting Forward Feature Selection...")
    print(f"Initial number of features: {len(remaining_features)}")
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    
    # Scale the target variable using RobustScaler (less sensitive to outliers)
    target_scaler = RobustScaler()
    y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel() # Fit and transform training target
    y_val_scaled = target_scaler.transform(y_val.values.reshape(-1, 1)).ravel() # Transform validation target
    
    # Scale features using RobustScaler
    feature_scaler = RobustScaler()
    X_train_scaled = feature_scaler.fit_transform(X_train) # Fit and transform training features
    X_val_scaled = feature_scaler.transform(X_val) # Transform validation features
    
    # Iterate while there are remaining features to test
    while remaining_features:
        best_feature = None # To store the best feature found in the current iteration
        best_r2_with_feature = -np.inf # To store the best R² achieved in the current iteration
        
        # Iterate through each remaining feature to test adding it
        for feature in remaining_features:
            features_to_test = selected_features + [feature] # Combine current selected features with the feature to test
            # Get the column indices for the features to test
            feature_indices = [X_train.columns.get_loc(f) for f in features_to_test]
            
            # Select the subset of scaled data for training and prediction with the current feature set
            X_train_subset = X_train_scaled[:, feature_indices]
            X_val_subset = X_val_scaled[:, feature_indices]
            
            # Train the model with the current feature subset and scaled target
            model.fit(X_train_subset, y_train_scaled)
            # Make predictions on the scaled validation data
            y_pred_val = model.predict(X_val_subset)
            
            # Calculate R² on the original scale of the target variable
            y_pred_val_original = target_scaler.inverse_transform(y_pred_val.reshape(-1, 1)).ravel()
            r2_with_feature = r2_score(y_val, y_pred_val_original)
            
            # Check if this feature combination gives a better R² score
            if r2_with_feature > best_r2_with_feature:
                best_r2_with_feature = r2_with_feature
                best_feature = feature # This feature is the best to add in this iteration
        
        # Check if the best R² achieved by adding a feature is better than the previous best R²
        if best_r2_with_feature > best_r2:
            selected_features.append(best_feature) # Add the best feature to the selected list
            remaining_features.remove(best_feature) # Remove the feature from the remaining list
            best_r2 = best_r2_with_feature # Update the best R²
            print(f"FFS: Added {best_feature}")
            print(f"New R²: {best_r2:.3f}")
            print(f"Selected features count: {len(selected_features)}")
        else:
            # If adding the best feature doesn't improve R², stop the selection process
            print("FFS: Adding more features did not improve R². Stopping.")
            break
    
    print(f"FFS finished after selecting {len(selected_features)} features")
    print(f"Final R² score: {best_r2:.3f}")
    print(f"Selected features: {selected_features}")
    return selected_features # Return the list of selected feature names

def check_data_suitability(X, y, min_points=5):
    """
    Check if the data is suitable for feature selection and model training.
    Ensures there are enough data points, no missing values, sufficient variance,
    and no constant features.
    
    Args:
        X (pd.DataFrame): Feature data.
        y (pd.Series): Target variable data.
        min_points (int): Minimum number of data points required.
        
    Returns:
        bool: True if data is suitable, False otherwise.
    """
    print("Data suitability checks:")
    
    # Check 1: Enough data points
    enough_points = len(X) >= min_points
    print(f"Enough data points ({len(X)} >= {min_points}): {'✓' if enough_points else '✗'}")

    # Check 2: No missing values in features or target
    no_missing_values = not X.isnull().any().any() and not y.isnull().any()
    print(f"No missing values: {'✓' if no_missing_values else '✗'}")

    # Check 3: Sufficient variance in the target variable
    sufficient_variance_target = y.std() > 1e-9 # Use a small tolerance for float comparison
    print(f"Sufficient variance in target: {'✓' if sufficient_variance_target else '✗'}")

    # Check 4: Features have variance (optional, but good practice)
    features_have_variance = all(X[col].std() > 1e-9 for col in X.columns) if not X.empty else True
    print(f"Features have variance: {'✓' if features_have_variance else '✗'}")

    # Check 5: No constant features (features with only one unique value)
    no_constant_features = not any(X[col].nunique() == 1 for col in X.columns) if not X.empty else True
    print(f"No constant features: {'✓' if no_constant_features else '✗'}")
            
    # Return True only if all checks pass
    return all([enough_points, no_missing_values, sufficient_variance_target, features_have_variance, no_constant_features])

def rolling_hrv_prediction_30day_window():
    """
    Main function to perform HRV prediction using a rolling 30-day window
    and Forward Feature Selection (FFS).
    
    Loads data, processes it, performs FFS on a rolling window, trains a model,
    makes predictions, calculates metrics, and generates plots.
    """
    # Create plots directory at the start if it doesn't exist
    os.makedirs('plots/ffs-30day-plots', exist_ok=True)
    
    # Define the window size and minimum data requirements
    train_window_size = 30  # Number of days in the training window
    prediction_offset = 1  # Number of days after the training window to predict
    validation_size = 0.2  # Percentage of training data to use for validation during feature selection
    min_data_points_for_fs = 5  # Minimum number of data points required for performing feature selection split
    min_data_points_per_day = 10  # Minimum number of data points required per day for meaningful analysis
    min_total_points = 300  # Minimum total points required for the entire training window (e.g., 30 days * 10 points/day)
    
    # Variables to track previous window results for reduced logging
    prev_selected_features = None
    prev_r2 = None
    prev_rmse = None
    prev_window_data_points = None
    
    # Read the combined data from the specified CSV file
    print("Loading data...")
    df = pd.read_csv('data/processed/combined_data_1min.csv', delimiter=';')
    time_column = 'minutes__minute' # Define the column containing timestamps
    
    # Convert the time column to datetime objects and set as index for easier date-based operations
    df[time_column] = pd.to_datetime(df[time_column], format='%d/%m/%Y %H:%M')
    df = df.set_index(time_column)
    
    # Determine the target variable for prediction
    target = 'night_hrv' if 'night_hrv' in df.columns else 'minutes__value__rmssd'
    print(f"Using target variable: {target}")

    # --- Feature Engineering: Add rate of change and rolling features for HRV ---
    print("Engineering new features...")
    # Ensure data is sorted by time for correct feature engineering
    df = df.sort_index()
    
    # Calculate 1-minute difference for the target variable (RMSSD or night_hrv)
    df['rmssd_diff_1min'] = df[target].diff(periods=1)
    # Calculate 1-minute difference for Heart Rate if available
    if 'minutes__value' in df.columns:
        df['hr_diff_1min_new'] = df['minutes__value'].diff(periods=1)

    # Calculate Rolling mean and standard deviation for the target variable over a 5-minute window
    df['rmssd_rolling_mean_5min'] = df[target].rolling(window='5min').mean()
    df['rmssd_rolling_std_5min'] = df[target].rolling(window='5min').std()

    # --- Data Filtering for Nighttime Data ---
    # Reset index to access date and hour components easily
    df = df.reset_index()
    print("Analyzing data distribution before nighttime filtering:")
    print(f"Total data points: {len(df)}")
    print(f"Date range: {df[time_column].min()} to {df[time_column].max()}")
    # Calculate average points per day before filtering
    print(f"Points per day (mean): {len(df) / len(df[time_column].dt.date.unique()):.1f}")
    
    # Filter data based on the 'night_hrv' column or time of day
    if 'night_hrv' in df.columns:
        print("Using 'night_hrv' column for nighttime filtering")
        df_night = df[df['night_hrv'] > 0].copy() # Keep rows where night_hrv is positive
        print(f"Points with night_hrv > 0: {len(df_night)}")
    else:
        print("Using time-based filtering for nighttime data (22:00 to 06:00)")
        df['hour'] = df[time_column].dt.hour # Extract hour
        df_night = df[((df['hour'] >= 22) | (df['hour'] < 6))].copy() # Filter for nighttime hours
        print(f"Points between 22:00 and 06:00: {len(df_night)}")
    
    # Re-set index to time_column for easier date-based operations later
    df_night = df_night.set_index(time_column)

    print("Analyzing nighttime data distribution:")
    print(f"Total nighttime data points: {len(df_night)}")
    print(f"Date range: {df_night.index.min()} to {df_night.index.max()}")
    
    # Calculate points per day for the filtered nighttime data
    points_per_day = df_night.groupby(df_night.index.date).size()
    print("Points per day statistics (Nighttime Data):")
    print(f"Mean points per day: {points_per_day.mean():.1f}")
    print(f"Min points per day: {points_per_day.min()}")
    print(f"Max points per day: {points_per_day.max()}")
    print(f"Days with < {min_data_points_per_day} points: {sum(points_per_day < min_data_points_per_day)}")
    
    # Get unique dates from the filtered nighttime data and sort them
    unique_dates = sorted(df_night.index.unique().date)
    
    print(f"Number of unique dates with nighttime data: {len(unique_dates)}")
    # Print the range of unique dates
    if unique_dates:
        print(f"Date range: {unique_dates[0]} to {unique_dates[-1]}")

    # Calculate the minimum required number of unique dates for the rolling window
    min_required_dates = train_window_size + prediction_offset
    if len(unique_dates) < min_required_dates:
        print(f"Not enough days of data ({len(unique_dates)}) for rolling window prediction with a {train_window_size}-day training window and {prediction_offset}-day offset. Need at least {min_required_dates} days.")
        print("Exiting script.")
        return # Exit the function if there aren't enough days

    # Prepare the initial list of potential features (all numeric columns excluding id, target, and temporary columns)
    numeric_columns = df_night.select_dtypes(include=[np.number]).columns
    features_to_exclude = ['id', target, 'minutes__value__rmssd', 'night_hrv', 'hour']
    features = [col for col in numeric_columns if col not in features_to_exclude]
    
    print(f"Number of initial features for model training: {len(features)}")
    print("Initial Features:", features)
    
    # Calculate the total number of possible prediction days
    possible_prediction_days = len(unique_dates) - train_window_size
    
    # Initialize lists and dictionaries to store results and data for plotting
    all_r2_scores = [] # To store R² scores for each window
    all_rmse_scores = [] # To store RMSE scores for each window
    all_predictions = {} # To store detailed prediction data per day
    plots_data = {} # To store data specifically for generating daily plots for selected indices
    
    # Select representative indices for generating detailed daily plots
    plot_indices = [0] # Always plot the first window
    if possible_prediction_days > 1:
        plot_indices.append(possible_prediction_days - 1) # Plot the last window
    if possible_prediction_days > 2:
        plot_indices.append((possible_prediction_days - 1) // 2) # Plot a middle window
    plot_indices = sorted(list(set(plot_indices))) # Ensure unique and sorted indices
    
    print(f"Total possible prediction days: {possible_prediction_days}")
    print(f"Selected indices for detailed daily plotting: {plot_indices}")

    # --- Rolling Window Prediction Loop ---
    for i in range(possible_prediction_days):
        print(f"--- Processing window {i+1}/{possible_prediction_days} ---")
        
        # Determine the start and end dates for the training window and the prediction date
        train_start_idx = i
        train_end_idx = i + train_window_size
        pred_idx = train_end_idx
        
        train_dates = unique_dates[train_start_idx:train_end_idx]
        pred_date = unique_dates[pred_idx]
        
        # Select data for the current training window and prediction day with detailed logging
        # Use .loc with date range for the training data
        train_data = df_night.loc[train_dates[0]:train_dates[-1]].copy()
        # Select data for the specific prediction date
        pred_data = df_night[df_night.index.date == pred_date].copy()
        
        # Log data points per day in the current training window
        train_points_per_day = train_data.groupby(train_data.index.date).size()
        current_window_data_points = len(train_data) # Total data points in the training window
        
        # Check if we have enough total data points in the training window
        if len(train_data) < min_total_points:
            print(f"Warning: Insufficient training data points ({len(train_data)}) for window {i+1}. Minimum required: {min_total_points}. Skipping window.")
            # Log days with insufficient points within this training window
            print(f"Days within window {i+1} with < {min_data_points_per_day} points: {sum(train_points_per_day < min_data_points_per_day)}")
            all_r2_scores.append(np.nan) # Append NaN for metrics as model was not trained
            all_rmse_scores.append(np.nan)
            continue # Move to the next window

        # Check if we have enough data points for the prediction day
        if len(pred_data) < min_data_points_per_day:
            print(f"Warning: Insufficient prediction data points ({len(pred_data)}) for date {pred_date} (window {i+1}). Minimum required: {min_data_points_per_day}. Skipping window.")
            all_r2_scores.append(np.nan) # Append NaN for metrics
            all_rmse_scores.append(np.nan)
            continue # Move to the next window

        # --- Handle Missing Values ---
        print("Handling missing values...")
        # Impute missing values using the median of the training data for each feature and target
        median_vals = train_data[features + [target]].median()
        train_data = train_data.fillna(median_vals)
        pred_data = pred_data.fillna(median_vals) # Use training median for prediction data

        # Prepare X and y for the current window
        # Select features that are present in both training and prediction data
        current_features = [f for f in features if f in train_data.columns and f in pred_data.columns]
        
        X_train_full = train_data[current_features].copy()
        y_train_full = train_data[target].copy()
        X_pred = pred_data[current_features].copy()
        y_actual = pred_data[target].copy()

        # Drop rows with NaN in target after imputation (should be none if imputation was successful for target)
        # Ensure index alignment if dropping NaNs
        valid_train_indices = y_train_full.dropna().index
        X_train_full = X_train_full.loc[valid_train_indices]
        y_train_full = y_train_full.loc[valid_train_indices]
        
        # Check if training or prediction data is empty after processing and dropping NaNs
        if X_train_full.empty or y_train_full.empty or X_pred.empty or y_actual.empty:
            print(f"Insufficient valid data points after processing for window {i+1}. Skipping.")
            all_r2_scores.append(np.nan) # Append NaN for metrics
            all_rmse_scores.append(np.nan)
            continue # Move to the next window

        # Check if y_train_full has sufficient variance for model training
        if y_train_full.std() == 0:
            print(f"Target variable has zero variance in training window {i+1}. Skipping model training.")
            all_r2_scores.append(np.nan) # Append NaN for metrics
            all_rmse_scores.append(np.nan)
            continue # Move to the next window

        # --- Prepare for Feature Selection ---
        print(f"Preparing for feature selection for window {i+1}...")
        
        # Check if the full training data is suitable for splitting and feature selection
        if not check_data_suitability(X_train_full, y_train_full, min_data_points_for_fs):
             print(f"Warning: Full training data for window {i+1} not suitable for feature selection. Using all current features.")
             # If not suitable, use all available features for the current window
             selected_features = current_features
        else:
            try:
                # Split the full training data into training and validation sets for feature selection
                X_train_fs, X_val_fs, y_train_fs, y_val_fs = train_test_split(
                    X_train_full, y_train_full, 
                    test_size=validation_size, 
                    random_state=42, # Use a fixed random state for reproducibility
                    shuffle=False # Maintain time series order
                )
                print(f"Feature selection split shapes: X_train_fs={X_train_fs.shape}, X_val_fs={X_val_fs.shape}")
                
                # Check if the split training and validation data are suitable for feature selection
                if not check_data_suitability(X_train_fs, y_train_fs, min_data_points_for_fs) or \
                   not check_data_suitability(X_val_fs, y_val_fs, 1): # Validation needs at least 1 point
                    print(f"Warning: Split training or validation data for window {i+1} not suitable for feature selection. Using all current features.")
                    selected_features = current_features
                else:
                    print("Data passed suitability checks after split. Proceeding with feature selection.")
                    # Create a fresh model instance specifically for feature selection
                    fs_model = RandomForestRegressor(
                        n_estimators=50, # Number of trees in the forest
                        max_depth=10, # Maximum depth of the trees
                        min_samples_split=5, # Minimum number of samples required to split an internal node
                        min_samples_leaf=2, # Minimum number of samples required to be at a leaf node
                        random_state=42 # For reproducibility
                    )
                    # Perform Forward Feature Selection
                    selected_features = forward_feature_selection(X_train_fs, y_train_fs, X_val_fs, y_val_fs, fs_model)
                    
            except ValueError as e:
                print(f"Could not split training data for feature selection in window {i+1}: {e}. Using all current features.")
                selected_features = current_features # Use all features if split fails

        # --- Logging for Window Processing ---
        # Check if current window results are identical to the previous window to avoid redundant logging
        # Compare selected features (as lists) and the number of data points
        if (prev_selected_features == selected_features and 
            prev_window_data_points == current_window_data_points):
            print(f"Window {i+1}: Results identical to previous window - skipping detailed window logging.")
        else:
            # Log detailed information if results are different
            print(f"--- Detailed Window Processing for {i+1}/{possible_prediction_days} ---")
            print(f"Training on dates: {train_dates[0]} to {train_dates[-1]}")
            print(f"Predicting for date: {pred_date}")
            print("Training data points per day statistics:")
            if not train_points_per_day.empty:
                 print(f"Mean: {train_points_per_day.mean():.1f}")
                 print(f"Min: {train_points_per_day.min()}")
                 print(f"Max: {train_points_per_day.max()}")
            else:
                 print("No data points in training window.")
            print(f"Total points in training window: {len(train_data)}")
            print(f"Selected Features for model training: {selected_features}")


        # --- Train Final Model and Evaluate with selected features ---
        r2 = np.nan # Initialize metrics as NaN
        rmse = np.nan

        # Proceed only if features were selected
        if selected_features:
            # Log model training information only if selected features or previous metrics changed
            if (prev_selected_features != selected_features or 
                prev_r2 != r2 or # This comparison will only be meaningful after metrics are calculated below
                prev_rmse != rmse): # Same as above
                print(f"Training final model with {len(selected_features)} selected features for window {i+1}...")
                
            # Prepare data with selected features for final model training and prediction
            X_train_final = X_train_full[selected_features]
            y_train_final = y_train_full
            X_pred_final = X_pred[selected_features] # Ensure prediction data uses the same selected features

            # Scale features for the final model training using RobustScaler
            feature_scaler_final = RobustScaler()
            X_train_scaled = feature_scaler_final.fit_transform(X_train_final) # Fit and transform training features
            X_pred_scaled = feature_scaler_final.transform(X_pred_final) # Transform prediction features
            
            # Scale target for the final model training using RobustScaler
            target_scaler_final = RobustScaler()
            y_train_scaled = target_scaler_final.fit_transform(y_train_final.values.reshape(-1, 1)).ravel() # Fit and transform training target
            
            # Create and train the final Random Forest Regressor model
            model = RandomForestRegressor(
                n_estimators=50,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42 # For reproducibility
            )
            model.fit(X_train_scaled, y_train_scaled) # Train the model on scaled data
            
            # Make predictions on the scaled prediction data
            y_pred_scaled = model.predict(X_pred_scaled)
            # Inverse transform predictions to the original scale of the target variable
            y_pred = target_scaler_final.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

            # Calculate performance metrics (R² and RMSE) for the current window
            # Ensure actual and predicted values have the same length and are not empty
            if len(y_actual) > 0 and len(y_pred) == len(y_actual):
                r2 = r2_score(y_actual, y_pred)
                rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
                
                # Log calculated metrics if they are different from previous window
                if (prev_r2 != r2 or prev_rmse != rmse):
                    print(f"Window {i+1} Metrics: R² Score: {r2:.3f}, RMSE: {rmse:.3f}")
            else:
                 print(f"Could not calculate metrics for window {i+1}. y_actual length: {len(y_actual)}, y_pred length: {len(y_pred)}")
                 r2 = np.nan # Set metrics to NaN if calculation is not possible
                 rmse = np.nan

            # Append calculated metrics to the overall lists
            all_r2_scores.append(r2)
            all_rmse_scores.append(rmse)
        else:
            # If no features were selected, skip model training and append NaN metrics
            print(f"No features selected for window {i+1}. Cannot train model.")
            all_r2_scores.append(np.nan)
            all_rmse_scores.append(np.nan)
        
        # Store predictions for every day where metrics were successfully calculated (R² is not NaN)
        if not np.isnan(r2):
            all_predictions[pred_date] = {
                'time': pred_data.index.tolist(), # Store original time index
                'actual': y_actual.values.tolist(), # Store actual values
                'predicted': y_pred.tolist() # Store predicted values
            }
        
        # Store data specifically for plotting detailed daily plots if this window's index is in plot_indices
        if i in plot_indices:
            if not np.isnan(r2): # Only store if metrics were calculated
                plots_data[i] = {
                    'actual': pd.DataFrame({'time': y_actual.index, 'actual': y_actual.values}),
                    'predicted': pd.DataFrame({'time': pred_data.index, 'predicted': y_pred}),
                    'date_str': pred_date.strftime('%Y-%m-%d') # Store date as a formatted string
                }
            else:
                # Log if no valid data is available for plotting for this selected day index
                print(f"No valid data or metrics available for plotting for day index {i+1}.")
                plots_data[i] = None # Store None if no data is available for plotting this index

        # Store feature importance for the first window where a model was successfully trained
        if i == 0:
            feature_importance = None # Initialize feature importance
            # Check if the model object exists and features were selected
            if 'model' in locals() and selected_features:
                # Create a DataFrame for feature importance and sort by importance
                feature_importance = pd.DataFrame({
                    'feature': selected_features,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
            else:
                 print("Model not trained in the first window, cannot get feature importance.")

        # Update previous window results for comparison in the next iteration
        prev_selected_features = selected_features.copy() if selected_features else None # Store a copy
        prev_r2 = r2
        prev_rmse = rmse
        prev_window_data_points = current_window_data_points

    # --- Post-processing and Plotting ---

    # Filter out outlier R² scores (e.g., below -10 which indicate very poor model performance) and corresponding RMSEs
    r2_threshold = -10.0 # Define the threshold for outlier R² scores
    print(f"Filtering out R² scores below {r2_threshold} for overall performance calculation.")
    
    # Create a list of results tuples (r2, rmse) excluding NaNs and outliers
    filtered_results = [(r2, rmse) for r2, rmse in zip(all_r2_scores, all_rmse_scores) if not np.isnan(r2) and r2 > r2_threshold]
    
    # --- Generate Plots ---
    print("Generating plots...")
    
    # Plot actual vs predicted and residuals for the selected representative days
    for idx in plot_indices:
        if idx in plots_data and plots_data[idx] is not None: # Check if data exists for this index
            # Call helper functions to generate and save daily plots
            plot_predictions(plots_data[idx]['actual'], plots_data[idx]['predicted'], 
                           plots_data[idx]['date_str'], f'predictions_day_{idx+1}_30day_window_new_features_ffs.png')
            plot_residuals(plots_data[idx]['actual'], plots_data[idx]['predicted'],
                         plots_data[idx]['date_str'], f'residuals_day_{idx+1}_30day_window_new_features_ffs.png')
        else:
            print(f"Skipping daily plots for day index {idx+1} due to no data or metrics available.")

    # Plot feature importance (only if feature importance data was captured in the first window)
    if 'feature_importance' in locals() and feature_importance is not None and not feature_importance.empty:
        plt.figure(figsize=(12, 6)) # Set figure size
        # Create a bar plot for the top 10 most important features
        sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
        plt.title('Top 10 Most Important Features (First Window - FFS)') # Plot title
        plt.tight_layout() # Adjust layout
        plt.savefig('plots/ffs-30day-plots/feature_importance_rolling_30day_window_new_features_ffs.png') # Save the plot
        plt.close() # Close the figure
    else:
         print("Not enough data or model not trained in first window to plot feature importance.")

    # Plot overall predictions, residuals, and performance analyses if any predictions were made
    if all_predictions:
        print("Generating overall prediction and performance plots...")
        plot_all_predictions(all_predictions, 'FFS')
        plot_all_residuals(all_predictions, 'FFS')
        plot_model_performance_metrics(all_predictions, 'FFS')
        plot_prediction_error_analysis(all_predictions, 'FFS')
        plot_time_series_analysis(all_predictions, 'FFS')
    else:
        print("No predictions available to generate overall plots.")


    # --- Print Overall Performance Metrics ---
    print("--- Overall Model Performance (Outliers Excluded - FFS) ---")

    # Calculate and print overall metrics based on filtered results
    if filtered_results:
        filtered_r2_scores = [res[0] for res in filtered_results]
        filtered_rmse_scores = [res[1] for res in filtered_results]
        
        # Calculate total data points across all included prediction days
        total_data_points = sum(len(all_predictions[date]['actual']) for date in all_predictions) if all_predictions else 0
        
        print(f"Number of prediction days included in analysis: {len(filtered_r2_scores)}")
        print(f"Total number of data points (measurements) across all included days: {total_data_points}")
        # Avoid division by zero if no days are included
        print(f"Average data points per day (included days): {total_data_points/len(filtered_r2_scores):.1f}" if len(filtered_r2_scores) > 0 else "Average data points per day: N/A")
        
        # Calculate and print mean and standard deviation for filtered R² and RMSE
        print(f"Average R² Score: {np.mean(filtered_r2_scores):.3f} ± {np.std(filtered_r2_scores):.3f}")
        print(f"Average RMSE: {np.mean(filtered_rmse_scores):.3f} ± {np.std(filtered_rmse_scores):.3f}")

        # Plot R² scores over time, showing both all and filtered scores
        plt.figure(figsize=(10, 6)) # Set figure size
        # Create indices corresponding to the valid R² scores
        valid_indices = [i for i, r2 in enumerate(all_r2_scores) if not np.isnan(r2)]
        valid_r2_scores = [r2 for r2 in all_r2_scores if not np.isnan(r2)]
        # Create indices corresponding to the filtered R² scores
        filtered_indices = [i for i, r2 in zip(valid_indices, valid_r2_scores) if r2 > r2_threshold]
        filtered_r2_scores_plot = [r2 for r2 in valid_r2_scores if r2 > r2_threshold]
        
        # Plot all valid R² scores
        plt.plot(valid_indices, valid_r2_scores, marker='o', linestyle='-', color='gray', alpha=0.5, label='All Valid R² Scores')
        # Plot filtered R² scores
        plt.plot(filtered_indices, filtered_r2_scores_plot, marker='o', linestyle='-', color='blue', label='Filtered R² Scores')
        # Add a horizontal line for the mean filtered R²
        plt.axhline(y=np.mean(filtered_r2_scores), color='r', linestyle='--', 
                    label=f'Mean Filtered R²: {np.mean(filtered_r2_scores):.3f}')
        plt.xlabel('Prediction Day Index') # X-axis label
        plt.ylabel('R² Score') # Y-axis label
        plt.title('R² Scores Over Time (Outliers Excluded from Average - FFS)') # Plot title
        plt.legend() # Show legend
        plt.tight_layout() # Adjust layout
        plt.savefig('plots/ffs-30day-plots/r2_scores_over_time_filtered_30day_window_new_features_ffs.png') # Save the plot
        plt.close() # Close the figure

    else:
        print("No R² scores met the filtering criteria for overall performance calculation.")

# --- Script Entry Point ---

if __name__ == "__main__":
    # This block is executed only when the script is run directly
=======
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta

def plot_predictions(actual_df, predicted_df, date_str, filename):
    """
    Helper function to plot actual vs predicted HRV for a given day.
    
    Args:
        actual_df (pd.DataFrame): DataFrame with actual HRV values (must have 'time' and 'actual' columns).
        predicted_df (pd.DataFrame): DataFrame with predicted HRV values (must have 'time' and 'predicted' columns).
        date_str (str): String representation of the date for the plot title.
        filename (str): The name of the file to save the plot to.
    """
    plt.figure(figsize=(15, 6)) # Set figure size
    # Plot actual HRV
    plt.plot(actual_df['time'], actual_df['actual'], 
             label='Actual HRV', color='blue')
    # Plot predicted HRV
    plt.plot(predicted_df['time'], predicted_df['predicted'], 
             label='Predicted HRV', color='red', linestyle='--')
    plt.xlabel('Time') # X-axis label
    plt.ylabel('HRV (RMSSD)') # Y-axis label
    plt.title(f'HRV Predictions vs Actual Values - {date_str}') # Plot title
    plt.legend() # Show legend
    plt.tight_layout() # Adjust layout
    plt.savefig(f'plots/ffs-30day-plots/{filename}') # Save the plot
    plt.close() # Close the figure to free up memory

def plot_residuals(actual_df, predicted_df, date_str, filename):
    """
    Helper function to plot residuals (actual - predicted) for a given day.
    
    Args:
        actual_df (pd.DataFrame): DataFrame with actual HRV values.
        predicted_df (pd.DataFrame): DataFrame with predicted HRV values.
        date_str (str): String representation of the date for the plot title.
        filename (str): The name of the file to save the plot to.
    """
    residuals = actual_df['actual'] - predicted_df['predicted'] # Calculate residuals
    plt.figure(figsize=(15, 6)) # Set figure size
    plt.scatter(predicted_df['time'], residuals, alpha=0.5) # Scatter plot of residuals over time
    plt.axhline(y=0, color='r', linestyle='--') # Add a horizontal line at y=0
    plt.xlabel('Time') # X-axis label
    plt.ylabel('Residuals (Actual - Predicted)') # Y-axis label
    plt.title(f'Residuals - {date_str}') # Plot title
    plt.tight_layout() # Adjust layout
    plt.savefig(f'plots/ffs-30day-plots/{filename}') # Save the plot
    plt.close() # Close the figure

def plot_all_predictions(all_predictions, method):
    """
    Helper function to plot all predictions vs actual values across all predicted days.
    
    Args:
        all_predictions (dict): Dictionary where keys are dates and values are dicts with 'time', 'actual', 'predicted'.
        method (str): Feature selection method used (e.g., 'FFS').
    """
    plt.figure(figsize=(20, 10)) # Set figure size
    
    # Sort predictions by date to ensure chronological plotting
    dates = sorted(all_predictions.keys())
    
    # Plot actual and predicted for each date
    for i, date in enumerate(dates):
        pred_data = all_predictions[date]
        # Plot actual HRV (lighter color)
        plt.plot(pred_data['time'], pred_data['actual'], 
                color='blue', alpha=0.3, label='Actual' if i == 0 else "") # Add label only for the first date
        # Plot predicted HRV (lighter color)
        plt.plot(pred_data['time'], pred_data['predicted'], 
                color='red', alpha=0.3, label='Predicted' if i == 0 else "") # Add label only for the first date
    
    plt.xlabel('Time') # X-axis label
    plt.ylabel('HRV (RMSSD)') # Y-axis label
    plt.title(f'All HRV Predictions vs Actual Values ({method})') # Plot title
    plt.legend() # Show legend
    plt.tight_layout() # Adjust layout
    plt.savefig(f'plots/ffs-30day-plots/all_predictions_{method}.png') # Save the plot
    plt.close() # Close the figure

def plot_all_residuals(all_predictions, method):
    """
    Helper function to plot all residuals across all predicted days.
    
    Args:
        all_predictions (dict): Dictionary where keys are dates and values are dicts with 'time', 'actual', 'predicted'.
        method (str): Feature selection method used (e.g., 'FFS').
    """
    plt.figure(figsize=(20, 10)) # Set figure size
    
    # Sort predictions by date
    dates = sorted(all_predictions.keys())
    
    # Plot residuals for each date as scattered points
    for i, date in enumerate(dates):
        pred_data = all_predictions[date]
        residuals = pred_data['actual'] - pred_data['predicted'] # Calculate residuals
        plt.scatter(pred_data['time'], residuals, alpha=0.3, 
                   label=date.strftime('%Y-%m-%d') if i == 0 else "") # Add label only for the first date
    
    plt.axhline(y=0, color='r', linestyle='--') # Add a horizontal line at y=0
    plt.xlabel('Time') # X-axis label
    plt.ylabel('Residuals (Actual - Predicted)') # Y-axis label
    plt.title(f'All Residuals ({method})') # Plot title
    plt.legend() # Show legend
    plt.tight_layout() # Adjust layout
    plt.savefig(f'plots/ffs-30day-plots/all_residuals_{method}.png') # Save the plot
    plt.close() # Close the figure

def plot_model_performance_metrics(all_predictions, method):
    """
    Plot comprehensive model performance metrics (R², RMSE, MAE, MAPE) over time.
    
    Args:
        all_predictions (dict): Dictionary where keys are dates and values are dicts with 'time', 'actual', 'predicted'.
        method (str): Feature selection method used (e.g., 'FFS').
    """
    if not all_predictions: # Check if there are any predictions
        print(f"No predictions available for {method} to plot performance metrics.")
        return
        
    plt.figure(figsize=(15, 10)) # Set figure size
    gs = plt.GridSpec(2, 2) # Create a 2x2 grid for subplots
    
    # Initialize lists to store metrics per day
    dates = []
    r2_scores = []
    rmse_scores = []
    mae_scores = []
    mape_scores = []
    
    # Calculate metrics for each predicted day
    for date, pred_data in all_predictions.items():
        actual = pred_data['actual']
        predicted = pred_data['predicted']
        
        # Calculate metrics using scikit-learn and numpy
        r2 = r2_score(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mae = np.mean(np.abs(actual - predicted))
        # Avoid division by zero for MAPE
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100 if np.mean(actual) != 0 else np.nan
        
        # Append metrics and date
        dates.append(date)
        r2_scores.append(r2)
        rmse_scores.append(rmse)
        mae_scores.append(mae)
        mape_scores.append(mape)
    
    # Plot R² Score over time
    ax1 = plt.subplot(gs[0, 0]) # Select top-left subplot
    ax1.plot(dates, r2_scores, marker='o') # Plot R² scores
    ax1.set_title('R² Score Over Time') # Subplot title
    ax1.set_xlabel('Date') # X-axis label
    ax1.set_ylabel('R² Score') # Y-axis label
    ax1.grid(True) # Show grid
    
    # Plot RMSE over time
    ax2 = plt.subplot(gs[0, 1]) # Select top-right subplot
    ax2.plot(dates, rmse_scores, marker='o', color='red') # Plot RMSE scores
    ax2.set_title('RMSE Over Time') # Subplot title
    ax2.set_xlabel('Date') # X-axis label
    ax2.set_ylabel('RMSE') # Y-axis label
    ax2.grid(True) # Show grid
    
    # Plot MAE over time
    ax3 = plt.subplot(gs[1, 0]) # Select bottom-left subplot
    ax3.plot(dates, mae_scores, marker='o', color='green') # Plot MAE scores
    ax3.set_title('MAE Over Time') # Subplot title
    ax3.set_xlabel('Date') # X-axis label
    ax3.set_ylabel('MAE') # Y-axis label
    ax3.grid(True) # Show grid
    
    # Plot MAPE over time
    ax4 = plt.subplot(gs[1, 1]) # Select bottom-right subplot
    ax4.plot(dates, mape_scores, marker='o', color='purple') # Plot MAPE scores
    ax4.set_title('MAPE Over Time') # Subplot title
    ax4.set_xlabel('Date') # X-axis label
    ax4.set_ylabel('MAPE (%)') # Y-axis label
    ax4.grid(True) # Show grid
    
    plt.tight_layout() # Adjust layout
    plt.savefig(f'plots/ffs-30day-plots/performance_metrics_{method}.png') # Save the plot
    plt.close() # Close the figure

def plot_prediction_error_analysis(all_predictions, method):
    """
    Plot detailed prediction error analysis, including scatter plots and distributions.
    
    Args:
        all_predictions (dict): Dictionary where keys are dates and values are dicts with 'time', 'actual', 'predicted'.
        method (str): Feature selection method used (e.g., 'FFS').
    """
    if not all_predictions: # Check if there are any predictions
        print(f"No predictions available for {method} to plot error analysis.")
        return
        
    plt.figure(figsize=(15, 10)) # Set figure size
    gs = plt.GridSpec(2, 2) # Create a 2x2 grid for subplots
    
    # Collect all actual, predicted, and error values from all days
    all_actual = []
    all_predicted = []
    all_errors = []
    
    for pred_data in all_predictions.values():
        all_actual.extend(pred_data['actual'])
        all_predicted.extend(pred_data['predicted'])
        all_errors.extend(pred_data['actual'] - pred_data['predicted']) # Calculate errors
    
    # Convert lists to numpy arrays for easier handling
    all_actual = np.array(all_actual)
    all_predicted = np.array(all_predicted)
    all_errors = np.array(all_errors)

    # Prediction vs Actual scatter plot
    ax1 = plt.subplot(gs[0, 0]) # Select top-left subplot
    ax1.scatter(all_actual, all_predicted, alpha=0.5) # Scatter plot
    # Add a line representing perfect prediction (actual == predicted)
    ax1.plot([min(all_actual), max(all_actual)], 
             [min(all_actual), max(all_actual)], 
             'r--', label='Perfect Prediction')
    ax1.set_title('Predicted vs Actual Values') # Subplot title
    ax1.set_xlabel('Actual HRV') # X-axis label
    ax1.set_ylabel('Predicted HRV') # Y-axis label
    ax1.legend() # Show legend
    ax1.grid(True) # Show grid
    
    # Error distribution
    ax2 = plt.subplot(gs[0, 1]) # Select top-right subplot
    sns.histplot(all_errors, kde=True, ax=ax2) # Histogram with KDE
    ax2.set_title('Error Distribution') # Subplot title
    ax2.set_xlabel('Prediction Error') # X-axis label
    ax2.set_ylabel('Frequency') # Y-axis label
    
    # Error vs Predicted value scatter plot
    ax3 = plt.subplot(gs[1, 0]) # Select bottom-left subplot
    ax3.scatter(all_predicted, all_errors, alpha=0.5) # Scatter plot
    ax3.axhline(y=0, color='r', linestyle='--') # Add a horizontal line at y=0
    ax3.set_title('Error vs Predicted Value') # Subplot title
    ax3.set_xlabel('Predicted HRV') # X-axis label
    ax3.set_ylabel('Prediction Error') # Y-axis label
    ax3.grid(True) # Show grid
    
    # Error vs Actual value scatter plot
    ax4 = plt.subplot(gs[1, 1]) # Select bottom-right subplot
    ax4.scatter(all_actual, all_errors, alpha=0.5) # Scatter plot
    ax4.axhline(y=0, color='r', linestyle='--') # Add a horizontal line at y=0
    ax4.set_title('Error vs Actual Value') # Subplot title
    ax4.set_xlabel('Actual HRV') # X-axis label
    ax4.set_ylabel('Prediction Error') # Y-axis label
    ax4.grid(True) # Show grid
    
    plt.tight_layout() # Adjust layout
    plt.savefig(f'plots/ffs-30day-plots/error_analysis_{method}.png') # Save the plot
    plt.close() # Close the figure

def plot_time_series_analysis(all_predictions, method):
    """
    Plot time series analysis of actual and predicted HRV values, including rolling means.
    
    Args:
        all_predictions (dict): Dictionary where keys are dates and values are dicts with 'time', 'actual', 'predicted'.
        method (str): Feature selection method used (e.g., 'FFS').
    """
    if not all_predictions: # Check if there are any predictions
        print(f"No predictions available for {method} to plot time series analysis.")
        return
        
    plt.figure(figsize=(15, 10)) # Set figure size
    gs = plt.GridSpec(2, 1, height_ratios=[2, 1]) # Create a 2x1 grid for subplots with specified height ratios
    
    # Collect all data points from all days
    all_times = []
    all_actual = []
    all_predicted = []
    
    for pred_data in all_predictions.values():
        all_times.extend(pred_data['time'])
        all_actual.extend(pred_data['actual'])
        all_predicted.extend(pred_data['predicted'])
    
    # Sort data chronologically by time
    sorted_indices = np.argsort(all_times)
    all_times = np.array(all_times)[sorted_indices]
    all_actual = np.array(all_actual)[sorted_indices]
    all_predicted = np.array(all_predicted)[sorted_indices]
    
    # Main time series plot (Actual vs Predicted)
    ax1 = plt.subplot(gs[0]) # Select the top subplot
    ax1.plot(all_times, all_actual, label='Actual HRV', alpha=0.5) # Plot actual HRV
    ax1.plot(all_times, all_predicted, label='Predicted HRV', alpha=0.5) # Plot predicted HRV
    ax1.set_title(f'HRV Time Series - {method}') # Subplot title
    ax1.set_xlabel('Time') # X-axis label
    ax1.set_ylabel('HRV (RMSSD)') # Y-axis label
    ax1.legend() # Show legend
    ax1.grid(True) # Show grid
    
    # Rolling statistics plot (Rolling Mean)
    window_size = 100  # Define the window size for rolling mean (adjust as needed)
    # Calculate rolling mean for actual and predicted values
    rolling_actual = pd.Series(all_actual).rolling(window=window_size).mean()
    rolling_predicted = pd.Series(all_predicted).rolling(window=window_size).mean()
    
    ax2 = plt.subplot(gs[1]) # Select the bottom subplot
    ax2.plot(all_times, rolling_actual, label='Actual HRV (Rolling Mean)', alpha=0.7) # Plot rolling mean of actual
    ax2.plot(all_times, rolling_predicted, label='Predicted HRV (Rolling Mean)', alpha=0.7) # Plot rolling mean of predicted
    ax2.set_title(f'Rolling Mean HRV (Window Size: {window_size})') # Subplot title
    ax2.set_xlabel('Time') # X-axis label
    ax2.set_ylabel('HRV (RMSSD)') # Y-axis label
    ax2.legend() # Show legend
    ax2.grid(True) # Show grid
    
    plt.tight_layout() # Adjust layout
    plt.savefig(f'plots/ffs-30day-plots/time_series_analysis_{method}.png') # Save the plot
    plt.close() # Close the figure

def forward_feature_selection(X_train, y_train, X_val, y_val, model):
    """
    Performs Iterative Forward Feature Selection based on R² score on a validation set.
    
    Starts with no features and adds the feature that provides the greatest improvement
    in R² score on the validation set at each step. Stops when adding no longer improves R².
    
    Args:
        X_train (pd.DataFrame): Training features data.
        y_train (pd.Series): Training target variable data.
        X_val (pd.DataFrame): Validation features data.
        y_val (pd.Series): Validation target variable data.
        model: The machine learning model instance (e.g., RandomForestRegressor).
        
    Returns:
        list: A list of the selected feature names.
    """
    selected_features = [] # Initialize with an empty list of selected features
    remaining_features = list(X_train.columns) # All features are initially remaining
    best_r2 = -np.inf # Initialize best R² with negative infinity
    
    print("Starting Forward Feature Selection...")
    print(f"Initial number of features: {len(remaining_features)}")
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    
    # Scale the target variable using RobustScaler (less sensitive to outliers)
    target_scaler = RobustScaler()
    y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel() # Fit and transform training target
    y_val_scaled = target_scaler.transform(y_val.values.reshape(-1, 1)).ravel() # Transform validation target
    
    # Scale features using RobustScaler
    feature_scaler = RobustScaler()
    X_train_scaled = feature_scaler.fit_transform(X_train) # Fit and transform training features
    X_val_scaled = feature_scaler.transform(X_val) # Transform validation features
    
    # Iterate while there are remaining features to test
    while remaining_features:
        best_feature = None # To store the best feature found in the current iteration
        best_r2_with_feature = -np.inf # To store the best R² achieved in the current iteration
        
        # Iterate through each remaining feature to test adding it
        for feature in remaining_features:
            features_to_test = selected_features + [feature] # Combine current selected features with the feature to test
            # Get the column indices for the features to test
            feature_indices = [X_train.columns.get_loc(f) for f in features_to_test]
            
            # Select the subset of scaled data for training and prediction with the current feature set
            X_train_subset = X_train_scaled[:, feature_indices]
            X_val_subset = X_val_scaled[:, feature_indices]
            
            # Train the model with the current feature subset and scaled target
            model.fit(X_train_subset, y_train_scaled)
            # Make predictions on the scaled validation data
            y_pred_val = model.predict(X_val_subset)
            
            # Calculate R² on the original scale of the target variable
            y_pred_val_original = target_scaler.inverse_transform(y_pred_val.reshape(-1, 1)).ravel()
            r2_with_feature = r2_score(y_val, y_pred_val_original)
            
            # Check if this feature combination gives a better R² score
            if r2_with_feature > best_r2_with_feature:
                best_r2_with_feature = r2_with_feature
                best_feature = feature # This feature is the best to add in this iteration
        
        # Check if the best R² achieved by adding a feature is better than the previous best R²
        if best_r2_with_feature > best_r2:
            selected_features.append(best_feature) # Add the best feature to the selected list
            remaining_features.remove(best_feature) # Remove the feature from the remaining list
            best_r2 = best_r2_with_feature # Update the best R²
            print(f"FFS: Added {best_feature}")
            print(f"New R²: {best_r2:.3f}")
            print(f"Selected features count: {len(selected_features)}")
        else:
            # If adding the best feature doesn't improve R², stop the selection process
            print("FFS: Adding more features did not improve R². Stopping.")
            break
    
    print(f"FFS finished after selecting {len(selected_features)} features")
    print(f"Final R² score: {best_r2:.3f}")
    print(f"Selected features: {selected_features}")
    return selected_features # Return the list of selected feature names

def check_data_suitability(X, y, min_points=5):
    """
    Check if the data is suitable for feature selection and model training.
    Ensures there are enough data points, no missing values, sufficient variance,
    and no constant features.
    
    Args:
        X (pd.DataFrame): Feature data.
        y (pd.Series): Target variable data.
        min_points (int): Minimum number of data points required.
        
    Returns:
        bool: True if data is suitable, False otherwise.
    """
    print("Data suitability checks:")
    
    # Check 1: Enough data points
    enough_points = len(X) >= min_points
    print(f"Enough data points ({len(X)} >= {min_points}): {'✓' if enough_points else '✗'}")

    # Check 2: No missing values in features or target
    no_missing_values = not X.isnull().any().any() and not y.isnull().any()
    print(f"No missing values: {'✓' if no_missing_values else '✗'}")

    # Check 3: Sufficient variance in the target variable
    sufficient_variance_target = y.std() > 1e-9 # Use a small tolerance for float comparison
    print(f"Sufficient variance in target: {'✓' if sufficient_variance_target else '✗'}")

    # Check 4: Features have variance (optional, but good practice)
    features_have_variance = all(X[col].std() > 1e-9 for col in X.columns) if not X.empty else True
    print(f"Features have variance: {'✓' if features_have_variance else '✗'}")

    # Check 5: No constant features (features with only one unique value)
    no_constant_features = not any(X[col].nunique() == 1 for col in X.columns) if not X.empty else True
    print(f"No constant features: {'✓' if no_constant_features else '✗'}")
            
    # Return True only if all checks pass
    return all([enough_points, no_missing_values, sufficient_variance_target, features_have_variance, no_constant_features])

def rolling_hrv_prediction_30day_window():
    """
    Main function to perform HRV prediction using a rolling 30-day window
    and Forward Feature Selection (FFS).
    
    Loads data, processes it, performs FFS on a rolling window, trains a model,
    makes predictions, calculates metrics, and generates plots.
    """
    # Create plots directory at the start if it doesn't exist
    os.makedirs('plots/ffs-30day-plots', exist_ok=True)
    
    # Define the window size and minimum data requirements
    train_window_size = 30  # Number of days in the training window
    prediction_offset = 1  # Number of days after the training window to predict
    validation_size = 0.2  # Percentage of training data to use for validation during feature selection
    min_data_points_for_fs = 5  # Minimum number of data points required for performing feature selection split
    min_data_points_per_day = 10  # Minimum number of data points required per day for meaningful analysis
    min_total_points = 300  # Minimum total points required for the entire training window (e.g., 30 days * 10 points/day)
    
    # Variables to track previous window results for reduced logging
    prev_selected_features = None
    prev_r2 = None
    prev_rmse = None
    prev_window_data_points = None
    
    # Read the combined data from the specified CSV file
    print("Loading data...")
    df = pd.read_csv('data/processed/combined_data_1min.csv', delimiter=';')
    time_column = 'minutes__minute' # Define the column containing timestamps
    
    # Convert the time column to datetime objects and set as index for easier date-based operations
    df[time_column] = pd.to_datetime(df[time_column], format='%d/%m/%Y %H:%M')
    df = df.set_index(time_column)
    
    # Determine the target variable for prediction
    target = 'night_hrv' if 'night_hrv' in df.columns else 'minutes__value__rmssd'
    print(f"Using target variable: {target}")

    # --- Feature Engineering: Add rate of change and rolling features for HRV ---
    print("Engineering new features...")
    # Ensure data is sorted by time for correct feature engineering
    df = df.sort_index()
    
    # Calculate 1-minute difference for the target variable (RMSSD or night_hrv)
    df['rmssd_diff_1min'] = df[target].diff(periods=1)
    # Calculate 1-minute difference for Heart Rate if available
    if 'minutes__value' in df.columns:
        df['hr_diff_1min_new'] = df['minutes__value'].diff(periods=1)

    # Calculate Rolling mean and standard deviation for the target variable over a 5-minute window
    df['rmssd_rolling_mean_5min'] = df[target].rolling(window='5min').mean()
    df['rmssd_rolling_std_5min'] = df[target].rolling(window='5min').std()

    # --- Data Filtering for Nighttime Data ---
    # Reset index to access date and hour components easily
    df = df.reset_index()
    print("Analyzing data distribution before nighttime filtering:")
    print(f"Total data points: {len(df)}")
    print(f"Date range: {df[time_column].min()} to {df[time_column].max()}")
    # Calculate average points per day before filtering
    print(f"Points per day (mean): {len(df) / len(df[time_column].dt.date.unique()):.1f}")
    
    # Filter data based on the 'night_hrv' column or time of day
    if 'night_hrv' in df.columns:
        print("Using 'night_hrv' column for nighttime filtering")
        df_night = df[df['night_hrv'] > 0].copy() # Keep rows where night_hrv is positive
        print(f"Points with night_hrv > 0: {len(df_night)}")
    else:
        print("Using time-based filtering for nighttime data (22:00 to 06:00)")
        df['hour'] = df[time_column].dt.hour # Extract hour
        df_night = df[((df['hour'] >= 22) | (df['hour'] < 6))].copy() # Filter for nighttime hours
        print(f"Points between 22:00 and 06:00: {len(df_night)}")
    
    # Re-set index to time_column for easier date-based operations later
    df_night = df_night.set_index(time_column)

    print("Analyzing nighttime data distribution:")
    print(f"Total nighttime data points: {len(df_night)}")
    print(f"Date range: {df_night.index.min()} to {df_night.index.max()}")
    
    # Calculate points per day for the filtered nighttime data
    points_per_day = df_night.groupby(df_night.index.date).size()
    print("Points per day statistics (Nighttime Data):")
    print(f"Mean points per day: {points_per_day.mean():.1f}")
    print(f"Min points per day: {points_per_day.min()}")
    print(f"Max points per day: {points_per_day.max()}")
    print(f"Days with < {min_data_points_per_day} points: {sum(points_per_day < min_data_points_per_day)}")
    
    # Get unique dates from the filtered nighttime data and sort them
    unique_dates = sorted(df_night.index.unique().date)
    
    print(f"Number of unique dates with nighttime data: {len(unique_dates)}")
    # Print the range of unique dates
    if unique_dates:
        print(f"Date range: {unique_dates[0]} to {unique_dates[-1]}")

    # Calculate the minimum required number of unique dates for the rolling window
    min_required_dates = train_window_size + prediction_offset
    if len(unique_dates) < min_required_dates:
        print(f"Not enough days of data ({len(unique_dates)}) for rolling window prediction with a {train_window_size}-day training window and {prediction_offset}-day offset. Need at least {min_required_dates} days.")
        print("Exiting script.")
        return # Exit the function if there aren't enough days

    # Prepare the initial list of potential features (all numeric columns excluding id, target, and temporary columns)
    numeric_columns = df_night.select_dtypes(include=[np.number]).columns
    features_to_exclude = ['id', target, 'minutes__value__rmssd', 'night_hrv', 'hour']
    features = [col for col in numeric_columns if col not in features_to_exclude]
    
    print(f"Number of initial features for model training: {len(features)}")
    print("Initial Features:", features)
    
    # Calculate the total number of possible prediction days
    possible_prediction_days = len(unique_dates) - train_window_size
    
    # Initialize lists and dictionaries to store results and data for plotting
    all_r2_scores = [] # To store R² scores for each window
    all_rmse_scores = [] # To store RMSE scores for each window
    all_predictions = {} # To store detailed prediction data per day
    plots_data = {} # To store data specifically for generating daily plots for selected indices
    
    # Select representative indices for generating detailed daily plots
    plot_indices = [0] # Always plot the first window
    if possible_prediction_days > 1:
        plot_indices.append(possible_prediction_days - 1) # Plot the last window
    if possible_prediction_days > 2:
        plot_indices.append((possible_prediction_days - 1) // 2) # Plot a middle window
    plot_indices = sorted(list(set(plot_indices))) # Ensure unique and sorted indices
    
    print(f"Total possible prediction days: {possible_prediction_days}")
    print(f"Selected indices for detailed daily plotting: {plot_indices}")

    # --- Rolling Window Prediction Loop ---
    for i in range(possible_prediction_days):
        print(f"--- Processing window {i+1}/{possible_prediction_days} ---")
        
        # Determine the start and end dates for the training window and the prediction date
        train_start_idx = i
        train_end_idx = i + train_window_size
        pred_idx = train_end_idx
        
        train_dates = unique_dates[train_start_idx:train_end_idx]
        pred_date = unique_dates[pred_idx]
        
        # Select data for the current training window and prediction day with detailed logging
        # Use .loc with date range for the training data
        train_data = df_night.loc[train_dates[0]:train_dates[-1]].copy()
        # Select data for the specific prediction date
        pred_data = df_night[df_night.index.date == pred_date].copy()
        
        # Log data points per day in the current training window
        train_points_per_day = train_data.groupby(train_data.index.date).size()
        current_window_data_points = len(train_data) # Total data points in the training window
        
        # Check if we have enough total data points in the training window
        if len(train_data) < min_total_points:
            print(f"Warning: Insufficient training data points ({len(train_data)}) for window {i+1}. Minimum required: {min_total_points}. Skipping window.")
            # Log days with insufficient points within this training window
            print(f"Days within window {i+1} with < {min_data_points_per_day} points: {sum(train_points_per_day < min_data_points_per_day)}")
            all_r2_scores.append(np.nan) # Append NaN for metrics as model was not trained
            all_rmse_scores.append(np.nan)
            continue # Move to the next window

        # Check if we have enough data points for the prediction day
        if len(pred_data) < min_data_points_per_day:
            print(f"Warning: Insufficient prediction data points ({len(pred_data)}) for date {pred_date} (window {i+1}). Minimum required: {min_data_points_per_day}. Skipping window.")
            all_r2_scores.append(np.nan) # Append NaN for metrics
            all_rmse_scores.append(np.nan)
            continue # Move to the next window

        # --- Handle Missing Values ---
        print("Handling missing values...")
        # Impute missing values using the median of the training data for each feature and target
        median_vals = train_data[features + [target]].median()
        train_data = train_data.fillna(median_vals)
        pred_data = pred_data.fillna(median_vals) # Use training median for prediction data

        # Prepare X and y for the current window
        # Select features that are present in both training and prediction data
        current_features = [f for f in features if f in train_data.columns and f in pred_data.columns]
        
        X_train_full = train_data[current_features].copy()
        y_train_full = train_data[target].copy()
        X_pred = pred_data[current_features].copy()
        y_actual = pred_data[target].copy()

        # Drop rows with NaN in target after imputation (should be none if imputation was successful for target)
        # Ensure index alignment if dropping NaNs
        valid_train_indices = y_train_full.dropna().index
        X_train_full = X_train_full.loc[valid_train_indices]
        y_train_full = y_train_full.loc[valid_train_indices]
        
        # Check if training or prediction data is empty after processing and dropping NaNs
        if X_train_full.empty or y_train_full.empty or X_pred.empty or y_actual.empty:
            print(f"Insufficient valid data points after processing for window {i+1}. Skipping.")
            all_r2_scores.append(np.nan) # Append NaN for metrics
            all_rmse_scores.append(np.nan)
            continue # Move to the next window

        # Check if y_train_full has sufficient variance for model training
        if y_train_full.std() == 0:
            print(f"Target variable has zero variance in training window {i+1}. Skipping model training.")
            all_r2_scores.append(np.nan) # Append NaN for metrics
            all_rmse_scores.append(np.nan)
            continue # Move to the next window

        # --- Prepare for Feature Selection ---
        print(f"Preparing for feature selection for window {i+1}...")
        
        # Check if the full training data is suitable for splitting and feature selection
        if not check_data_suitability(X_train_full, y_train_full, min_data_points_for_fs):
             print(f"Warning: Full training data for window {i+1} not suitable for feature selection. Using all current features.")
             # If not suitable, use all available features for the current window
             selected_features = current_features
        else:
            try:
                # Split the full training data into training and validation sets for feature selection
                X_train_fs, X_val_fs, y_train_fs, y_val_fs = train_test_split(
                    X_train_full, y_train_full, 
                    test_size=validation_size, 
                    random_state=42, # Use a fixed random state for reproducibility
                    shuffle=False # Maintain time series order
                )
                print(f"Feature selection split shapes: X_train_fs={X_train_fs.shape}, X_val_fs={X_val_fs.shape}")
                
                # Check if the split training and validation data are suitable for feature selection
                if not check_data_suitability(X_train_fs, y_train_fs, min_data_points_for_fs) or \
                   not check_data_suitability(X_val_fs, y_val_fs, 1): # Validation needs at least 1 point
                    print(f"Warning: Split training or validation data for window {i+1} not suitable for feature selection. Using all current features.")
                    selected_features = current_features
                else:
                    print("Data passed suitability checks after split. Proceeding with feature selection.")
                    # Create a fresh model instance specifically for feature selection
                    fs_model = RandomForestRegressor(
                        n_estimators=50, # Number of trees in the forest
                        max_depth=10, # Maximum depth of the trees
                        min_samples_split=5, # Minimum number of samples required to split an internal node
                        min_samples_leaf=2, # Minimum number of samples required to be at a leaf node
                        random_state=42 # For reproducibility
                    )
                    # Perform Forward Feature Selection
                    selected_features = forward_feature_selection(X_train_fs, y_train_fs, X_val_fs, y_val_fs, fs_model)
                    
            except ValueError as e:
                print(f"Could not split training data for feature selection in window {i+1}: {e}. Using all current features.")
                selected_features = current_features # Use all features if split fails

        # --- Logging for Window Processing ---
        # Check if current window results are identical to the previous window to avoid redundant logging
        # Compare selected features (as lists) and the number of data points
        if (prev_selected_features == selected_features and 
            prev_window_data_points == current_window_data_points):
            print(f"Window {i+1}: Results identical to previous window - skipping detailed window logging.")
        else:
            # Log detailed information if results are different
            print(f"--- Detailed Window Processing for {i+1}/{possible_prediction_days} ---")
            print(f"Training on dates: {train_dates[0]} to {train_dates[-1]}")
            print(f"Predicting for date: {pred_date}")
            print("Training data points per day statistics:")
            if not train_points_per_day.empty:
                 print(f"Mean: {train_points_per_day.mean():.1f}")
                 print(f"Min: {train_points_per_day.min()}")
                 print(f"Max: {train_points_per_day.max()}")
            else:
                 print("No data points in training window.")
            print(f"Total points in training window: {len(train_data)}")
            print(f"Selected Features for model training: {selected_features}")


        # --- Train Final Model and Evaluate with selected features ---
        r2 = np.nan # Initialize metrics as NaN
        rmse = np.nan

        # Proceed only if features were selected
        if selected_features:
            # Log model training information only if selected features or previous metrics changed
            if (prev_selected_features != selected_features or 
                prev_r2 != r2 or # This comparison will only be meaningful after metrics are calculated below
                prev_rmse != rmse): # Same as above
                print(f"Training final model with {len(selected_features)} selected features for window {i+1}...")
                
            # Prepare data with selected features for final model training and prediction
            X_train_final = X_train_full[selected_features]
            y_train_final = y_train_full
            X_pred_final = X_pred[selected_features] # Ensure prediction data uses the same selected features

            # Scale features for the final model training using RobustScaler
            feature_scaler_final = RobustScaler()
            X_train_scaled = feature_scaler_final.fit_transform(X_train_final) # Fit and transform training features
            X_pred_scaled = feature_scaler_final.transform(X_pred_final) # Transform prediction features
            
            # Scale target for the final model training using RobustScaler
            target_scaler_final = RobustScaler()
            y_train_scaled = target_scaler_final.fit_transform(y_train_final.values.reshape(-1, 1)).ravel() # Fit and transform training target
            
            # Create and train the final Random Forest Regressor model
            model = RandomForestRegressor(
                n_estimators=50,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42 # For reproducibility
            )
            model.fit(X_train_scaled, y_train_scaled) # Train the model on scaled data
            
            # Make predictions on the scaled prediction data
            y_pred_scaled = model.predict(X_pred_scaled)
            # Inverse transform predictions to the original scale of the target variable
            y_pred = target_scaler_final.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

            # Calculate performance metrics (R² and RMSE) for the current window
            # Ensure actual and predicted values have the same length and are not empty
            if len(y_actual) > 0 and len(y_pred) == len(y_actual):
                r2 = r2_score(y_actual, y_pred)
                rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
                
                # Log calculated metrics if they are different from previous window
                if (prev_r2 != r2 or prev_rmse != rmse):
                    print(f"Window {i+1} Metrics: R² Score: {r2:.3f}, RMSE: {rmse:.3f}")
            else:
                 print(f"Could not calculate metrics for window {i+1}. y_actual length: {len(y_actual)}, y_pred length: {len(y_pred)}")
                 r2 = np.nan # Set metrics to NaN if calculation is not possible
                 rmse = np.nan

            # Append calculated metrics to the overall lists
            all_r2_scores.append(r2)
            all_rmse_scores.append(rmse)
        else:
            # If no features were selected, skip model training and append NaN metrics
            print(f"No features selected for window {i+1}. Cannot train model.")
            all_r2_scores.append(np.nan)
            all_rmse_scores.append(np.nan)
        
        # Store predictions for every day where metrics were successfully calculated (R² is not NaN)
        if not np.isnan(r2):
            all_predictions[pred_date] = {
                'time': pred_data.index.tolist(), # Store original time index
                'actual': y_actual.values.tolist(), # Store actual values
                'predicted': y_pred.tolist() # Store predicted values
            }
        
        # Store data specifically for plotting detailed daily plots if this window's index is in plot_indices
        if i in plot_indices:
            if not np.isnan(r2): # Only store if metrics were calculated
                plots_data[i] = {
                    'actual': pd.DataFrame({'time': y_actual.index, 'actual': y_actual.values}),
                    'predicted': pd.DataFrame({'time': pred_data.index, 'predicted': y_pred}),
                    'date_str': pred_date.strftime('%Y-%m-%d') # Store date as a formatted string
                }
            else:
                # Log if no valid data is available for plotting for this selected day index
                print(f"No valid data or metrics available for plotting for day index {i+1}.")
                plots_data[i] = None # Store None if no data is available for plotting this index

        # Store feature importance for the first window where a model was successfully trained
        if i == 0:
            feature_importance = None # Initialize feature importance
            # Check if the model object exists and features were selected
            if 'model' in locals() and selected_features:
                # Create a DataFrame for feature importance and sort by importance
                feature_importance = pd.DataFrame({
                    'feature': selected_features,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
            else:
                 print("Model not trained in the first window, cannot get feature importance.")

        # Update previous window results for comparison in the next iteration
        prev_selected_features = selected_features.copy() if selected_features else None # Store a copy
        prev_r2 = r2
        prev_rmse = rmse
        prev_window_data_points = current_window_data_points

    # --- Post-processing and Plotting ---

    # Filter out outlier R² scores (e.g., below -10 which indicate very poor model performance) and corresponding RMSEs
    r2_threshold = -10.0 # Define the threshold for outlier R² scores
    print(f"Filtering out R² scores below {r2_threshold} for overall performance calculation.")
    
    # Create a list of results tuples (r2, rmse) excluding NaNs and outliers
    filtered_results = [(r2, rmse) for r2, rmse in zip(all_r2_scores, all_rmse_scores) if not np.isnan(r2) and r2 > r2_threshold]
    
    # --- Generate Plots ---
    print("Generating plots...")
    
    # Plot actual vs predicted and residuals for the selected representative days
    for idx in plot_indices:
        if idx in plots_data and plots_data[idx] is not None: # Check if data exists for this index
            # Call helper functions to generate and save daily plots
            plot_predictions(plots_data[idx]['actual'], plots_data[idx]['predicted'], 
                           plots_data[idx]['date_str'], f'predictions_day_{idx+1}_30day_window_new_features_ffs.png')
            plot_residuals(plots_data[idx]['actual'], plots_data[idx]['predicted'],
                         plots_data[idx]['date_str'], f'residuals_day_{idx+1}_30day_window_new_features_ffs.png')
        else:
            print(f"Skipping daily plots for day index {idx+1} due to no data or metrics available.")

    # Plot feature importance (only if feature importance data was captured in the first window)
    if 'feature_importance' in locals() and feature_importance is not None and not feature_importance.empty:
        plt.figure(figsize=(12, 6)) # Set figure size
        # Create a bar plot for the top 10 most important features
        sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
        plt.title('Top 10 Most Important Features (First Window - FFS)') # Plot title
        plt.tight_layout() # Adjust layout
        plt.savefig('plots/ffs-30day-plots/feature_importance_rolling_30day_window_new_features_ffs.png') # Save the plot
        plt.close() # Close the figure
    else:
         print("Not enough data or model not trained in first window to plot feature importance.")

    # Plot overall predictions, residuals, and performance analyses if any predictions were made
    if all_predictions:
        print("Generating overall prediction and performance plots...")
        plot_all_predictions(all_predictions, 'FFS')
        plot_all_residuals(all_predictions, 'FFS')
        plot_model_performance_metrics(all_predictions, 'FFS')
        plot_prediction_error_analysis(all_predictions, 'FFS')
        plot_time_series_analysis(all_predictions, 'FFS')
    else:
        print("No predictions available to generate overall plots.")


    # --- Print Overall Performance Metrics ---
    print("--- Overall Model Performance (Outliers Excluded - FFS) ---")

    # Calculate and print overall metrics based on filtered results
    if filtered_results:
        filtered_r2_scores = [res[0] for res in filtered_results]
        filtered_rmse_scores = [res[1] for res in filtered_results]
        
        # Calculate total data points across all included prediction days
        total_data_points = sum(len(all_predictions[date]['actual']) for date in all_predictions) if all_predictions else 0
        
        print(f"Number of prediction days included in analysis: {len(filtered_r2_scores)}")
        print(f"Total number of data points (measurements) across all included days: {total_data_points}")
        # Avoid division by zero if no days are included
        print(f"Average data points per day (included days): {total_data_points/len(filtered_r2_scores):.1f}" if len(filtered_r2_scores) > 0 else "Average data points per day: N/A")
        
        # Calculate and print mean and standard deviation for filtered R² and RMSE
        print(f"Average R² Score: {np.mean(filtered_r2_scores):.3f} ± {np.std(filtered_r2_scores):.3f}")
        print(f"Average RMSE: {np.mean(filtered_rmse_scores):.3f} ± {np.std(filtered_rmse_scores):.3f}")

        # Plot R² scores over time, showing both all and filtered scores
        plt.figure(figsize=(10, 6)) # Set figure size
        # Create indices corresponding to the valid R² scores
        valid_indices = [i for i, r2 in enumerate(all_r2_scores) if not np.isnan(r2)]
        valid_r2_scores = [r2 for r2 in all_r2_scores if not np.isnan(r2)]
        # Create indices corresponding to the filtered R² scores
        filtered_indices = [i for i, r2 in zip(valid_indices, valid_r2_scores) if r2 > r2_threshold]
        filtered_r2_scores_plot = [r2 for r2 in valid_r2_scores if r2 > r2_threshold]
        
        # Plot all valid R² scores
        plt.plot(valid_indices, valid_r2_scores, marker='o', linestyle='-', color='gray', alpha=0.5, label='All Valid R² Scores')
        # Plot filtered R² scores
        plt.plot(filtered_indices, filtered_r2_scores_plot, marker='o', linestyle='-', color='blue', label='Filtered R² Scores')
        # Add a horizontal line for the mean filtered R²
        plt.axhline(y=np.mean(filtered_r2_scores), color='r', linestyle='--', 
                    label=f'Mean Filtered R²: {np.mean(filtered_r2_scores):.3f}')
        plt.xlabel('Prediction Day Index') # X-axis label
        plt.ylabel('R² Score') # Y-axis label
        plt.title('R² Scores Over Time (Outliers Excluded from Average - FFS)') # Plot title
        plt.legend() # Show legend
        plt.tight_layout() # Adjust layout
        plt.savefig('plots/ffs-30day-plots/r2_scores_over_time_filtered_30day_window_new_features_ffs.png') # Save the plot
        plt.close() # Close the figure

    else:
        print("No R² scores met the filtering criteria for overall performance calculation.")

# --- Script Entry Point ---

if __name__ == "__main__":
    # This block is executed only when the script is run directly
>>>>>>> b3c5f6d48550dcf3f2f580ae4362896e4a64ab14
    rolling_hrv_prediction_30day_window() # Call the main function to start the process 