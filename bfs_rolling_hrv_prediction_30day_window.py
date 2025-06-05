import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os
from datetime import datetime, timedelta

def remove_outliers(df, columns, n_std=3):
    """
    Removes outliers from specified columns in a DataFrame based on standard deviation.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        columns (list): A list of column names to check for outliers.
        n_std (int): The number of standard deviations to use as the threshold for outlier detection.
        
    Returns:
        pd.DataFrame: A new DataFrame with outliers removed.
    """
    df_clean = df.copy()
    for col in columns:
        if col in df_clean.columns:
            mean = df_clean[col].mean()
            std = df_clean[col].std()
            df_clean = df_clean[abs(df_clean[col] - mean) <= (n_std * std)]
        else:
            print(f"Warning: Column '{col}' not found for outlier removal.")
    return df_clean

def plot_predictions(actual_df, predicted_df, date_str, filename):
    """
    Helper function to plot actual vs predicted HRV for a given day.
    
    Args:
        actual_df (pd.DataFrame): DataFrame with actual HRV values (must have 'time' and 'actual' columns).
        predicted_df (pd.DataFrame): DataFrame with predicted HRV values (must have 'time' and 'predicted' columns).
        date_str (str): String representation of the date for the plot title.
        filename (str): The name of the file to save the plot to.
    """
    plt.figure(figsize=(15, 6))
    plt.plot(actual_df['time'], actual_df['actual'], 
             label='Actual HRV', color='blue')
    plt.plot(predicted_df['time'], predicted_df['predicted'], 
             label='Predicted HRV', color='red', linestyle='--')
    plt.xlabel('Time')
    plt.ylabel('HRV (RMSSD)')
    plt.title(f'HRV Predictions vs Actual Values - {date_str}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'plots/bfs-30day-plots/{filename}')
    plt.close()

def plot_residuals(actual_df, predicted_df, date_str, filename):
    """
    Helper function to plot residuals (actual - predicted) for a given day.
    
    Args:
        actual_df (pd.DataFrame): DataFrame with actual HRV values.
        predicted_df (pd.DataFrame): DataFrame with predicted HRV values.
        date_str (str): String representation of the date for the plot title.
        filename (str): The name of the file to save the plot to.
    """
    residuals = actual_df['actual'] - predicted_df['predicted']
    plt.figure(figsize=(15, 6))
    plt.scatter(predicted_df['time'], residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Time')
    plt.ylabel('Residuals (Actual - Predicted)')
    plt.title(f'Residuals - {date_str}')
    plt.tight_layout()
    plt.savefig(f'plots/bfs-30day-plots/{filename}')
    plt.close()

def plot_all_predictions(all_predictions, method):
    """
    Helper function to plot all predictions vs actual values across all predicted days.
    
    Args:
        all_predictions (dict): Dictionary where keys are dates and values are dicts with 'time', 'actual', 'predicted'.
        method (str): Feature selection method used (e.g., 'BFS').
    """
    plt.figure(figsize=(20, 10))
    
    dates = sorted(all_predictions.keys())
    
    for i, date in enumerate(dates):
        pred_data = all_predictions[date]
        plt.plot(pred_data['time'], pred_data['actual'], 
                color='blue', alpha=0.3, label='Actual' if i == 0 else "")
        plt.plot(pred_data['time'], pred_data['predicted'], 
                color='red', alpha=0.3, label='Predicted' if i == 0 else "")
    
    plt.xlabel('Time')
    plt.ylabel('HRV (RMSSD)')
    plt.title(f'All HRV Predictions vs Actual Values ({method})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'plots/bfs-30day-plots/all_predictions_{method}.png')
    plt.close()

def plot_all_residuals(all_predictions, method):
    """
    Helper function to plot all residuals across all predicted days.
    
    Args:
        all_predictions (dict): Dictionary where keys are dates and values are dicts with 'time', 'actual', 'predicted'.
        method (str): Feature selection method used (e.g., 'BFS').
    """
    plt.figure(figsize=(20, 10))
    
    dates = sorted(all_predictions.keys())
    
    for i, date in enumerate(dates):
        pred_data = all_predictions[date]
        residuals = pred_data['actual'] - pred_data['predicted']
        plt.scatter(pred_data['time'], residuals, alpha=0.3, 
                   label=date.strftime('%Y-%m-%d') if i == 0 else "")
    
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Time')
    plt.ylabel('Residuals (Actual - Predicted)')
    plt.title(f'All Residuals ({method})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'plots/bfs-30day-plots/all_residuals_{method}.png')
    plt.close()

def plot_daily_predictions(all_predictions, method, date):
    """
    Plot predictions for a specific day with detailed analysis (actual vs predicted, residuals, residuals distribution).
    
    Args:
        all_predictions (dict): Dictionary where keys are dates and values are dicts with 'time', 'actual', 'predicted'.
        method (str): Feature selection method used (e.g., 'BFS').
        date (datetime.date): The specific date to plot.
    """
    if date not in all_predictions:
        print(f"No predictions available for {date} to plot daily analysis.")
        return
        
    pred_data = all_predictions[date]
    plt.figure(figsize=(15, 10))
    
    gs = plt.GridSpec(3, 1, height_ratios=[2, 1, 1])
    
    ax1 = plt.subplot(gs[0])
    ax1.plot(pred_data['time'], pred_data['actual'], 
             label='Actual HRV', color='blue', marker='o')
    ax1.plot(pred_data['time'], pred_data['predicted'], 
             label='Predicted HRV', color='red', linestyle='--', marker='x')
    ax1.set_title(f'HRV Predictions vs Actual Values - {date} ({method})')
    ax1.set_ylabel('HRV (RMSSD)')
    ax1.legend()
    ax1.grid(True)
    
    ax2 = plt.subplot(gs[1])
    residuals = pred_data['actual'] - pred_data['predicted']
    ax2.scatter(pred_data['time'], residuals, alpha=0.5, color='green')
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.set_title('Residuals')
    ax2.set_ylabel('Residuals (Actual - Predicted)')
    ax2.grid(True)
    
    ax3 = plt.subplot(gs[2])
    sns.histplot(residuals, kde=True, ax=ax3)
    ax3.set_title('Residuals Distribution')
    ax3.set_xlabel('Residual Value')
    ax3.set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(f'plots/bfs-30day-plots/daily_analysis_{date}_{method}.png')
    plt.close()

def plot_model_performance_metrics(all_predictions, method):
    """
    Plot comprehensive model performance metrics (R², RMSE, MAE, MAPE) over time.
    
    Args:
        all_predictions (dict): Dictionary where keys are dates and values are dicts with 'time', 'actual', 'predicted'.
        method (str): Feature selection method used (e.g., 'BFS').
    """
    if not all_predictions:
        print(f"No predictions available for {method} to plot performance metrics.")
        return
        
    plt.figure(figsize=(15, 10))
    gs = plt.GridSpec(2, 2)
    
    dates = []
    r2_scores = []
    rmse_scores = []
    mae_scores = []
    mape_scores = []
    
    for date, pred_data in all_predictions.items():
        actual = pred_data['actual']
        predicted = pred_data['predicted']
        
        r2 = r2_score(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mae = np.mean(np.abs(actual - predicted))
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100 if np.mean(actual) != 0 else np.nan
        
        dates.append(date)
        r2_scores.append(r2)
        rmse_scores.append(rmse)
        mae_scores.append(mae)
        mape_scores.append(mape)
    
    ax1 = plt.subplot(gs[0, 0])
    ax1.plot(dates, r2_scores, marker='o')
    ax1.set_title('R² Score Over Time')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('R² Score')
    ax1.grid(True)
    
    ax2 = plt.subplot(gs[0, 1])
    ax2.plot(dates, rmse_scores, marker='o', color='red')
    ax2.set_title('RMSE Over Time')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('RMSE')
    ax2.grid(True)
    
    ax3 = plt.subplot(gs[1, 0])
    ax3.plot(dates, mae_scores, marker='o', color='green')
    ax3.set_title('MAE Over Time')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('MAE')
    ax3.grid(True)
    
    ax4 = plt.subplot(gs[1, 1])
    ax4.plot(dates, mape_scores, marker='o', color='purple')
    ax4.set_title('MAPE Over Time')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('MAPE (%)')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'plots/bfs-30day-plots/performance_metrics_{method}.png')
    plt.close()

def plot_prediction_error_analysis(all_predictions, method):
    """
    Plot detailed prediction error analysis, including scatter plots and distributions.
    
    Args:
        all_predictions (dict): Dictionary where keys are dates and values are dicts with 'time', 'actual', 'predicted'.
        method (str): Feature selection method used (e.g., 'BFS').
    """
    if not all_predictions:
        print(f"No predictions available for {method} to plot error analysis.")
        return
        
    plt.figure(figsize=(15, 10))
    gs = plt.GridSpec(2, 2)
    
    all_actual = []
    all_predicted = []
    all_errors = []
    
    for pred_data in all_predictions.values():
        all_actual.extend(pred_data['actual'])
        all_predicted.extend(pred_data['predicted'])
        all_errors.extend(pred_data['actual'] - pred_data['predicted'])
    
    all_actual = np.array(all_actual)
    all_predicted = np.array(all_predicted)
    all_errors = np.array(all_errors)

    ax1 = plt.subplot(gs[0, 0])
    ax1.scatter(all_actual, all_predicted, alpha=0.5)
    ax1.plot([min(all_actual), max(all_actual)], 
             [min(all_actual), max(all_actual)], 
             'r--', label='Perfect Prediction')
    ax1.set_title('Predicted vs Actual Values')
    ax1.set_xlabel('Actual HRV')
    ax1.set_ylabel('Predicted HRV')
    ax1.legend()
    ax1.grid(True)
    
    ax2 = plt.subplot(gs[0, 1])
    sns.histplot(all_errors, kde=True, ax=ax2)
    ax2.set_title('Error Distribution')
    ax2.set_xlabel('Prediction Error')
    ax2.set_ylabel('Frequency')
    
    ax3 = plt.subplot(gs[1, 0])
    ax3.scatter(all_predicted, all_errors, alpha=0.5)
    ax3.axhline(y=0, color='r', linestyle='--')
    ax3.set_title('Error vs Predicted Value')
    ax3.set_xlabel('Predicted HRV')
    ax3.set_ylabel('Prediction Error')
    ax3.grid(True)
    
    ax4 = plt.subplot(gs[1, 1])
    ax4.scatter(all_actual, all_errors, alpha=0.5)
    ax4.axhline(y=0, color='r', linestyle='--')
    ax4.set_title('Error vs Actual Value')
    ax4.set_xlabel('Actual HRV')
    ax4.set_ylabel('Prediction Error')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'plots/bfs-30day-plots/error_analysis_{method}.png')
    plt.close()

def plot_time_series_analysis(all_predictions, method):
    """
    Plot time series analysis of actual and predicted HRV values, including rolling means.
    
    Args:
        all_predictions (dict): Dictionary where keys are dates and values are dicts with 'time', 'actual', 'predicted'.
        method (str): Feature selection method used (e.g., 'BFS').
    """
    if not all_predictions:
        print(f"No predictions available for {method} to plot time series analysis.")
        return
        
    plt.figure(figsize=(15, 10))
    gs = plt.GridSpec(2, 1, height_ratios=[2, 1])
    
    all_times = []
    all_actual = []
    all_predicted = []
    
    for pred_data in all_predictions.values():
        all_times.extend(pred_data['time'])
        all_actual.extend(pred_data['actual'])
        all_predicted.extend(pred_data['predicted'])
    
    sorted_indices = np.argsort(all_times)
    all_times = np.array(all_times)[sorted_indices]
    all_actual = np.array(all_actual)[sorted_indices]
    all_predicted = np.array(all_predicted)[sorted_indices]
    
    ax1 = plt.subplot(gs[0])
    ax1.plot(all_times, all_actual, label='Actual HRV', alpha=0.5)
    ax1.plot(all_times, all_predicted, label='Predicted HRV', alpha=0.5)
    ax1.set_title(f'HRV Time Series - {method}')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('HRV (RMSSD)')
    ax1.legend()
    ax1.grid(True)
    
    window_size = 100
    rolling_actual = pd.Series(all_actual).rolling(window=window_size).mean()
    rolling_predicted = pd.Series(all_predicted).rolling(window=window_size).mean()
    
    ax2 = plt.subplot(gs[1])
    ax2.plot(all_times, rolling_actual, label='Actual HRV (Rolling Mean)', alpha=0.7)
    ax2.plot(all_times, rolling_predicted, label='Predicted HRV (Rolling Mean)', alpha=0.7)
    ax2.set_title(f'Rolling Mean HRV (Window Size: {window_size})')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('HRV (RMSSD)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'plots/bfs-30day-plots/time_series_analysis_{method}.png')
    plt.close()

def backward_feature_selection(X_train, y_train, X_val, y_val, model):
    """
    Performs Iterative Backward Feature Selection based on R² score on a validation set.
    
    Starts with all features and iteratively removes the feature whose removal least
    degrades the R² score on the validation set. Stops when removing features
    significantly degrades performance or after a set number of non-improving steps.
    
    Args:
        X_train (pd.DataFrame): Training features data.
        y_train (pd.Series): Training target variable data.
        X_val (pd.DataFrame): Validation features data.
        y_val (pd.Series): Validation target variable data.
        model: The machine learning model instance (e.g., RandomForestRegressor).
        
    Returns:
        list: A list of the selected feature names.
    """
    selected_features = list(X_train.columns)
    best_r2 = -np.inf
    
    print("\nStarting Backward Feature Selection...")
    print(f"Initial number of features: {len(selected_features)}")
    print(f"Training data shape: {X_train.shape}")
    print(f"Validation data shape: {X_val.shape}")
    
    target_scaler = RobustScaler()
    y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()
    y_val_scaled = target_scaler.transform(y_val.values.reshape(-1, 1)).ravel()
    
    feature_scaler = RobustScaler()
    X_train_scaled = feature_scaler.fit_transform(X_train)
    X_val_scaled = feature_scaler.transform(X_val)
    
    if len(selected_features) > 0:
        model.fit(X_train_scaled, y_train_scaled)
        y_pred_val = model.predict(X_val_scaled)
        y_pred_val_original = target_scaler.inverse_transform(y_pred_val.reshape(-1, 1)).ravel()
        best_r2 = r2_score(y_val, y_pred_val_original)
        print(f"Initial R² with all features: {best_r2:.3f}")
    else:
        print("BFS: Warning: No initial features to start with!")
        return []

    no_degradation_count = 0
    max_no_degradation = 3

    while len(selected_features) > 1 and no_degradation_count < max_no_degradation:
        worst_feature = None
        largest_r2_after_removal = -np.inf
        current_best_r2 = best_r2
        degradation_this_step = False

        print(f"\nTesting removal of {len(selected_features)} features...")
        for feature in selected_features:
            features_to_test = [f for f in selected_features if f != feature]
            if not features_to_test:
                continue
            
            feature_indices = [X_train.columns.get_loc(f) for f in features_to_test]
            X_train_subset = X_train_scaled[:, feature_indices]
            X_val_subset = X_val_scaled[:, feature_indices]
            
            model.fit(X_train_subset, y_train_scaled)
            y_pred_val = model.predict(X_val_subset)
            
            y_pred_val_original = target_scaler.inverse_transform(y_pred_val.reshape(-1, 1)).ravel()
            r2_after_removal = r2_score(y_val, y_pred_val_original)

            if r2_after_removal > largest_r2_after_removal:
                largest_r2_after_removal = r2_after_removal
                worst_feature = feature

        if worst_feature and largest_r2_after_removal >= current_best_r2:
            selected_features.remove(worst_feature)
            best_r2 = largest_r2_after_removal
            no_degradation_count = 0
            print(f"BFS: Removed {worst_feature}")
            print(f"New R²: {best_r2:.3f}")
            print(f"Remaining features count: {len(selected_features)}")
        else:
            no_degradation_count += 1
            print(f"BFS: Removing feature degraded performance. No degradation count: {no_degradation_count}")

    print(f"\nBFS finished after removing {len(X_train.columns) - len(selected_features)} features")
    print(f"Final R² score: {best_r2:.3f}")
    print(f"Selected features: {selected_features}")
    return selected_features

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
    print("\nData suitability checks:")
    
    enough_points = len(X) >= min_points
    print(f"Enough data points ({len(X)} >= {min_points}): {'✓' if enough_points else '✗'}")

    no_missing_values = not X.isnull().any().any() and not y.isnull().any()
    print(f"No missing values: {'✓' if no_missing_values else '✗'}")

    sufficient_variance_target = y.std() > 1e-9
    print(f"Sufficient variance in target: {'✓' if sufficient_variance_target else '✗'}")

    features_have_variance = all(X[col].std() > 1e-9 for col in X.columns) if not X.empty else True
    print(f"Features have variance: {'✓' if features_have_variance else '✗'}")

    no_constant_features = not any(X[col].nunique() == 1 for col in X.columns) if not X.empty else True
    print(f"No constant features: {'✓' if no_constant_features else '✗'}")

    return all([enough_points, no_missing_values, sufficient_variance_target, features_have_variance, no_constant_features])

def rolling_hrv_prediction_30day_window():
    """
    Main function to perform HRV prediction using a rolling 30-day window
    and Backward Feature Selection (BFS).
    
    Loads data, processes it, performs BFS on a rolling window, trains a model,
    makes predictions, calculates metrics, and generates plots.
    """
    os.makedirs('plots/bfs-30day-plots', exist_ok=True)
    
    train_window_size = 30
    prediction_offset = 1
    validation_size = 0.2
    min_data_points_for_fs = 5
    min_data_points_per_day = 10
    min_total_points = 300
    
    prev_selected_features = None
    prev_r2 = None
    prev_rmse = None
    prev_window_data_points = None
    
    print("Loading data...")
    df = pd.read_csv('data/processed/combined_data_1min.csv', delimiter=';')
    time_column = 'minutes__minute'
    
    df[time_column] = pd.to_datetime(df[time_column], format='%d/%m/%Y %H:%M')
    df = df.set_index(time_column)
    
    target = 'night_hrv' if 'night_hrv' in df.columns else 'minutes__value__rmssd'
    print(f"Using target variable: {target}")

    print("\nEngineering new features...")
    df = df.sort_index()
    
    df['rmssd_diff_1min'] = df[target].diff(periods=1)
    if 'minutes__value' in df.columns:
        df['hr_diff_1min_new'] = df['minutes__value'].diff(periods=1)

    df['rmssd_rolling_mean_5min'] = df[target].rolling(window='5min').mean()
    df['rmssd_rolling_std_5min'] = df[target].rolling(window='5min').std()

    df = df.reset_index()
    print("\nAnalyzing data distribution before nighttime filtering:")
    print(f"Total data points: {len(df)}")
    print(f"Date range: {df[time_column].min()} to {df[time_column].max()}")
    print(f"Points per day (mean): {len(df) / len(df[time_column].dt.date.unique()):.1f}")
    
    if 'night_hrv' in df.columns:
        print("\nUsing 'night_hrv' column for nighttime filtering")
        df_night = df[df['night_hrv'] > 0].copy()
        print(f"Points with night_hrv > 0: {len(df_night)}")
    else:
        print("\nUsing time-based filtering for nighttime data (22:00 to 06:00)")
        df['hour'] = df[time_column].dt.hour
        df_night = df[((df['hour'] >= 22) | (df['hour'] < 6))].copy()
        print(f"Points between 22:00 and 06:00: {len(df_night)}")
    
    df_night = df_night.set_index(time_column)

    print("\nAnalyzing nighttime data distribution:")
    print(f"Total nighttime data points: {len(df_night)}")
    print(f"Date range: {df_night.index.min()} to {df_night.index.max()}")
    
    points_per_day = df_night.groupby(df_night.index.date).size()
    print("\nPoints per day statistics (Nighttime Data):")
    print(f"Mean points per day: {points_per_day.mean():.1f}")
    print(f"Min points per day: {points_per_day.min()}")
    print(f"Max points per day: {points_per_day.max()}")
    print(f"Days with < {min_data_points_per_day} points: {sum(points_per_day < min_data_points_per_day)}")
    
    unique_dates = sorted(df_night.index.unique().date)
    
    print(f"\nNumber of unique dates with nighttime data: {len(unique_dates)}")
    if unique_dates:
        print(f"Date range: {unique_dates[0]} to {unique_dates[-1]}")
    else:
        print("No unique dates found with nighttime data.")

    min_required_dates = train_window_size + prediction_offset
    if len(unique_dates) < min_required_dates:
        print(f"Not enough days of data ({len(unique_dates)}) for rolling window prediction with a {train_window_size}-day training window and {prediction_offset}-day offset. Need at least {min_required_dates} days.")
        print("Exiting script.")
        return
    
    numeric_columns = df_night.select_dtypes(include=[np.number]).columns
    features_to_exclude = ['id', target, 'minutes__value__rmssd', 'night_hrv', 'hour']
    features = [col for col in numeric_columns if col not in features_to_exclude]
    
    print(f"\nNumber of initial features for model training: {len(features)}")
    print("Initial Features:", features)
    
    possible_prediction_days = len(unique_dates) - train_window_size
    
    all_r2_scores = []
    all_rmse_scores = []
    all_predictions = {}
    plots_data = {}
    
    plot_indices = [0]
    if possible_prediction_days > 1:
        plot_indices.append(possible_prediction_days - 1)
    if possible_prediction_days > 2:
        plot_indices.append((possible_prediction_days - 1) // 2)
    plot_indices = sorted(list(set(plot_indices)))
    
    print(f"\nTotal possible prediction days: {possible_prediction_days}")
    print(f"Selected indices for detailed daily plotting: {plot_indices}")

    for i in range(possible_prediction_days):
        print(f"\n--- Processing window {i+1}/{possible_prediction_days} ---")
        
        train_start_idx = i
        train_end_idx = i + train_window_size
        pred_idx = train_end_idx
        
        train_dates = unique_dates[train_start_idx:train_end_idx]
        pred_date = unique_dates[pred_idx]
        
        train_data = df_night.loc[train_dates[0]:train_dates[-1]].copy()
        pred_data = df_night[df_night.index.date == pred_date].copy()
        
        train_points_per_day = train_data.groupby(train_data.index.date).size()
        current_window_data_points = len(train_data)
        
        if len(train_data) < min_total_points:
            print(f"Warning: Insufficient training data points ({len(train_data)}) for window {i+1}. Minimum required: {min_total_points}. Skipping window.")
            print(f"Days within window {i+1} with < {min_data_points_per_day} points: {sum(train_points_per_day < min_data_points_per_day)}")
            all_r2_scores.append(np.nan)
            all_rmse_scores.append(np.nan)
            continue

        if len(pred_data) < min_data_points_per_day:
            print(f"Warning: Insufficient prediction data points ({len(pred_data)}) for date {pred_date} (window {i+1}). Minimum required: {min_data_points_per_day}. Skipping window.")
            all_r2_scores.append(np.nan)
            all_rmse_scores.append(np.nan)
            continue

        print("Handling missing values...")
        median_vals = train_data[features + [target]].median()
        train_data = train_data.fillna(median_vals)
        pred_data = pred_data.fillna(median_vals)

        current_features = [f for f in features if f in train_data.columns and f in pred_data.columns]
        
        X_train_full = train_data[current_features].copy()
        y_train_full = train_data[target].copy()
        X_pred = pred_data[current_features].copy()
        y_actual = pred_data[target].copy()

        valid_train_indices = y_train_full.dropna().index
        X_train_full = X_train_full.loc[valid_train_indices]
        y_train_full = y_train_full.loc[valid_train_indices]
        
        if X_train_full.empty or y_train_full.empty or X_pred.empty or y_actual.empty:
            print(f"Insufficient valid data points after processing for window {i+1}. Skipping.")
            all_r2_scores.append(np.nan)
            all_rmse_scores.append(np.nan)
            continue

        if y_train_full.std() == 0:
            print(f"Target variable has zero variance in training window {i+1}. Skipping model training.")
            all_r2_scores.append(np.nan)
            all_rmse_scores.append(np.nan)
            continue

        print(f"\nPreparing for feature selection for window {i+1}...")
        
        if not check_data_suitability(X_train_full, y_train_full, min_data_points_for_fs):
             print(f"Warning: Full training data for window {i+1} not suitable for feature selection. Using all current features.")
             selected_features = current_features
        else:
            try:
                X_train_fs, X_val_fs, y_train_fs, y_val_fs = train_test_split(
                    X_train_full, y_train_full, 
                    test_size=validation_size, 
                    random_state=42,
                    shuffle=False
                )
                print(f"Feature selection split shapes: X_train_fs={X_train_fs.shape}, X_val_fs={X_val_fs.shape}")
                
                if not check_data_suitability(X_train_fs, y_train_fs, min_data_points_for_fs) or \
                   not check_data_suitability(X_val_fs, y_val_fs, 1):
                    print(f"Warning: Split training or validation data for window {i+1} not suitable for feature selection. Using all current features.")
                    selected_features = current_features
                else:
                    print("Data passed suitability checks after split. Proceeding with feature selection.")
                    fs_model = RandomForestRegressor(
                        n_estimators=50,
                        max_depth=10,
                        min_samples_split=5,
                        min_samples_leaf=2,
                        random_state=42
                    )
                    selected_features = backward_feature_selection(X_train_fs, y_train_fs, X_val_fs, y_val_fs, fs_model)
                    
            except ValueError as e:
                print(f"Could not split training data for feature selection in window {i+1}: {e}. Using all current features.")
                selected_features = current_features

        if (prev_selected_features == selected_features and 
            prev_window_data_points == current_window_data_points):
            print(f"Window {i+1}: Results identical to previous window - skipping detailed window logging.")
        else:
            print(f"\n--- Detailed Window Processing for {i+1}/{possible_prediction_days} ---")
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

        r2 = np.nan
        rmse = np.nan

        if selected_features:
            if (prev_selected_features != selected_features or 
                prev_r2 != r2 or
                prev_rmse != rmse):
                print(f"\nTraining final model with {len(selected_features)} selected features for window {i+1}...")
            
            X_train_final = X_train_full[selected_features]
            y_train_final = y_train_full
            X_pred_final = X_pred[selected_features]

            feature_scaler_final = RobustScaler()
            X_train_scaled = feature_scaler_final.fit_transform(X_train_final)
            X_pred_scaled = feature_scaler_final.transform(X_pred_final)
            
            target_scaler_final = RobustScaler()
            y_train_scaled = target_scaler_final.fit_transform(y_train_final.values.reshape(-1, 1)).ravel()
            
            model = RandomForestRegressor(
                n_estimators=50,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
            model.fit(X_train_scaled, y_train_scaled)
            
            y_pred_scaled = model.predict(X_pred_scaled)
            y_pred = target_scaler_final.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

            if len(y_actual) > 0 and len(y_pred) == len(y_actual):
                r2 = r2_score(y_actual, y_pred)
                rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
                
                if (prev_r2 != r2 or prev_rmse != rmse):
                    print(f"Window {i+1} Metrics: R² Score: {r2:.3f}, RMSE: {rmse:.3f}")
            else:
                 print(f"Could not calculate metrics for window {i+1}. y_actual length: {len(y_actual)}, y_pred length: {len(y_pred)}")
                 r2 = np.nan
                 rmse = np.nan

            all_r2_scores.append(r2)
            all_rmse_scores.append(rmse)
        else:
            print(f"No features selected for window {i+1}. Cannot train model.")
            all_r2_scores.append(np.nan)
            all_rmse_scores.append(np.nan)
        
        if not np.isnan(r2):
            all_predictions[pred_date] = {
                'time': pred_data.index.tolist(),
                'actual': y_actual.values.tolist(),
                'predicted': y_pred.tolist()
            }
        
        if i in plot_indices:
            if not np.isnan(r2):
                plots_data[i] = {
                    'actual': pd.DataFrame({'time': y_actual.index, 'actual': y_actual.values}),
                    'predicted': pd.DataFrame({'time': pred_data.index, 'predicted': y_pred}),
                    'date_str': pred_date.strftime('%Y-%m-%d')
                }
            else:
                print(f"No valid data or metrics available for plotting for day index {i+1}.")
                plots_data[i] = None

        if i == 0:
            feature_importance = None
            if 'model' in locals() and selected_features:
                feature_importance = pd.DataFrame({
                    'feature': selected_features,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
            else:
                 print("Model not trained in the first window, cannot get feature importance.")

        prev_selected_features = selected_features.copy() if selected_features else None
        prev_r2 = r2
        prev_rmse = rmse
        prev_window_data_points = current_window_data_points

    filtered_results = [(r2, rmse) for r2, rmse in zip(all_r2_scores, all_rmse_scores) if not np.isnan(r2) and r2 > -10.0]
    
    print("\nGenerating plots...")
    
    for idx in plot_indices:
        if idx in plots_data and plots_data[idx] is not None:
            plot_predictions(plots_data[idx]['actual'], plots_data[idx]['predicted'], 
                           plots_data[idx]['date_str'], f'predictions_day_{idx+1}_30day_window_new_features_bfs.png')
            plot_residuals(plots_data[idx]['actual'], plots_data[idx]['predicted'],
                         plots_data[idx]['date_str'], f'residuals_day_{idx+1}_30day_window_new_features_bfs.png')
        else:
            print(f"Skipping daily plots for day index {idx+1} due to no data or metrics available.")

    if 'feature_importance' in locals() and feature_importance is not None and not feature_importance.empty:
        plt.figure(figsize=(12, 6))
        sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
        plt.title('Top 10 Most Important Features (First Window - BFS)')
        plt.tight_layout()
        plt.savefig('plots/bfs-30day-plots/feature_importance_rolling_30day_window_new_features_bfs.png')
        plt.close()
    else:
         print("Not enough data or model not trained in first window to plot feature importance.")

    if all_predictions:
        print("Generating overall prediction and performance plots...")
        plot_all_predictions(all_predictions, 'BFS')
        plot_all_residuals(all_predictions, 'BFS')
        plot_model_performance_metrics(all_predictions, 'BFS')
        plot_prediction_error_analysis(all_predictions, 'BFS')
        plot_time_series_analysis(all_predictions, 'BFS')
    else:
        print("No predictions available to generate overall plots.")

    print("\n--- Overall Model Performance (Outliers Excluded - BFS) ---")

    if filtered_results:
        filtered_r2_scores = [res[0] for res in filtered_results]
        filtered_rmse_scores = [res[1] for res in filtered_results]
        
        total_data_points = sum(len(all_predictions[date]['actual']) for date in all_predictions) if all_predictions else 0
        
        print(f"\nNumber of prediction days included in analysis: {len(filtered_r2_scores)}")
        print(f"Total number of data points (measurements) across all included days: {total_data_points}")
        print(f"Average data points per day (included days): {total_data_points/len(filtered_r2_scores):.1f}" if len(filtered_r2_scores) > 0 else "Average data points per day: N/A")
        
        print(f"Average R² Score: {np.mean(filtered_r2_scores):.3f} ± {np.std(filtered_r2_scores):.3f}")
        print(f"Average RMSE: {np.mean(filtered_rmse_scores):.3f} ± {np.std(filtered_rmse_scores):.3f}")

        plt.figure(figsize=(10, 6))
        valid_indices = [i for i, r2 in enumerate(all_r2_scores) if not np.isnan(r2)]
        valid_r2_scores = [r2 for r2 in all_r2_scores if not np.isnan(r2)]
        filtered_indices = [i for i, r2 in zip(valid_indices, valid_r2_scores) if r2 > -10.0]
        filtered_r2_scores_plot = [r2 for r2 in valid_r2_scores if r2 > -10.0]
        
        plt.plot(valid_indices, valid_r2_scores, marker='o', linestyle='-', color='gray', alpha=0.5, label='All Valid R² Scores')
        plt.plot(filtered_indices, filtered_r2_scores_plot, marker='o', linestyle='-', color='blue', label='Filtered R² Scores')
        plt.axhline(y=np.mean(filtered_r2_scores), color='r', linestyle='--', 
                    label=f'Mean Filtered R²: {np.mean(filtered_r2_scores):.3f}')
        plt.xlabel('Prediction Day Index')
        plt.ylabel('R² Score')
        plt.title('R² Scores Over Time (Outliers Excluded from Average - BFS)')
        plt.legend()
        plt.tight_layout()
        plt.savefig('plots/bfs-30day-plots/r2_scores_over_time_filtered_30day_window_new_features_bfs.png')
        plt.close()

    else:
        print("\nNo R² scores met the filtering criteria for overall performance calculation.")

if __name__ == "__main__":
    rolling_hrv_prediction_30day_window() 