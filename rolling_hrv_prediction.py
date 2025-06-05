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
    """Remove outliers based on standard deviation"""
    df_clean = df.copy()
    for col in columns:
        mean = df_clean[col].mean()
        std = df_clean[col].std()
        df_clean = df_clean[abs(df_clean[col] - mean) <= (n_std * std)]
    return df_clean

def plot_predictions(actual_df, predicted_df, date_str, filename):
    """Helper function to plot actual vs predicted HRV for a given day"""
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
    plt.savefig(f'plots/{filename}')
    plt.close()

def plot_residuals(actual_df, predicted_df, date_str, filename):
    """Helper function to plot residuals for a given day"""
    residuals = actual_df['actual'] - predicted_df['predicted']
    plt.figure(figsize=(15, 6))
    plt.scatter(predicted_df['time'], residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Time')
    plt.ylabel('Residuals (Actual - Predicted)')
    plt.title(f'Residuals - {date_str}')
    plt.tight_layout()
    plt.savefig(f'plots/{filename}')
    plt.close()

def forward_feature_selection(X_train, y_train, X_val, y_val, model):
    """Iterative Forward Feature Selection based on R2 on validation set"""
    selected_features = []
    remaining_features = list(X_train.columns)
    best_r2 = -np.inf
    no_improvement_count = 0
    max_no_improvement = 3 # Stop if no improvement for 3 steps

    print("Starting Forward Feature Selection...")
    while remaining_features and no_improvement_count < max_no_improvement:
        improvement_this_step = False
        best_feature = None
        current_best_r2 = best_r2

        for feature in remaining_features:
            features_to_test = selected_features + [feature]
            model.fit(X_train[features_to_test], y_train)
            y_pred_val = model.predict(X_val[features_to_test])
            r2 = r2_score(y_val, y_pred_val)

            if r2 > current_best_r2:
                current_best_r2 = r2
                best_feature = feature
                improvement_this_step = True

        if best_feature and current_best_r2 > best_r2:
            selected_features.append(best_feature)
            remaining_features.remove(best_feature)
            best_r2 = current_best_r2
            no_improvement_count = 0 # Reset counter on improvement
            print(f"FFS: Added {best_feature}, new best R2: {best_r2:.3f}, selected features count: {len(selected_features)}")
        else:
            no_improvement_count += 1
            print(f"FFS: No improvement in this step. No improvement count: {no_improvement_count}")

    print(f"FFS finished. Selected features ({len(selected_features)}): {selected_features}")
    return selected_features

def backward_feature_selection(X_train, y_train, X_val, y_val, model):
    """Iterative Backward Feature Selection based on R2 on validation set"""
    selected_features = list(X_train.columns)
    best_r2 = -np.inf # Initialize with negative infinity
    
    # Train initial model with all features to get a baseline R2
    if len(selected_features) > 0:
        model.fit(X_train[selected_features], y_train)
        y_pred_val = model.predict(X_val[selected_features])
        best_r2 = r2_score(y_val, y_pred_val)
    else:
        print("BFS: Warning: No initial features to start with!")
        return []


    no_degradation_count = 0
    max_no_degradation = 3 # Stop if removing feature degrades performance significantly or no improvement

    print("Starting Backward Feature Selection...")
    while len(selected_features) > 1 and no_degradation_count < max_no_degradation:
        worst_feature = None
        largest_r2_after_removal = -np.inf
        current_best_r2 = best_r2
        degradation_this_step = False

        for feature in selected_features:
            features_to_test = [f for f in selected_features if f != feature]
            if not features_to_test:
                 continue # Should not happen if len(selected_features) > 1
            model.fit(X_train[features_to_test], y_train)
            y_pred_val = model.predict(X_val[features_to_test])
            r2_after_removal = r2_score(y_val, y_pred_val)

            # We are looking for the feature whose removal results in the HIGHEST R2 (least degradation or improvement)
            if r2_after_removal > largest_r2_after_removal:
                 largest_r2_after_removal = r2_after_removal
                 worst_feature = feature

        # Check if removing the worst feature improves performance or doesn't degrade it too much
        if worst_feature and largest_r2_after_removal >= current_best_r2:
            selected_features.remove(worst_feature)
            best_r2 = largest_r2_after_removal # Update best_r2 to the R2 after removing the worst feature
            no_degradation_count = 0 # Reset counter
            print(f"BFS: Removed {worst_feature}, new R2: {best_r2:.3f}, selected features count: {len(selected_features)}")
        else:
            no_degradation_count += 1
            print(f"BFS: Removing feature degraded performance. No degradation count: {no_degradation_count}")

    print(f"BFS finished. Selected features ({len(selected_features)}): {selected_features}")
    return selected_features

def rolling_hrv_prediction():
    # Define the window size
    train_window_size = 7 # days (original was 7)
    prediction_offset = 1 # days after training window
    validation_size = 0.2 # 20% of training data for validation feature selection
    
    # Read the combined data
    print("Loading data...")
    df = pd.read_csv('data/processed/combined_data_1min.csv', delimiter=';')
    time_column = 'minutes__minute'
    
    # Convert the time column to datetime objects
    df[time_column] = pd.to_datetime(df[time_column], format='%d/%m/%Y %H:%M')
    
    # Determine the target variable
    target = 'night_hrv' if 'night_hrv' in df.columns else 'minutes__value__rmssd'
    print(f"Using target variable: {target}")
    
    # Filter for nighttime data
    if 'night_hrv' in df.columns:
        df_night = df[df['night_hrv'] > 0].copy()
    else:
        df['hour'] = df[time_column].dt.hour
        df_night = df[((df['hour'] >= 22) | (df['hour'] < 6))].copy()
    
    print(f"\nTotal data points: {len(df)}")
    print(f"Nighttime data points: {len(df_night)}")
    
    # Get unique dates in the dataset
    unique_dates = sorted(df_night[time_column].dt.date.unique())
    print(f"\nNumber of unique dates: {len(unique_dates)}")
    print(f"Date range: {unique_dates[0]} to {unique_dates[-1]}")
    
    # Calculate minimum required dates
    min_required_dates = train_window_size + prediction_offset
    if len(unique_dates) < min_required_dates:
        print(f"Not enough days of data ({len(unique_dates)}) for rolling window prediction with a {train_window_size}-day training window and {prediction_offset}-day offset. Need at least {min_required_dates} days.")
        return
    
    # Prepare features
    numeric_columns = df_night.select_dtypes(include=[np.number]).columns
    features_to_exclude = ['id', target, 'minutes__value__rmssd', 'hour'] # Exclude hour
    features = [col for col in numeric_columns if col not in features_to_exclude]
    
    print(f"\nNumber of initial features: {len(features)}")
    print("Initial Features:", features)
    
    # Initialize lists to store results for both FFS and BFS
    all_r2_scores_ffs = []
    all_rmse_scores_ffs = []
    all_r2_scores_bfs = []
    all_rmse_scores_bfs = []
    
    # Store data for plotting specific days
    plots_data = {}
    # Calculate possible prediction days
    possible_prediction_days = len(unique_dates) - train_window_size
    
    plot_indices = [0] # First prediction day index
    if possible_prediction_days > 1:
        plot_indices.append(possible_prediction_days - 1) # Last prediction day index
    if possible_prediction_days > 2:
        plot_indices.append((possible_prediction_days - 1) // 2) # Middle prediction day index
    plot_indices = sorted(list(set(plot_indices))) # Ensure unique and sorted
    
    print(f"\nTotal possible prediction days: {possible_prediction_days}")
    print(f"Plotting indices: {plot_indices}")

    # Rolling window prediction
    for i in range(possible_prediction_days):
        print(f"\n---\nProcessing window {i+1}/{possible_prediction_days}")
        
        # Get training window dates
        train_dates = unique_dates[i:i+train_window_size]
        # Get prediction date
        pred_date = unique_dates[i+train_window_size]
        
        print(f"Training on dates: {train_dates[0]} to {train_dates[-1]}")
        print(f"Predicting for date: {pred_date}")
        
        # Prepare training data
        train_data = df_night[df_night[time_column].dt.date.isin(train_dates)].copy()
        pred_data = df_night[df_night[time_column].dt.date == pred_date].copy()
        
        # If pred_data is a single row (only one data point for the day), convert to DataFrame
        # No need to transpose here as it's filtered by date, not a single index lookup

        # Handle missing values BEFORE splitting for feature selection
        print("Handling missing values...")
        for col in features + [target]:
            if col in train_data.columns:
                # Impute training data missing values with the median of the training data
                median_val = train_data[col].median()
                train_data[col] = train_data[col].fillna(median_val)
            # Use training data median for imputation in prediction data to avoid data leakage
            if col in pred_data.columns:
                 # Ensure median_val exists (e.g., if train_data for a column is empty)
                 if not np.isnan(median_val):
                     pred_data[col] = pred_data[col].fillna(median_val)
                 else:
                     # If median_val is NaN, use 0 or another strategy
                     pred_data[col] = pred_data[col].fillna(0) # Fallback to 0 imputation
        
        print(f"Training data points: {len(train_data)}")
        print(f"Prediction data points: {len(pred_data)}")
        
        # Prepare X and y for current window
        current_features = [f for f in features if f in train_data.columns and f in pred_data.columns]
        
        X_train_full = train_data[current_features]
        y_train_full = train_data[target]
        X_pred = pred_data[current_features]
        y_actual = pred_data[target]

        # Drop rows with NaN in target after imputation, just in case
        train_data.dropna(subset=[target], inplace=True)
        # Re-select X_train_full and y_train_full after dropping NaNs
        X_train_full = train_data[current_features]
        y_train_full = train_data[target]
        
        # Check if training or prediction data is empty after processing
        if X_train_full.empty or y_train_full.empty or X_pred.empty or y_actual.empty or len(y_actual) == 0:
            print(f"Insufficient valid data points for window {i+1}. Skipping.")
            all_r2_scores_ffs.append(np.nan)
            all_rmse_scores_ffs.append(np.nan)
            all_r2_scores_bfs.append(np.nan)
            all_rmse_scores_bfs.append(np.nan)
            continue # Skip to the next window

        # Check if y_train_full has sufficient variance for scaling and model training
        if y_train_full.std() == 0:
            print(f"Target variable has zero variance in training window {i+1}. Skipping model training.")
            all_r2_scores_ffs.append(np.nan)
            all_rmse_scores_ffs.append(np.nan)
            all_r2_scores_bfs.append(np.nan)
            all_rmse_scores_bfs.append(np.nan)
            continue

        # Split training data into training and validation sets for feature selection
        print(f"Splitting training data for feature selection (validation size: {validation_size})...")
        try:
            X_train_fs, X_val_fs, y_train_fs, y_val_fs = train_test_split(X_train_full, y_train_full, test_size=validation_size, random_state=42, shuffle=False) # Use shuffle=False to maintain time order
            print(f"X_train_fs shape: {X_train_fs.shape}, X_val_fs shape: {X_val_fs.shape}")
        except ValueError as e:
            print(f"Could not split training data for feature selection in window {i+1}: {e}. Skipping feature selection.")
            # Proceed without feature selection for this window, using all available features
            selected_features_ffs = current_features
            selected_features_bfs = current_features
            print("Using all features for this window due to split error.")
            # Need to ensure train/val split variables are defined even if we skip try block
            X_train_fs = X_train_full
            y_train_fs = y_train_full
            X_val_fs = pd.DataFrame(columns=current_features) # Empty validation set
            y_val_fs = pd.Series(dtype=y_train_full.dtype) # Empty validation set

        # Define the base model for feature selection and final prediction
        base_model = RandomForestRegressor(
            n_estimators=50,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )

        # Perform Feature Selection (only if split was successful and data is sufficient)
        if not X_train_fs.empty and not X_val_fs.empty and not y_train_fs.empty and not y_val_fs.empty:
            # Scale data for feature selection models
            feature_scaler_fs = RobustScaler()
            X_train_fs_scaled = feature_scaler_fs.fit_transform(X_train_fs)
            X_val_fs_scaled = feature_scaler_fs.transform(X_val_fs)
            y_train_fs_scaled = RobustScaler().fit_transform(y_train_fs.values.reshape(-1, 1)).ravel()
            y_val_fs_scaled = RobustScaler().fit_transform(y_val_fs.values.reshape(-1, 1)).ravel() # Scale validation target too

            # Use scaled dataframes for FS functions
            X_train_fs_scaled_df = pd.DataFrame(X_train_fs_scaled, columns=X_train_fs.columns, index=X_train_fs.index)
            X_val_fs_scaled_df = pd.DataFrame(X_val_fs_scaled, columns=X_val_fs.columns, index=X_val_fs.index)
            y_train_fs_scaled_series = pd.Series(y_train_fs_scaled, index=y_train_fs.index, name=target)
            y_val_fs_scaled_series = pd.Series(y_val_fs_scaled, index=y_val_fs.index, name=target)

            # Create a fresh model instance for FS to avoid carry-over states
            fs_model = RandomForestRegressor(
                n_estimators=50,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )

            selected_features_ffs = forward_feature_selection(X_train_fs_scaled_df, y_train_fs_scaled_series, X_val_fs_scaled_df, y_val_fs_scaled_series, fs_model)

            # Create a fresh model instance for BFS
            fs_model = RandomForestRegressor(
                n_estimators=50,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
            selected_features_bfs = backward_feature_selection(X_train_fs_scaled_df, y_train_fs_scaled_series, X_val_fs_scaled_df, y_val_fs_scaled_series, fs_model)
        else:
             print("Insufficient data after train/val split for feature selection. Using all features.")
             selected_features_ffs = current_features
             selected_features_bfs = current_features

        # Scale features for final model training and prediction (using the full training window)
        # Use separate scalers for FFS and BFS features to avoid leakage if feature sets are different
        
        r2_ffs = np.nan
        rmse_ffs = np.nan
        r2_bfs = np.nan
        rmse_bfs = np.nan

        # --- Train and Evaluate with FFS features ---
        if selected_features_ffs:
            print("\nTraining model with FFS selected features...")
            feature_scaler_ffs = RobustScaler()
            X_train_scaled_ffs = feature_scaler_ffs.fit_transform(X_train_full[selected_features_ffs])
            X_pred_scaled_ffs = feature_scaler_ffs.transform(X_pred[selected_features_ffs])
            
            target_scaler_ffs = RobustScaler()
            y_train_scaled_ffs = target_scaler_ffs.fit_transform(y_train_full.values.reshape(-1, 1)).ravel()
            
            model_ffs = RandomForestRegressor(
                n_estimators=50,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
            model_ffs.fit(X_train_scaled_ffs, y_train_scaled_ffs)
            y_pred_scaled_ffs = model_ffs.predict(X_pred_scaled_ffs)
            y_pred_ffs = target_scaler_ffs.inverse_transform(y_pred_scaled_ffs.reshape(-1, 1)).ravel()

            if len(y_actual) > 0 and len(y_pred_ffs) == len(y_actual):
                r2_ffs = r2_score(y_actual, y_pred_ffs)
                rmse_ffs = np.sqrt(mean_squared_error(y_actual, y_pred_ffs))
                print(f"FFS Features: {selected_features_ffs}")
                print(f"FFS R² Score: {r2_ffs:.3f}, RMSE: {rmse_ffs:.3f}")
            else:
                 print("Could not calculate FFS metrics for this window.")
            all_r2_scores_ffs.append(r2_ffs)
            all_rmse_scores_ffs.append(rmse_ffs)
        else:
            print("No features selected by FFS for this window. Cannot train FFS model.")
            all_r2_scores_ffs.append(np.nan)
            all_rmse_scores_ffs.append(np.nan)

        # --- Train and Evaluate with BFS features ---
        if selected_features_bfs:
            print("\nTraining model with BFS selected features...")
            feature_scaler_bfs = RobustScaler()
            X_train_scaled_bfs = feature_scaler_bfs.fit_transform(X_train_full[selected_features_bfs])
            X_pred_scaled_bfs = feature_scaler_bfs.transform(X_pred[selected_features_bfs])
            
            target_scaler_bfs = RobustScaler()
            y_train_scaled_bfs = target_scaler_bfs.fit_transform(y_train_full.values.reshape(-1, 1)).ravel()
            
            model_bfs = RandomForestRegressor(
                n_estimators=50,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
            model_bfs.fit(X_train_scaled_bfs, y_train_scaled_bfs)
            y_pred_scaled_bfs = model_bfs.predict(X_pred_scaled_bfs)
            y_pred_bfs = target_scaler_bfs.inverse_transform(y_pred_scaled_bfs.reshape(-1, 1)).ravel()

            if len(y_actual) > 0 and len(y_pred_bfs) == len(y_actual):
                r2_bfs = r2_score(y_actual, y_pred_bfs)
                rmse_bfs = np.sqrt(mean_squared_error(y_actual, y_pred_bfs))
                print(f"BFS Features: {selected_features_bfs}")
                print(f"BFS R² Score: {r2_bfs:.3f}, RMSE: {rmse_bfs:.3f}")
            else:
                 print("Could not calculate BFS metrics for this window.")
            all_r2_scores_bfs.append(np.nan)
            all_rmse_scores_bfs.append(rmse_bfs)

        else:
             print("No features selected by BFS for this window. Cannot train BFS model.")
             all_r2_scores_bfs.append(np.nan)
             all_rmse_scores_bfs.append(np.nan)

        
        # Store data for plotting if this is one of the selected indices
        # Decide which model's predictions to plot (e.g., the one with better R2 for this day)
        if i in plot_indices:
            if not np.isnan(r2_ffs) and (np.isnan(r2_bfs) or r2_ffs >= r2_bfs):
                 # Plot FFS results
                 plots_data[i] = {
                    'actual': pd.DataFrame({'time': pred_data[time_column], 'actual': y_actual}),
                    'predicted': pd.DataFrame({'time': pred_data[time_column], 'predicted': y_pred_ffs}),
                    'date_str': pred_date.strftime('%Y-%m-%d'),
                    'method': 'FFS'
                }
            elif not np.isnan(r2_bfs):
                 # Plot BFS results
                 plots_data[i] = {
                    'actual': pd.DataFrame({'time': pred_data[time_column], 'actual': y_actual}),
                    'predicted': pd.DataFrame({'time': pred_data[time_column], 'predicted': y_pred_bfs}),
                    'date_str': pred_date.strftime('%Y-%m-%d'),
                    'method': 'BFS'
                }
            else:
                print(f"No valid metrics for plotting for day index {i+1}.")
                plots_data[i] = None # Store None if no data to plot

        # Store feature importance for the first model (index 0) - We'll store both FFS and BFS feature importances
        if i == 0:
            feature_importance_ffs = None
            feature_importance_bfs = None
            # Ensure the models were trained before trying to get feature importances
            if 'model_ffs' in locals() and selected_features_ffs:
                feature_importance_ffs = pd.DataFrame({
                    'feature': selected_features_ffs,
                    'importance': model_ffs.feature_importances_
                }).sort_values('importance', ascending=False)
            else:
                 print("FFS model not trained in the first window, cannot get feature importance.")

            if 'model_bfs' in locals() and selected_features_bfs:
                 feature_importance_bfs = pd.DataFrame({
                    'feature': selected_features_bfs,
                    'importance': model_bfs.feature_importances_
                }).sort_values('importance', ascending=False)
            else:
                 print("BFS model not trained in the first window, cannot get feature importance.")

    # --- Post-processing and Plotting ---

    # Filter out outlier R² scores and corresponding RMSEs for both methods
    # Define a threshold for R² scores. R² can be negative when predictions are poor.
    # A threshold of -10 seems reasonable to remove extremely bad predictions.
    r2_threshold = -10
    
    filtered_results_ffs = [(r2, rmse) for r2, rmse in zip(all_r2_scores_ffs, all_rmse_scores_ffs) if not np.isnan(r2) and r2 > r2_threshold]
    filtered_results_bfs = [(r2, rmse) for r2, rmse in zip(all_r2_scores_bfs, all_rmse_scores_bfs) if not np.isnan(r2) and r2 > r2_threshold]
    
    # Create plots directory
    os.makedirs('plots', exist_ok=True)

    # Plot actual vs predicted and residuals for selected days
    for idx in plot_indices:
        if idx in plots_data and plots_data[idx] is not None:
            plot_predictions(plots_data[idx]['actual'], plots_data[idx]['predicted'], 
                             plots_data[idx]['date_str'], f'predictions_day_{idx+1}_{plots_data[idx]['method']}.png')
            plot_residuals(plots_data[idx]['actual'], plots_data[idx]['predicted'],
                           plots_data[idx]['date_str'], f'residuals_day_{idx+1}_{plots_data[idx]['method']}.png')
        else:
             print(f"No data or metrics available for plotting for day index {idx+1}.")

    # Plot feature importance (only if data exists) - Plot for both FFS and BFS if available
    if 'feature_importance_ffs' in locals() and feature_importance_ffs is not None and not feature_importance_ffs.empty:
        plt.figure(figsize=(12, 6))
        sns.barplot(data=feature_importance_ffs.head(10), x='importance', y='feature')
        plt.title('Top 10 Most Important Features (First Window FFS - 7 day window)')
        plt.tight_layout()
        plt.savefig('plots/feature_importance_rolling_ffs.png')
        plt.close()
    else:
         print("Not enough data or FFS model not trained in first window to plot feature importance.")

    if 'feature_importance_bfs' in locals() and feature_importance_bfs is not None and not feature_importance_bfs.empty:
        plt.figure(figsize=(12, 6))
        sns.barplot(data=feature_importance_bfs.head(10), x='importance', y='feature')
        plt.title('Top 10 Most Important Features (First Window BFS - 7 day window)')
        plt.tight_layout()
        plt.savefig('plots/feature_importance_rolling_bfs.png')
        plt.close()
    else:
         print("Not enough data or BFS model not trained in first window to plot feature importance.")


    # Print overall performance metrics and compare
    print("\n--- Overall Model Performance (Outliers Excluded - 7 day window) ---")

    if filtered_results_ffs:
        filtered_r2_scores_ffs = [res[0] for res in filtered_results_ffs]
        filtered_rmse_scores_ffs = [res[1] for res in filtered_results_ffs]
        print("\nForward Feature Selection:")
        print(f"Number of prediction days included: {len(filtered_r2_scores_ffs)}")
        print(f"Average R² Score: {np.mean(filtered_r2_scores_ffs):.3f} ± {np.std(filtered_r2_scores_ffs):.3f}")
        print(f"Average RMSE: {np.mean(filtered_rmse_scores_ffs):.3f} ± {np.std(filtered_rmse_scores_ffs):.3f}")

        # Plot FFS R² scores over time
        plt.figure(figsize=(10, 6))
        original_indices_ffs = [i for i, r2 in enumerate(all_r2_scores_ffs) if not np.isnan(r2)]
        filtered_indices_for_plot_ffs = [original_indices_ffs[i] for i, r2 in enumerate(all_r2_scores_ffs) if not np.isnan(r2) and r2 > r2_threshold]
        
        plt.plot(range(len(all_r2_scores_ffs)), all_r2_scores_ffs, marker='o', linestyle='-', color='gray', alpha=0.5, label='All FFS R² Scores')
        plt.plot(filtered_indices_for_plot_ffs, filtered_r2_scores_ffs, marker='o', linestyle='-', color='blue', label='Filtered FFS R² Scores')
        plt.axhline(y=np.mean(filtered_r2_scores_ffs), color='r', linestyle='--', 
                    label=f'Mean Filtered FFS R²: {np.mean(filtered_r2_scores_ffs):.3f}')
        plt.xlabel('Prediction Day Index')
        plt.ylabel('R² Score')
        plt.title('FFS R² Scores Over Time (Outliers Excluded from Average - 7 day window)')
        plt.legend()
        plt.tight_layout()
        plt.savefig('plots/r2_scores_over_time_filtered_ffs.png')
        plt.close()

    else:
        print("\nNo FFS R² scores met the filtering criteria.")

    if filtered_results_bfs:
        filtered_r2_scores_bfs = [res[0] for res in filtered_results_bfs]
        filtered_rmse_scores_bfs = [res[1] for res in filtered_results_bfs]
        print("\nBackward Feature Selection:")
        print(f"Number of prediction days included: {len(filtered_r2_scores_bfs)}")
        print(f"Average R² Score: {np.mean(filtered_r2_scores_bfs):.3f} ± {np.std(filtered_r2_scores_bfs):.3f}")
        print(f"Average RMSE: {np.mean(filtered_rmse_scores_bfs):.3f} ± {np.std(filtered_rmse_scores_bfs):.3f}")

        # Plot BFS R² scores over time
        plt.figure(figsize=(10, 6))
        original_indices_bfs = [i for i, r2 in enumerate(all_r2_scores_bfs) if not np.isnan(r2)]
        filtered_indices_for_plot_bfs = [original_indices_bfs[i] for i, r2 in enumerate(all_r2_scores_bfs) if not np.isnan(r2) and r2 > r2_threshold]
        
        plt.plot(range(len(all_r2_scores_bfs)), all_r2_scores_bfs, marker='o', linestyle='-', color='gray', alpha=0.5, label='All BFS R² Scores')
        plt.plot(filtered_indices_for_plot_bfs, filtered_r2_scores_bfs, marker='o', linestyle='-', color='green', label='Filtered BFS R² Scores')
        plt.axhline(y=np.mean(filtered_r2_scores_bfs), color='purple', linestyle='--', 
                    label=f'Mean Filtered BFS R²: {np.mean(filtered_r2_scores_bfs):.3f}')
        plt.xlabel('Prediction Day Index')
        plt.ylabel('R² Score')
        plt.title('BFS R² Scores Over Time (Outliers Excluded from Average - 7 day window)')
        plt.legend()
        plt.tight_layout()
        plt.savefig('plots/r2_scores_over_time_filtered_bfs.png')
        plt.close()

    else:
        print("\nNo BFS R² scores met the filtering criteria.")


    # Compare overall average R² scores
    print("\n--- Feature Selection Method Comparison ---")
    mean_r2_ffs = np.mean(filtered_r2_scores_ffs) if filtered_results_ffs else -np.inf
    mean_r2_bfs = np.mean(filtered_r2_scores_bfs) if filtered_results_bfs else -np.inf

    if mean_r2_ffs > mean_r2_bfs:
        print("Forward Feature Selection resulted in a better average R².")
    elif mean_r2_bfs > mean_r2_ffs:
        print("Backward Feature Selection resulted in a better average R².")
    else:
        print("Both feature selection methods resulted in similar average R² scores or no valid scores were obtained.")


if __name__ == "__main__":
    rolling_hrv_prediction() 