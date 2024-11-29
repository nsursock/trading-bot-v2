import logging
import shap
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import argparse

from parameters import selected_params, financial_params, constant_params, training_params

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeRegressor

# def extract_features_from_ppo(model, env):
#     # Get the dimension of the flattened observation space
#     obs_dim = get_flattened_obs_dim(env.observation_space)
    
#     # Extract the weights from the first layer of the policy network
#     weights = model.policy.mlp_extractor.policy_net[0].weight.detach().numpy()
    
#     # Calculate feature importance as the sum of absolute weights for each feature
#     feature_importance = np.sum(np.abs(weights), axis=0)
    
#     return feature_importance

def calculate_feature_importance(X, y, model, env=None, feature_names=None, method='shap'):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if isinstance(model, PPO):
        # Extract feature importance from PPO model
        # ppo_feature_importance = extract_features_from_ppo(model, env)
        
        # Create a RandomForestRegressor and fit it to the data
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        
        # Use the RandomForestRegressor's feature importances
        model = rf_model

    if method == 'shap':
        # Create a SHAP explainer
        explainer = shap.TreeExplainer(model)
        # Calculate SHAP values
        shap_values = explainer.shap_values(X_test)
        feature_importance = np.abs(shap_values).mean(0)
    elif method == 'perm':
        perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
        feature_importance = perm_importance.importances_mean
    elif method == 'grad':
        # Assuming the model has a feature_importances_ attribute (e.g., RandomForestRegressor)
        if hasattr(model, 'feature_importances_'):
            feature_importance = model.feature_importances_
        else:
            raise ValueError("The selected model doesn't support gradient-based feature importance.")
    elif method == 'tree':
        dt_model = DecisionTreeRegressor(random_state=42)
        dt_model.fit(X_train, y_train)
        feature_importance = dt_model.feature_importances_
    else:
        raise ValueError("Invalid method selected. Choose from 'shap', 'perm', 'grad', or 'tree'.")

    # Normalize feature importance scores
    feature_importance = feature_importance / np.sum(feature_importance)

    # Use the provided feature names if available
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(X.shape[1])]

    return feature_importance

def calculate_future_returns(data, mapping):
    future_returns = np.zeros(len(data) - 1)
    for i in range(len(data) - 1):
        future_returns[i] = np.mean((data[i+1, :, mapping['close']] - data[i, :, mapping['close']]) / data[i, :, mapping['close']])
    return future_returns

def save_plot(fig, filename):
    output_dir = "feature_importance_plots"
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, filename))
    plt.close(fig)

def create_symbol_feature_importance_plots(feature_importance, feature_names, valid_symbols, method, output_dir):
    logging.info("Creating feature importance plots for each symbol")
    for symbol in valid_symbols:
        symbol_features = [f for f in feature_names if f.startswith(f"{symbol}_")]
        symbol_importance = [imp for feat, imp in zip(feature_names, feature_importance) if feat in symbol_features]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.bar(range(len(symbol_importance)), symbol_importance)
        plt.title(f"{method.capitalize()} Feature Importance for {symbol}")
        plt.xlabel("Feature")
        plt.ylabel("Importance")
        plt.xticks(range(len(symbol_features)), [f.split('_', 1)[1] for f in symbol_features], rotation=90)
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, f'{symbol}_feature_importance_{method}.png'))
    logging.info(f"Feature importance plots for each symbol saved in the 'feature_importance_plots' directory")

def create_summary_feature_importance_plot(feature_importance, feature_names, valid_symbols, method, output_dir):
    logging.info("Creating summary feature importance plot across all symbols")
    
    # Group feature importances by feature type (e.g., 'open', 'close', 'high', etc.)
    feature_types = set(f.split('_', 1)[1] for f in feature_names)
    grouped_importances = {ft: [] for ft in feature_types}
    
    for feat, imp in zip(feature_names, feature_importance):
        _, feat_type = feat.split('_', 1)
        grouped_importances[feat_type].append(imp)
    
    # Calculate average importance for each feature type
    avg_importances = {ft: np.mean(imps) for ft, imps in grouped_importances.items()}
    
    # Sort feature types by importance
    sorted_features = sorted(avg_importances.items(), key=lambda x: x[1], reverse=True)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(sorted_features)), [imp for _, imp in sorted_features])
    
    # Customize the plot
    ax.set_xlabel("Feature Type")
    ax.set_ylabel("Average Importance")
    ax.set_title(f"{method.capitalize()} Feature Importance Summary Across All Symbols")
    ax.set_xticks(range(len(sorted_features)))
    ax.set_xticklabels([feat for feat, _ in sorted_features], rotation=45, ha='right')
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2%}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'summary_feature_importance_{method}.png'))
    logging.info(f"Summary feature importance plot saved as {method}_feature_importance_summary.png")

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description="Calculate feature importance using various methods")
    parser.add_argument("-m", "--model", help="Path to the saved model file")
    parser.add_argument("-o", "--method", choices=['shap', 'perm', 'grad', 'tree'], default='shap',
                        help="Method to calculate feature importance")
    args = parser.parse_args()

    try:
        logger.info("Starting feature importance analysis")
        
        # # Replace with your function to create financial parameters
        # financial_params = create_financial_params(selected_params)
        # financial_params['initial_balance'] = 5000
        # financial_params['symbols'] = ['GAINS_20'] 
        # logger.info("Financial parameters created")
        
        # Use preprocess_data from utilities.py
        from utilities import preprocess_data
        logger.info("Preprocessing data")
        symbols = selected_params['symbols']
        interval = selected_params.get('interval', '1d')
        limit = selected_params.get('limit', 365)
        end_time = selected_params.get('end_time', None)
        
        data_matrix, timestamps, field_mapping, valid_symbols, _ = preprocess_data(10, symbols, interval, limit, end_time)
        logger.info(f"Field mapping: {field_mapping}")
        valid_symbols = sorted(valid_symbols)
        logger.info(f"Data preprocessed. Shape: {data_matrix.shape}, Symbols: {len(valid_symbols)}")
        
        if data_matrix.shape[0] != selected_params['limit']:
            logger.info(f"Data shape does not match limit. Expected {selected_params['limit']}, got {data_matrix.shape[0]}")
            sys.exit(1)
        
        # Replace with your function to calculate future returns
        logger.info("Calculating future returns")
        y = calculate_future_returns(data_matrix, field_mapping)
        logger.info(f"Future returns calculated. Shape: {y.shape}")

        # Prepare features (X)
        logger.info("Preparing features")
        num_candles, num_symbols, num_features = data_matrix.shape
        X = data_matrix[:-1].reshape(num_candles - 1, num_symbols * num_features)  # Remove last row to align with y
        logger.info(f"Features prepared. Shape: {X.shape}")

        # Create meaningful feature names
        feature_names = []
        for symbol in valid_symbols:
            for feature, index in field_mapping.items():
                feature_names.append(f"{symbol}_{feature}")

        # Ensure that all 44 features per symbol are included
        # If field_mapping does not cover all features, you need to update it
        if len(feature_names) != num_symbols * num_features:
            logger.warning(f"Feature names count ({len(feature_names)}) does not match expected count ({num_symbols * num_features}).")

        logger.info(f"Feature names created. Total features: {len(feature_names)}")
    
        if args.model:
            logger.info(f"Loading PPO model from {args.model}")
            
            # Use initialize_environments from utilities.py
            from utilities import initialize_environments
            train_env, eval_env, test_env, agent, market_conditions = initialize_environments(selected_params, training_params)
            logger.info("Environments initialized successfully")
            
            logger.info(f"Starting to load PPO model from {args.model}...")
            model = PPO.load(args.model, env=train_env)
            logger.info("PPO model loaded successfully")
            
            # Normalize features
            logger.info("Normalizing features")
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            logger.info("Features normalized")
            
            # Calculate feature importance using the specified method
            logger.info(f"Calculating feature importance using {args.method} method")
            feature_importance = calculate_feature_importance(X_scaled, y, model, train_env, feature_names, method=args.method)
            
            # Create and save feature importance plots for each symbol
            create_symbol_feature_importance_plots(feature_importance, feature_names, valid_symbols, args.method, "feature_importance_plots")
            
            # Create and save summary feature importance plot
            create_summary_feature_importance_plot(feature_importance, feature_names, valid_symbols, args.method, "feature_importance_plots")
        else:
            logger.info("No model specified. Using Random Forest Regressor.")
            
            # Normalize features
            logger.info("Normalizing features")
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            logger.info("Features normalized")
            
            # Use RandomForestRegressor
            logger.info("Training Random Forest model")
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(X_scaled, y)
            logger.info("Random Forest model trained")
            
            # Calculate feature importance using the specified method
            logger.info(f"Calculating feature importance using {args.method} method")
            feature_importance = calculate_feature_importance(X_scaled, y, rf_model, feature_names=feature_names, method=args.method)
            
            # Create and save summary feature importance plot
            create_summary_feature_importance_plot(feature_importance, feature_names, valid_symbols, args.method, "feature_importance_plots")

        # Print top 10 most important features
        top_features = sorted(zip(feature_names, feature_importance), key=lambda x: x[1], reverse=True)[:10]
        print(f"Top 10 most important features ({args.method} method):")
        for name, importance in top_features:
            print(f"{name}: {importance:.2%}")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        sys.exit(1)

    logger.info("Feature importance analysis completed successfully.")