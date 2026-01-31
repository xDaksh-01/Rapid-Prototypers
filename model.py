
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import config
import feature_engineering

# Features to use for training
FEATURE_COLS = [
    'fan_in', 'fan_out', 'total_in', 'total_out', 
    'unique_sources', 'unique_dests', 'mule_score', 'balance_ratio',
    'avg_time_gap', 'std_time_gap'
]

def train_model():
    """Trains the MLP model and saves artifacts."""
    
    # 1. Get Data
    df_trans, df_labels = feature_engineering.load_data()
    df_features = feature_engineering.compute_features(df_trans, df_labels)
    
    # 2. Filter for Labeled Data
    # Drop rows where 'Label' is NaN
    labeled_df = df_features.dropna(subset=['Label'])
    
    print(f"Training on {len(labeled_df)} labeled instances.")
    if len(labeled_df) == 0:
        print("No labeled data found! Cannot train.")
        return

    X = labeled_df[FEATURE_COLS]
    y = labeled_df['Label'].astype(int)
    
    # 3. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 4. Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 5. Model Definition
    # Simple MLP
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=42
    )
    
    print("Training MLP...")
    mlp.fit(X_train_scaled, y_train)
    
    # 6. Evaluation
    y_pred = mlp.predict(X_test_scaled)
    y_prob = mlp.predict_proba(X_test_scaled)[:, 1]
    
    print("\nModel Evaluation:")
    print(classification_report(y_test, y_pred))
    print(f"AUC-ROC: {roc_auc_score(y_test, y_prob):.4f}")
    
    # 7. Save Artifacts
    print("Saving model and scaler...")
    with open(config.MODEL_PATH, 'wb') as f:
        pickle.dump(mlp, f)
        
    with open(config.SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
        
    print("Done.")

def load_artifacts():
    """Loads model and scaler."""
    try:
        with open(config.MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(config.SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    except FileNotFoundError:
        print("Model artifacts not found. Please train first.")
        return None, None

def predict_risk(model, scaler, feature_dict):
    """
    Predicts risk for a single wallet or list of wallets.
    feature_dict: DataFrame or dict of features
    """
    # Ensure input is DataFrame
    if isinstance(feature_dict, dict):
        df = pd.DataFrame([feature_dict])
    else:
        df = feature_dict
        
    X = df[FEATURE_COLS]
    X_scaled = scaler.transform(X)
    
    prob = model.predict_proba(X_scaled)[:, 1]
    return prob

if __name__ == "__main__":
    train_model()
