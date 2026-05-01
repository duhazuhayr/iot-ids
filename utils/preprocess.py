import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib
import os

def preprocess_data(input_path='data/iot_intrusion_dataset.csv', output_dir='data/processed'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    df = pd.read_csv(input_path)
    
    # 1. Split into Seen and Unseen Attacks
    # Seen: normal, dos, ddos, injection
    # Unseen: scanning, backdoor
    unseen_types = ['scanning', 'backdoor']
    
    seen_df = df[~df['type'].isin(unseen_types)].copy()
    unseen_df = df[df['type'].isin(unseen_types)].copy()
    
    print(f"Seen samples: {len(seen_df)}, Unseen samples: {len(unseen_df)}")
    
    # 2. Feature Selection
    # Drop irrelevant or high-cardinality features for a lightweight generic model
    drop_cols = ['ts', 'src_ip', 'dst_ip', 'type'] # 'type' is target for multiclass, 'label' for binary
    
    # Label encoding for categorical features
    cat_cols = ['proto', 'service', 'conn_state']
    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        seen_df[col] = le.fit_transform(seen_df[col])
        unseen_df[col] = le.transform(unseen_df[col])
        le_dict[col] = le
    
    # 3. Handle Target
    # For training the Teacher/Student: Multiclass Classification on Seen types
    target_le = LabelEncoder()
    seen_df['target'] = target_le.fit_transform(seen_df['type'])
    
    # For Anomaly Detector: Binary (Normal=0, Attack=1)
    # Already has 'label' col
    
    # 4. Scaling
    features = seen_df.drop(columns=drop_cols + ['target', 'label']).columns
    scaler = StandardScaler()
    
    X_seen = seen_df[features]
    y_seen = seen_df['target']
    
    scaler.fit(X_seen)
    
    X_seen_scaled = scaler.transform(X_seen)
    X_unseen_scaled = scaler.transform(unseen_df[features])
    
    # 5. Balancing (SMOTE) on Seen data
    print("Applying SMOTE to balance seen classes...")
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_seen_scaled, y_seen)
    
    # 6. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42, stratify=y_res)
    
    # 7. Save everything
    np.save(f"{output_dir}/X_train.npy", X_train)
    np.save(f"{output_dir}/X_test.npy", X_test)
    np.save(f"{output_dir}/y_train.npy", y_train)
    np.save(f"{output_dir}/y_test.npy", y_test)
    
    # Save unseen data for zero-day testing
    np.save(f"{output_dir}/X_unseen.npy", X_unseen_scaled)
    unseen_df.to_csv(f"{output_dir}/unseen_metadata.csv", index=False)
    
    # Save normal-only data for Autoencoder training
    # Only use 'normal' from training set features
    normal_indices = np.where(y_train == target_le.transform(['normal'])[0])[0]
    X_train_normal = X_train[normal_indices]
    np.save(f"{output_dir}/X_train_normal.npy", X_train_normal)
    
    # Save artifacts
    joblib.dump(scaler, f"{output_dir}/scaler.joblib")
    joblib.dump(target_le, f"{output_dir}/target_le.joblib")
    joblib.dump(le_dict, f"{output_dir}/cat_les.joblib")
    
    print("Preprocessing complete. Data saved in data/processed/")
    print(f"Features: {list(features)}")
    print(f"Seen Classes: {target_le.classes_}")

if __name__ == "__main__":
    preprocess_data()
