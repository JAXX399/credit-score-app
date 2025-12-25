import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# --- PATH SETUP ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
DATA_PATH = os.path.join(ROOT_DIR, 'data', 'german_credit_data.csv')
MODEL_DIR = os.path.join(ROOT_DIR, 'models')

# Create models directory if it doesn't exist
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

def generate():
    print(f"1. Loading data from: {DATA_PATH}")
    if not os.path.exists(DATA_PATH):
        print("❌ Error: Data file not found. Please make sure 'data/german_credit_data.csv' exists.")
        return

    df = pd.read_csv(DATA_PATH)

    # --- FIX IS HERE ---
    target_col = 'credit_risk'
    
    # Check if 'credit_risk' exists, otherwise look for alternatives
    if target_col not in df.columns:
         print(f"   'credit_risk' not found. Checking alternatives...")
         if 'Risk' in df.columns:
             target_col = 'Risk'
         elif 'class' in df.columns:
             target_col = 'class'
         else:
             # Fallback: Assume the very last column is the target
             target_col = df.columns[-1] 
    
    print(f"   Target Column identified as: '{target_col}'")

    # 2. Prepare Features
    X = df.drop([target_col], axis=1)
    y = df[target_col]

    # One-Hot Encoding
    print("2. Encoding categorical features...")
    X_encoded = pd.get_dummies(X)
    
    # 3. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

    # 4. Scale Data
    print("3. Scaling data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. Define the 3 Experts (Ensemble)
    clf1 = LogisticRegression(random_state=1, max_iter=1000)
    clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
    clf3 = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=1)

    ensemble = VotingClassifier(
        estimators=[('lr', clf1), ('rf', clf2), ('gb', clf3)],
        voting='soft'
    )

    # 6. Train
    print("4. Training the Ensemble Model (this may take a moment)...")
    ensemble.fit(X_train_scaled, y_train)

    # Evaluate
    acc = accuracy_score(y_test, ensemble.predict(X_test_scaled))
    print(f"   ✅ Model Accuracy: {acc:.4f}")

    # 7. Save Files
    print(f"5. Saving artifacts to {MODEL_DIR}...")
    
    joblib.dump(ensemble, os.path.join(MODEL_DIR, 'ensemble_model.pkl'))
    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))
    joblib.dump(list(X_encoded.columns), os.path.join(MODEL_DIR, 'model_columns.pkl'))

    print("\n🎉 Success! You can now run the frontend.")

if __name__ == "__main__":
    generate()