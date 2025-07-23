import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import numpy as np
import json # Import json to save metrics

# --- Configuration ---
DATASET_PATH = 'employee_data.csv'
MODEL_PATH = 'salary_model.pkl'
ENCODER_GENDER_PATH = 'le_gender.pkl'
ENCODER_EDUCATION_PATH = 'le_education.pkl'
ENCODER_ROLE_PATH = 'le_role.pkl'
METRICS_PATH = 'model_performance_metrics.json' # File to save performance metrics

# --- Load Dataset ---
try:
    df = pd.read_csv(DATASET_PATH)
    print(f"Dataset '{DATASET_PATH}' loaded successfully.")
except FileNotFoundError:
    print(f"Error: '{DATASET_PATH}' not found. Please ensure the dataset is in the same directory.")
    exit() # Exit if dataset is not found

# --- Data Preprocessing ---
# Filter invalid age and experience ranges
initial_rows = len(df)
df = df[(df['Age'] >= 18) & (df['Age'] <= 60)]
df = df[(df['Experience'] >= 0) & (df['Experience'] <= 35)]
filtered_rows = len(df)
if initial_rows - filtered_rows > 0:
    print(f"Filtered out {initial_rows - filtered_rows} rows due to invalid Age/Experience.")

# Initialize LabelEncoders for categorical features
le_gender = LabelEncoder()
le_education = LabelEncoder()
le_role = LabelEncoder()

# Fit and transform categorical columns
df['Gender'] = le_gender.fit_transform(df['Gender'])
df['Education'] = le_education.fit_transform(df['Education'])
df['Role'] = le_role.fit_transform(df['Role'])

# Define features (x) and target (y)
x = df[['Age', 'Gender', 'Experience', 'Education', 'Role']]
y = df['Income']

# --- Data Splitting ---
# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print(f"Data split into training ({len(x_train)} samples) and testing ({len(x_test)} samples).")

# --- Model Training ---
print("Training RandomForestRegressor model...")
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(x_train, y_train)
print("Model training complete.")

# --- Model Evaluation ---
print("Evaluating model performance...")
y_pred = model.predict(x_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Display performance
print("\n✅ Model Trained Successfully!")
print(f"R² Score         : {r2:.2f}")
print(f"MAE (Avg Error)  : ₹{mae:,.0f}")
print(f"RMSE             : ₹{rmse:,.0f}")

# --- Save Model and Encoders ---
try:
    joblib.dump(model, MODEL_PATH)
    joblib.dump(le_gender, ENCODER_GENDER_PATH)
    joblib.dump(le_education, ENCODER_EDUCATION_PATH)
    joblib.dump(le_role, ENCODER_ROLE_PATH)
    print(f"\nModel saved to '{MODEL_PATH}'")
    print(f"Encoders saved to '{ENCODER_GENDER_PATH}', '{ENCODER_EDUCATION_PATH}', '{ENCODER_ROLE_PATH}'.")
except Exception as e:
    print(f"Error saving model or encoders: {e}")

# --- Save Performance Metrics and Plot Data to JSON ---
try:
    metrics = {
        'r2_score': round(r2, 2),
        'mae': round(mae, 0),
        'rmse': round(rmse, 0),
        'actual_salaries': y_test.tolist(), # Convert Series to list
        'predicted_salaries': y_pred.tolist() # Convert numpy array to list
    }
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Performance metrics and plot data saved to '{METRICS_PATH}'.")
except Exception as e:
    print(f"Error saving performance metrics: {e}")

print("\nTraining script finished.")