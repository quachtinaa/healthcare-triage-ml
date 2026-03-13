"""
Loads CSV data, encodes categorical variables, scales numeric features, and splits into training 
    and testing sets.
    
Returns:
    X_train, X_test, y_train, y_test
"""
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# 20% of data is kept for testing the model
# random_state ensures reproducibility, same split every time
def preprocess_data(test_size = 0.2, random_state=42):
    # import data
    data_path = Path(__file__).resolve().parent.parent / "data" / "synthetic_medical_triage.csv"
    df = pd.read_csv(data_path)

    # separate features and target
    X = df.drop("triage_level", axis = 1)
    y = df["triage_level"] # the target column we want to predict

    # if there are any null values, drop them
    if df.isnull().sum().sum() > 0:
        df = df.dropna()
    
    # encode categorical features
    X = pd.get_dummies(X, columns=["arrival_mode"])

    # encode target
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    # scale numeric features
    numeric_cols = ["age", "heart_rate", "systolic_blood_pressure", "oxygen_saturation",
                    "body_temperature", "pain_level", "chronic_disease_count", "previous_er_visits"]
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, 
                                                        random_state=random_state)

    # Convert boolean columns to float so PyTorch can handle them
    X_train = X_train.astype(float)
    X_test = X_test.astype(float)

    return X_train, X_test, y_train, y_test
