import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import joblib
from datetime import datetime

# Load dataset (placeholder function, replace with actual path)
def load_data():
    train_data = pd.read_csv('train.csv')  # Replace with actual path
    test_data = pd.read_csv('test.csv')    # Replace with actual path
    return train_data, test_data

# Preprocessing: Handle datetime and feature engineering
def preprocess_data(data):
    data['Date'] = pd.to_datetime(data['Date'])
    
    # Extract datetime features
    data['Weekday'] = data['Date'].dt.weekday
    data['IsWeekend'] = data['Weekday'] >= 5
    data['Month'] = data['Date'].dt.month
    data['MonthSegment'] = pd.cut(
        data['Date'].dt.day,
        bins=[0, 10, 20, 31],
        labels=['Beginning', 'Mid', 'End'],
        right=False
    )
    data['Quarter'] = data['Date'].dt.quarter
    data['IsMonthStart'] = data['Date'].dt.is_month_start
    data['IsMonthEnd'] = data['Date'].dt.is_month_end

    # Holiday features
    data['DaysToHoliday'] = data['StateHoliday'].replace('0', np.nan).notnull().astype(int).cumsum()
    data['DaysAfterHoliday'] = data['StateHoliday'].replace('0', np.nan).notnull().astype(int)[::-1].cumsum()

    # Handle categorical variables
    data['StateHoliday'] = data['StateHoliday'].astype(str)

    return data

# Prepare the dataset for modeling
def prepare_features(train_data, test_data):
    train_data = preprocess_data(train_data)
    test_data = preprocess_data(test_data)

    # Select features and target
    feature_columns = ['Weekday', 'IsWeekend', 'Month', 'Quarter', 'MonthSegment',
                       'IsMonthStart', 'IsMonthEnd', 'DaysToHoliday', 'DaysAfterHoliday',
                       'Promo', 'StateHoliday']
    target_column = 'Sales'

    X_train = train_data[feature_columns]
    y_train = train_data[target_column]
    X_test = test_data[feature_columns]

    return X_train, y_train, X_test

# Build the pipeline
def build_pipeline():
    numeric_features = ['Weekday', 'DaysToHoliday', 'DaysAfterHoliday', 'Month', 'Quarter']
    categorical_features = ['Promo', 'StateHoliday', 'MonthSegment', 'IsWeekend', 'IsMonthStart', 'IsMonthEnd']

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    model = RandomForestRegressor(random_state=42)

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])

    return pipeline

# Train and evaluate the model
def train_and_evaluate(X_train, y_train, pipeline):
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_val)

    mse = mean_squared_error(y_val, predictions)
    print(f"Validation Mean Squared Error: {mse:.2f}")

    return pipeline

# Save the model with a timestamp
def save_model(pipeline):
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    filename = f"model-{timestamp}.pkl"
    joblib.dump(pipeline, filename)
    print(f"Model saved as {filename}")

# Main execution
def main():
    train_data, test_data = load_data()
    X_train, y_train, X_test = prepare_features(train_data, test_data)

    pipeline = build_pipeline()
    trained_pipeline = train_and_evaluate(X_train, y_train, pipeline)

    save_model(trained_pipeline)

if __name__ == "__main__":
    main()
