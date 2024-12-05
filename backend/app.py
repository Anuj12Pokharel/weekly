import os
from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)
CORS(app)

# Global variable to store future data
future_data = pd.DataFrame()

# Load and preprocess data
def load_data():
    file_path = r'C:\Users\hp\Desktop\weekly\backend\dashboard1_modified - Sheet1.csv'  # Update file path
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at {file_path}")
    
    data = pd.read_csv(file_path)
    
    # Handle missing data
    missing_data = data.isnull().sum()
    if missing_data.any():
        # Fill missing data using backward fill for features
        data = data.bfill()  # Backward fill missing values in features

        # Handle missing values in the target variable `weekly_hospitalised_cases`
        if data['weekly_hospitalised_cases'].isnull().any():
            # Option 1: Drop rows with missing target values
            data = data.dropna(subset=['weekly_hospitalised_cases'])

            # Option 2: Fill missing target values with the mean of the target variable
            # data['weekly_hospitalised_cases'] = data['weekly_hospitalised_cases'].fillna(data['weekly_hospitalised_cases'].mean())

    # Ensure required columns exist
    required_columns = ['year', 'district', 'population', 'week', 'rainsum', 'meantemperature', 'weekly_hospitalised_cases']
    if not all(col in data.columns for col in required_columns):
        raise ValueError(f"Missing required columns: {', '.join([col for col in required_columns if col not in data.columns])}")
    
    # Filter historical data
    historical_data = data.iloc[1:253]  # Historical data (rows 1 to 252)
    historical_data = historical_data[required_columns]
    
    return historical_data

# Train the model
def train_model(historical_data):
    X = historical_data[['rainsum', 'meantemperature']]
    y = historical_data['weekly_hospitalised_cases']
    
    # Check if y contains any NaN values
    if y.isnull().any():
        raise ValueError("Target variable 'weekly_hospitalised_cases' contains NaN values after preprocessing.")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Generate visualization
def generate_visualization(historical_data, future_data, model):
    predicted_cases = model.predict(future_data[['rainsum', 'meantemperature']])
    future_data['weekly_hospitalised_cases'] = predicted_cases
    future_data['week'] = range(historical_data['week'].max() + 1, historical_data['week'].max() + 1 + len(future_data))
    future_data['year'] = "Predicted"

    combined_data = pd.concat([historical_data, future_data], ignore_index=True)

    fig, ax = plt.subplots()
    for year in combined_data['year'].unique():
        yearly_data = combined_data[combined_data['year'] == year]
        if year == "Predicted":
            ax.plot(yearly_data['week'], yearly_data['weekly_hospitalised_cases'], label="Predicted", linestyle='--', color='orange')
        else:
            ax.plot(yearly_data['week'], yearly_data['weekly_hospitalised_cases'], label=f"Year {int(year)}")

    ax.set_title('Weekly Hospitalized Cases (Historical and Predicted)')
    ax.set_xlabel('Week')
    ax.set_ylabel('Weekly Hospitalized Cases')
    ax.legend(title='Year')

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return plot_url

# Initialize data and model
historical_data = load_data()
model = train_model(historical_data)

# API Routes
@app.route('/api/historical', methods=['GET'])
def get_historical_data():
    return jsonify(historical_data.to_dict(orient='records'))

@app.route('/api/upload', methods=['POST'])
def upload_data():
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'No file uploaded'}), 400

    try:
        # Read the file into a Pandas DataFrame
        if file.filename.endswith('.xlsx'):
            data = pd.read_excel(file)
        elif file.filename.endswith('.csv'):
            data = pd.read_csv(file)
        else:
            return jsonify({'error': 'Unsupported file type. Please upload .xlsx or .csv'}), 400

        # Validate required columns
        required_columns = ['rainsum', 'meantemperature']
        if not all(col in data.columns for col in required_columns):
            return jsonify({'error': f"Missing required columns: {', '.join(required_columns)}"}), 400

        # Handle missing values
        data.fillna(method='ffill', inplace=True)

        # Update global future_data variable
        global future_data
        future_data = data[required_columns]
        return jsonify({'message': 'File uploaded successfully. Ready for predictions.'})
    except Exception as e:
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500

@app.route('/api/predict', methods=['GET'])
def get_predicted_data():
    try:
        if future_data.empty:
            return jsonify({'error': 'No input data available for prediction. Please upload a file first.'}), 400

        # Predict using the uploaded data
        predicted_cases = model.predict(future_data)
        future_data_copy = future_data.copy()
        future_data_copy['weekly_hospitalised_cases'] = predicted_cases
        
        # Add additional columns as per the uploaded file structure
        future_data_copy['year'] = historical_data['year'].iloc[-1]  # Use the last year from historical data
        future_data_copy['district'] = historical_data['district'].iloc[0]  # Use the district from the historical data (can adjust as needed)
        future_data_copy['population'] = historical_data['population'].iloc[0]  # Same for population
        future_data_copy['week'] = range(historical_data['week'].max() + 1, historical_data['week'].max() + 1 + len(future_data_copy))
        
        # Rearrange the columns to match the requested structure
        future_data_copy = future_data_copy[['year', 'district', 'population', 'week', 'rainsum', 'meantemperature', 'weekly_hospitalised_cases']]
        
        return jsonify(future_data_copy.to_dict(orient='records'))
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

@app.route('/api/visualization', methods=['GET'])
def get_visualization():
    try:
        if future_data.empty:
            return jsonify({'error': 'No input data available for visualization. Please upload a file first.'}), 400

        plot_url = generate_visualization(historical_data, future_data, model)
        return jsonify({'image': plot_url})
    except Exception as e:
        return jsonify({'error': f'Visualization error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
