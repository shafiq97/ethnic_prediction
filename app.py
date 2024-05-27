from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from io import BytesIO
import base64

app = Flask(__name__)

# Load the dataset
excel_file = 'fetal_growth_chart.xlsx'
dfs = pd.read_excel(excel_file, sheet_name=None)
combined_data = pd.concat(dfs.values(), ignore_index=True)

# Filter data for each race
indian_data = combined_data[combined_data['RACE'] == 'I']
chinese_data = combined_data[combined_data['RACE'] == 'C']
malay_data = combined_data[combined_data['RACE'] == 'M']

# Define the feature columns
feature_cols = ['BPD', 'HC', 'AC', 'FL']

# Function to preprocess the data
def preprocess_data(data):
    X = data[feature_cols].copy()
    y = data['EFW'].apply(str).str.replace(',', '').astype(float).copy()
    X['GA'] = data['GA'].str.extract(r'(\d+\.?\d*)').astype(float)
    imputer = SimpleImputer(strategy='mean')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
    y = pd.Series(imputer.fit_transform(y.values.reshape(-1, 1)).flatten())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Preprocess the data for each race
X_train_indian, X_test_indian, y_train_indian, y_test_indian = preprocess_data(indian_data)
X_train_chinese, X_test_chinese, y_train_chinese, y_test_chinese = preprocess_data(chinese_data)
X_train_malay, X_test_malay, y_train_malay, y_test_malay = preprocess_data(malay_data)

# Train models
rf_model_malay = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42)
rf_model_malay.fit(X_train_malay, y_train_malay)

rf_model_indian = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42)
rf_model_indian.fit(X_train_indian, y_train_indian)

rf_model_chinese = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42)
rf_model_chinese.fit(X_train_chinese, y_train_chinese)

def plot_to_img_tag(plt):
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    return f"data:image/png;base64,{img}"

def predict_ethnicity(BPD, HC, AC, FL, GA):
    feature_array = [[BPD, HC, AC, FL, GA]]
    malay_prediction = rf_model_malay.predict(feature_array)[0]
    indian_prediction = rf_model_indian.predict(feature_array)[0]
    chinese_prediction = rf_model_chinese.predict(feature_array)[0]

    predictions = {
        'Malay': malay_prediction,
        'Indian': indian_prediction,
        'Chinese': chinese_prediction
    }

    closest_ethnicity = min(predictions, key=predictions.get)
    return closest_ethnicity, predictions

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    BPD = float(request.form['BPD'])
    HC = float(request.form['HC'])
    AC = float(request.form['AC'])
    FL = float(request.form['FL'])
    GA = float(request.form['GA'])
    race = request.form['race']

    feature_array = [[BPD, HC, AC, FL, GA]]

    if race == 'Malay':
        model = rf_model_malay
    elif race == 'Indian':
        model = rf_model_indian
    elif race == 'Chinese':
        model = rf_model_chinese
    else:
        return "Invalid race selected", 400

    prediction = model.predict(feature_array)[0]

    # Generate the prediction bar chart
    plt.figure(figsize=(6, 4))
    plt.bar(['Predicted'], [prediction], color='blue')
    plt.xlabel('Feature')
    plt.ylabel('Value')
    plt.title(f'Predicted Feature Value for {race} Race')
    prediction_img_tag = plot_to_img_tag(plt)
    plt.close()

    # Generate the actual vs. predicted values graph
    X_train, X_test, y_train, y_test = preprocess_data(combined_data[combined_data['RACE'] == race[0]])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    plt.figure(figsize=(6, 4))
    plt.scatter(y_test, predictions, color='blue', label='Predicted')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='black', linestyle='--', label='Actual')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'Actual vs. Predicted Values - {race} Race')
    plt.legend()
    actual_vs_predicted_img_tag = plot_to_img_tag(plt)
    plt.close()

    # Determine the closest ethnic prediction
    closest_ethnicity, ethnicity_predictions = predict_ethnicity(BPD, HC, AC, FL, GA)

    return render_template(
        'result.html',
        prediction_img_tag=prediction_img_tag,
        actual_vs_predicted_img_tag=actual_vs_predicted_img_tag,
        race=race,
        prediction=prediction,
        closest_ethnicity=closest_ethnicity,
        ethnicity_predictions=ethnicity_predictions
    )

if __name__ == '__main__':
    app.run(debug=True, port=8000)
