import pandas as pd
import numpy as np
import joblib
import json
import io
import base64
import matplotlib
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from flask import Flask, request, jsonify, render_template
import random

app = Flask(__name__)

try:
    model = joblib.load('exoplanet_model.joblib')
    scaler = joblib.load('scaler.joblib')
    le = joblib.load('label_encoder.joblib')
    print("Models loaded successfully!")
except FileNotFoundError:
    print("Error: Model files not found. Run model.py first.")
    exit()

MODEL_FEATURES = [
    'orbital_period', 'transit_duration', 'transit_depth_ppm',
    'planet_radius', 'stellar_temp', 'stellar_logg', 'stellar_radius'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/guides')
def guias():
    return render_template('guides.html')

@app.route('/results')
def results():
    return render_template('results.html')

@app.route('/predict', methods=['POST'])
def predict():
    json_data = request.get_json()
    if not json_data:
        return jsonify({"error": "No data received"}), 400

    input_data = json_data.get('data')
    mapping = json_data.get('mapping')

    if not input_data or not mapping:
        return jsonify({"error": "Invalid data structure"}), 400

    df_raw = pd.DataFrame(input_data)
    df_mapped = pd.DataFrame()
    for model_col, user_col in mapping.items():
        if user_col in df_raw.columns:
            df_mapped[model_col] = df_raw[user_col]

    df_processed = df_mapped.reindex(columns=MODEL_FEATURES)
    for col in df_processed.columns:
        df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')

    df_processed.fillna(0, inplace=True)


    X_scaled = scaler.transform(df_processed)
    predictions_encoded = model.predict(X_scaled)
    predictions = le.inverse_transform(predictions_encoded)
    df_raw['classification'] = predictions

    final_results = df_raw.to_dict(orient='records')

    return jsonify(final_results)


# --- rota metricas ---
@app.route('/model_metrics')
def model_metrics():
    try:
        with open('model_metrics.json', 'r') as f:
            metrics = json.load(f)
        return jsonify(metrics)
    except FileNotFoundError:
        return jsonify({"error": "Metrics file not found"}), 404


@app.route('/general_tree_image')
def general_tree_image():
    estimator = model.estimators_[0]
    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(estimator,
              feature_names=MODEL_FEATURES,
              class_names=le.classes_,
              filled=True,
              rounded=True,
              max_depth=4,
              proportion=False,
              precision=2,
              ax=ax,
              fontsize=8)

    buf = io.BytesIO()
    plt.savefig(buf, format='svg', bbox_inches='tight')
    plt.close(fig)
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    return jsonify({'image': f'data:image/svg+xml;base64,{image_base64}'})


@app.route('/random_tree_image')
def random_tree_image():
    try:
        num_estimators = len(model.estimators_)
        random_tree_index = random.randint(0, num_estimators - 1)
        estimator = model.estimators_[random_tree_index]
        fig, ax = plt.subplots(figsize=(25, 20))
        plot_tree(estimator,
                  feature_names=MODEL_FEATURES,
                  class_names=le.classes_,
                  filled=True,
                  rounded=True,
                  max_depth=4,
                  proportion=False,
                  precision=2,
                  ax=ax,
                  fontsize=10)

        ax.set_title(f"Random Tree Visualization #{random_tree_index + 1}", fontsize=20)

        buf = io.BytesIO()
        plt.savefig(buf, format='svg', bbox_inches='tight')
        plt.close(fig)
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        buf.close()
        return jsonify({'image': f'data:image/svg+xml;base64,{image_base64}'})
    except Exception as e:
        print(f"Error generating random tree: {e}")
        return jsonify({"error": "Failed to generate tree image"}), 500

if __name__ == '__main__':
    app.run(debug=True)