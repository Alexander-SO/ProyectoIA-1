import os
import pickle
import time

from flask import Flask, jsonify, render_template, request
from sklearn.model_selection import train_test_split

from model.naive_bayes import (NaiveBayesClassifier, evaluate_model,
                               load_model, save_model)
from utils.data_loader import load_bbc_news_dataset

app = Flask(__name__)
model_path = 'model/modelo_entrenado.pkl'
evaluation_report = {}

# Carga o entrenamiento del modelo
if os.path.exists(model_path):
    print("Modelo ya entrenado encontrado. Cargando...")
    classifier, evaluation_report = load_model(model_path)
else:
    print("Entrenando modelo desde cero...")
    data = load_bbc_news_dataset()
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    classifier = NaiveBayesClassifier()
    classifier.train(train_data)
    evaluation_report = evaluate_model(classifier, test_data)
    save_model(classifier, evaluation_report, model_path)
    print("Modelo entrenado y guardado.")

@app.route('/report')
def report():
    return jsonify(evaluation_report)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    start = time.time()
    input_text = request.json.get("text", "")
    prediction = classifier.predict(input_text)
    elapsed = round((time.time() - start) * 1000000)
    return jsonify({
        "category": prediction,
        "elapsed_ms": elapsed
    })

if __name__ == '__main__':
    app.run(debug=True)
