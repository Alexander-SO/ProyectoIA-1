#daniel molina
#alex solorzano
#omar vasquez

#PROYECTO 1 IA
import os
import pickle
import time

from flask import Flask, jsonify, render_template, request
from sklearn.model_selection import train_test_split

from model.naive_bayes import (
    NaiveBayesClassifier,
    evaluate_model,
    load_model,
    save_model
)
from utils.data_loader import load_bbc_news_dataset

app = Flask(__name__)
modelo_guardado = 'model/modelo_entrenado.pkl'
reporte_evaluacion = {}

if os.path.exists(modelo_guardado):
    print("Modelo existente detectado. Procediendo a cargar...")
    clasificador, reporte_evaluacion = load_model(modelo_guardado)
else:
    print("No se encontró un modelo. Iniciando entrenamiento...")
    datos = load_bbc_news_dataset()
    datos_entrenamiento, datos_prueba = train_test_split(datos, test_size=0.2, random_state=42)

    clasificador = NaiveBayesClassifier()
    clasificador.train(datos_entrenamiento)

    reporte_evaluacion = evaluate_model(clasificador, datos_prueba)

    save_model(clasificador, reporte_evaluacion, modelo_guardado)
    print("Entrenamiento finalizado y modelo guardado con éxito.")

@app.route('/reporte')
def mostrar_reporte():
    """Devuelve el reporte de evaluación del modelo en formato JSON."""
    return jsonify(reporte_evaluacion)

@app.route('/')
def inicio():
    """Renderiza la página principal."""
    return render_template('index.html')

@app.route('/predecir', methods=['POST'])
def realizar_prediccion():
    """Recibe un texto y devuelve la categoría predicha y el tiempo de respuesta."""
    tiempo_inicio = time.time()
    texto_usuario = request.json.get("text", "")
    resultado = clasificador.predict(texto_usuario)
    tiempo_total = round((time.time() - tiempo_inicio) * 1000000)  # en microsegundos

    return jsonify({
        "categoria": resultado,
        "tiempo_respuesta_us": tiempo_total
    })

if __name__ == '__main__':
    app.run(debug=True)
