# 🧠 Clasificador de Noticias BBC con Naïve Bayes

Este proyecto implementa un clasificador de noticias basado en el dataset **BBC News Summary**, utilizando un modelo **Naïve Bayes desarrollado desde cero en Python**. La aplicación cuenta con una interfaz web construida con **Flask + HTML + JavaScript**, que permite ingresar una noticia y clasificarla en una de las 5 categorías: `business`, `entertainment`, `politics`, `sport`, `tech`.


## 🚀 Tecnologías Utilizadas

- **Python 3.10**
- **Flask** (framework backend)
- **HTML + CSS + JavaScript** (frontend)
- **scikit-learn** (solo para evaluación)
- **Dataset**: [BBC News Summary (Kaggle)](https://www.kaggle.com/datasets/pariza/bbc-news-summary)


## 📁 Estructura del Proyecto
```
ProyectoIA/
├── app.py                 # Servidor Flask
├── model/
│   └── naive_bayes.py     # Clasificador Naïve Bayes desde cero
├── utils/
│   └── data_loader.py     # Función para cargar el dataset
├── dataset/
│   └── News Articles/     # Archivos .txt organizados por categoría
├── templates/
│   └── index.html         # Interfaz principal
├── static/
│   ├── style.css          # Estilos visuales
│   └── script.js          # Lógica de frontend
├── requirements.txt       # Dependencias del proyecto
└── README.md              # Este archivo
```

## ▶️ ¿Cómo ejecutar el proyecto?

1. Clona este repositorio o descarga el ZIP.
2. Asegúrate de tener Python 3.10+
3. Instala dependencias utilizando este comando en la consola:
```
    pip install -r requirements.txt
```
    
4. Ejecuta la app:
```
    python app.py
```
5. Abre en el navegador:

    http://127.0.0.1:5000/