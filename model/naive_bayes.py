import math
import pickle
import re
from collections import Counter, defaultdict

from sklearn.metrics import classification_report


class NaiveBayesClassifier:
    def __init__(self):
        # Conjunto total de palabras distintas vistas durante el entrenamiento
        self.vocab = set()
        # Diccionario con el conteo de palabras por clase, ej: {"sport": {"goal": 10, "match": 5}, ...}
        self.class_word_counts = defaultdict(Counter)
        # Conteo total de documentos por clase, ej: {"tech": 200, "business": 150}
        self.class_counts = defaultdict(int)
        # Total de documentos usados en el entrenamiento
        self.total_docs = 0

    def tokenize(self, text):
        tokens = re.findall(r'\b\w+\b', text.lower())
        return [t for t in tokens if len(t) > 2]

    def train(self, data):
        for text, label in data:
            self.total_docs += 1                           # Aumenta el contador de documentos totales
            self.class_counts[label] += 1                  # Aumenta el contador de la clase correspondiente
            tokens = self.tokenize(text)                   # Tokeniza el texto
            self.vocab.update(tokens)                      # Agrega tokens al vocabulario general
            self.class_word_counts[label].update(tokens)   # Actualiza el conteo de palabras por clase


    def predict(self, text):
        tokens = self.tokenize(text)   # Limpia y tokeniza el texto de entrada
        scores = {}                    # Guardará las probabilidades logarítmicas para cada clase

        for label in self.class_counts:
            # Probabilidad a priori de la clase (log P(C))
            log_prob = math.log(self.class_counts[label] / self.total_docs)
            
            # Cantidad total de palabras observadas en esta clase
            total_words = sum(self.class_word_counts[label].values())

            for token in tokens:
                word_freq = self.class_word_counts[label][token] + 1
                word_prob = word_freq / (total_words + len(self.vocab))

                # Suma del log de la probabilidad
                log_prob += math.log(word_prob)

            scores[label] = log_prob  # Guarda el puntaje total (log) para esta clase

        # Devuelve la clase con mayor probabilidad (mayor log-probabilidad)
        return max(scores, key=scores.get)


def evaluate_model(classifier, test_data):
    y_true = []
    y_pred = []
    for text, true_label in test_data:
        pred = classifier.predict(text)
        y_true.append(true_label)
        y_pred.append(pred)
    report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return report_dict

def save_model(classifier, evaluation_report, path='model/modelo_entrenado.pkl'):
    with open(path, 'wb') as f:
        pickle.dump({
            "classifier": classifier,
            "evaluation": evaluation_report
        }, f)

def load_model(path='model/modelo_entrenado.pkl'):
    with open(path, 'rb') as f:
        data = pickle.load(f)
        return data["classifier"], data.get("evaluation", {})

