import sys
sys.path.append(".")

import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, f1_score
from models.embeddings import embedding_classifier

# Auslesen der Datensätze
df_train = pd.read_parquet("data/processed/train.parquet")
df_validation = pd.read_parquet("data/processed/val.parquet")

# Laden der Embeddings für Training und Validierung
X_train_embeddings = np.load("data/embeddings/train_embeddings.npy")
X_validation_embeddings = np.load("data/embeddings/val_embeddings.npy")

# Aufteilen der Labels für Training und Validierung
y_train = df_train["faction_id"]
y_validation = df_validation["faction_id"]

embedding_results = []

for c_value in [0.01, 0.1, 1, 10, 100]:
    classifier = embedding_classifier(X_train_embeddings, y_train, regularization=c_value) # Fitting des Modells mit dem aktuellen Regularisierungswert
    predictions = classifier.predict(X_validation_embeddings) # Vorhersagen auf dem Validierungsdatensatz
    embedding_results.append({"C": c_value, 
                              "accuracy": accuracy_score(y_validation, predictions), 
                              "macro_f1": f1_score(y_validation, predictions, average="macro")}) # List von Dicts mit den Validierungsmaßen
    
    print(f"Regularization C={c_value}: Accuracy={embedding_results[-1]['accuracy']}, Macro F1={embedding_results[-1]['macro_f1']}")