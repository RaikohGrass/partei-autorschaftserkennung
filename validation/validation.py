import sys
sys.path.append(".")

import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, f1_score
from models.baseline import baseline_model
from models.ngram import ngram_model
from models.embeddings import embedding_classifier

# Auslesen der Datensätze
df_train = pd.read_parquet("data/processed/train.parquet")
df_validation = pd.read_parquet("data/processed/val.parquet")
df_test = pd.read_parquet("data/processed/test.parquet")

# Aufteilen der Features und Labels für Training und Validierung
X_train = df_train["masked_speech"]
y_train = df_train["faction_id"]

X_validation = df_validation["masked_speech"]
y_validation = df_validation["faction_id"]

# Laden der Embeddings für Training und Validierung
X_train_embeddings = np.load("data/embeddings/train_embeddings.npy")
X_validation_embeddings = np.load("data/embeddings/val_embeddings.npy")

dummy = baseline_model(X_train, y_train) # Fitting des Baseline-Modells
ngram = ngram_model(X_train, y_train) # Fitting des n-gram Modells
embedding_classifier = embedding_classifier(X_train_embeddings, y_train, regularization=100.0) # Fitting des Embedding-Klassifikators

# Vorhersagen auf dem Validierungsdatensatz
dummy_predictions = dummy.predict(X_validation)
ngram_predictions = ngram.predict(X_validation)
embedding_predictions = embedding_classifier.predict(X_validation_embeddings)

print("dummy Validation accuracy:",accuracy_score(y_validation, dummy_predictions))
print("ngram Validation accuracy:",accuracy_score(y_validation, ngram_predictions))
print("embedding Validation accuracy:",accuracy_score(y_validation, embedding_predictions))

print("dummy Validation F1 score:",f1_score(y_validation, dummy_predictions, average="weighted"))
print("ngram Validation F1 score:",f1_score(y_validation, ngram_predictions, average="weighted"))
print("embedding Validation F1 score:",f1_score(y_validation, embedding_predictions, average="weighted"))
