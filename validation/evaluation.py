import sys
sys.path.append(".")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay
from models.baseline import baseline_model
from models.ngram import ngram_model
from models.embeddings import embedding_classifier

# Auslesen der Datensätze
df_train = pd.read_parquet("data/processed/train.parquet")
df_validation = pd.read_parquet("data/processed/val.parquet")
df_test = pd.read_parquet("data/processed/test.parquet")

# Aufteilen der Features und Labels für Training und Test
X_train = df_train["masked_speech"]
y_train = df_train["faction_id"]

X_test = df_test["masked_speech"]
y_test = df_test["faction_id"]   

# Laden der Embeddings für Training und Test
X_train_embeddings = np.load("data/embeddings/train_embeddings.npy")
X_test_embeddings = np.load("data/embeddings/test_embeddings.npy")

dummy = baseline_model(X_train, y_train) # Fitting des Baseline-Modells
ngram = ngram_model(X_train, y_train) # Fitting des n-gram Modells
embedding_classifier = embedding_classifier(X_train_embeddings, y_train, regularization=100.0) # Fitting des Embedding-Klassifikators

# Vorhersagen auf dem Testdatensatz
dummy_test_predictions = dummy.predict(X_test)
ngram_test_predictions = ngram.predict(X_test)
embedding_test_predictions = embedding_classifier.predict(X_test_embeddings)

print("dummy Test accuracy:",accuracy_score(y_test, dummy_test_predictions))
print("ngram Test accuracy:",accuracy_score(y_test, ngram_test_predictions))
print("embedding Test accuracy:",accuracy_score(y_test, embedding_test_predictions))

print("dummy Test F1 score:",f1_score(y_test, dummy_test_predictions, average="macro"))
print("ngram Test F1 score:",f1_score(y_test, ngram_test_predictions, average="macro"))
print("embedding Test F1 score:",f1_score(y_test, embedding_test_predictions, average="macro"))

# Klassenebeneanalyse der Fehler
classes = sorted(df_train["faction_id"].unique())

# Berechnung der F1-Scores pro Klasse für das n-gram Modell
ngram_f1_per_class = f1_score(
    y_test,
    ngram_test_predictions,
    labels=classes,
    average=None,
    zero_division=0,
)

# Berechnung der F1-Scores pro Klasse für das Embedding-Modell
embeddings_f1_per_class = f1_score(
    y_test,
    embedding_test_predictions,
    labels=classes,
    average=None,
    zero_division=0,
)

# Ausgabe der F1-Scores pro Klasse für das n-gram Modell
for faction, score in zip(classes, ngram_f1_per_class):
    print(f"{faction}: {score:.3f}")

# Ausgabe der F1-Scores pro Klasse für das Embedding-Modell
for faction, score in zip(classes, embeddings_f1_per_class):
    print(f"{faction}: {score:.3f}")


# Erstellung der Konfusionsmatrix für das n-gram Modell
ConfusionMatrixDisplay.from_predictions(
    y_test,
    ngram_test_predictions,
    normalize="true",
    cmap="Blues",
    xticks_rotation=45
)

plt.tight_layout()
plt.show()

# Erstellung der Konfusionsmatrix für das Embedding-Modell
ConfusionMatrixDisplay.from_predictions(
    y_test,
    embedding_test_predictions,
    normalize="true",
    cmap="Blues",
    xticks_rotation=45
)

plt.tight_layout()
plt.show()
