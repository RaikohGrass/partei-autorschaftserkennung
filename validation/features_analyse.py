import sys
sys.path.append(".")

import pandas as pd
import numpy as np
from models.ngram import ngram_model

df_train = pd.read_parquet("data/processed/train.parquet") # Trainingsatz auslesen

X_train = df_train["masked_speech"] # Maskierte Reden als X
y_train = df_train["faction_id"] # Fraktionen als Y

ngram = ngram_model(X_train, y_train) # Fitting des n-gram Modells

vectorizer = ngram.named_steps["tfidf"] # Extrahieren des TF-IDF Vektorisierers
classifier = ngram.named_steps["classifier"] # Extrahieren des Klassifikators

feature_names = np.array(vectorizer.get_feature_names_out()) # Extrahieren der Feature-Namen

for class_index, faction in enumerate(classifier.classes_): # Für jede Fraktion
    coefficients = classifier.coef_[class_index] # Koeffizienten der Klasse
    top_indices = coefficients.argsort()[-20:][::-1] # Top 20 Features

    print(f"\n{faction}:")
    print(feature_names[top_indices])