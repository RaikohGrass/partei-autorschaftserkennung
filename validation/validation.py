import sys
sys.path.append(".")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, f1_score, classification_report, ConfusionMatrixDisplay
from models.baseline import baseline_model
from models.ngram import ngram_model
from models.embeddings import embedding_classifier

df_train = pd.read_parquet("data/processed/train.parquet")
df_validation = pd.read_parquet("data/processed/val.parquet")
df_test = pd.read_parquet("data/processed/test.parquet")

X_train = df_train["masked_speech"]
y_train = df_train["faction_id"]

X_validation = df_validation["masked_speech"]
y_validation = df_validation["faction_id"]

X_test = df_test["masked_speech"]
y_test = df_test["faction_id"]   

X_train_embeddings = np.load("data/embeddings/train_embeddings_token.npy")
X_validation_embeddings = np.load("data/embeddings/val_embeddings_token.npy")
X_test_embeddings = np.load("data/embeddings/test_embeddings_token.npy")

dummy = baseline_model(X_train, y_train)
ngram = ngram_model(X_train, y_train)
embedding_classifier = embedding_classifier(X_train_embeddings, y_train, regularization=100.0)


embedding_predictions = embedding_classifier.predict(X_validation_embeddings)
embedding_training_predictions = embedding_classifier.predict(X_train_embeddings)

print(
    "Training accuracy:",
    accuracy_score(y_train, embedding_training_predictions)
)

print(
    "Validation accuracy:",
    accuracy_score(y_validation, embedding_predictions)
)

dummy_predictions = dummy.predict(X_validation)
ngram_predictions = ngram.predict(X_validation)
embedding_predictions = embedding_classifier.predict(X_validation_embeddings)

print("dummy Validation accuracy:",accuracy_score(y_validation, dummy_predictions))
print("ngram Validation accuracy:",accuracy_score(y_validation, ngram_predictions))
print("embedding Validation accuracy:",accuracy_score(y_validation, embedding_predictions))

print("dummy Validation F1 score:",f1_score(y_validation, dummy_predictions, average="weighted"))
print("ngram Validation F1 score:",f1_score(y_validation, ngram_predictions, average="weighted"))
print("embedding Validation F1 score:",f1_score(y_validation, embedding_predictions, average="weighted"))


embedding_results = []

for c_value in [0.01, 0.1, 1, 10, 100]:
    # classifier = LogisticRegression(
    #     C=c_value,
    #     max_iter=3000,
    #     random_state=42
    # )

    classifier = embedding_classifier(X_train_embeddings, y_train, regularization=c_value)

    predictions = classifier.predict(
        X_validation_embeddings
    )

    embedding_results.append({
        "C": c_value,
        "accuracy": accuracy_score(
            y_validation,
            predictions
        ),
        "macro_f1": f1_score(
            y_validation,
            predictions,
            average="macro"
        )
    })



print(
    classification_report(
        y_validation,
        ngram_predictions,
        digits=3
    )
)

ConfusionMatrixDisplay.from_predictions(
    y_validation,
    ngram_predictions,
    normalize="true",
    cmap="Blues",
    xticks_rotation=45
)

plt.tight_layout()
plt.show()

vectorizer = ngram.named_steps["tfidf"]
classifier = ngram.named_steps["classifier"]

feature_names = np.array(vectorizer.get_feature_names_out())

for class_index, faction in enumerate(classifier.classes_):
    coefficients = classifier.coef_[class_index]
    top_indices = coefficients.argsort()[-20:][::-1]

    print(f"\n{faction}:")
    print(feature_names[top_indices])