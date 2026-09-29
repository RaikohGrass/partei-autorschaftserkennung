import sys
sys.path.append(".")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
from models import dataframe_to_chunks
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay

FINAL_MODEL_PATH = "models/gbert-faction-classifier"
GBERT_tokenizer = AutoTokenizer.from_pretrained(FINAL_MODEL_PATH)
GBERT_model = AutoModelForSequenceClassification.from_pretrained(FINAL_MODEL_PATH)

df_test = pd.read_parquet("data/processed/test.parquet")

classes = sorted(df_test["faction_id"].unique())
label2id = {label: i for i, label in enumerate(classes)}
id2label = {i: label for label, i in label2id.items()}

logit_columns = [f"logit_{i}" for i in range(len(classes))]

MAX_LENGTH = 256
MAX_CHUNKS_PER_SPEECH = 3
STRIDE = 64

LABEL_COL = "faction_id"
TEXT_COL = "masked_speech"

test_chunks = dataframe_to_chunks(df_test)
test_speech_ids = test_chunks["speech_id"].to_numpy()

test_dataset = Dataset.from_pandas(test_chunks.drop(columns=["speech_id", "chunk_number"]), preserve_index=False)

prediction_output = GBERT_model.predict(test_dataset)
chunk_logits = prediction_output.predictions

test_logits_df = pd.DataFrame(
    chunk_logits,
    columns=logit_columns,
)

test_logits_df["speech_id"] = test_speech_ids
test_speech_logits = (test_logits_df.groupby("speech_id")[logit_columns].mean().sort_index())
y_test_pred = test_speech_logits.to_numpy().argmax(axis=1)

y_test_true = (
    df_test
    .reset_index(drop=True)[LABEL_COL]
    .map(label2id)
    .to_numpy()
)

test_accuracy = accuracy_score(y_test_true, y_test_pred)
test_macro_f1 = f1_score(y_test_true, y_test_pred, average="macro")

print("Transformer test accuracy:", test_accuracy)
print("Transformer test macro-F1:", test_macro_f1)

ConfusionMatrixDisplay.from_predictions(
    y_test_true,
    y_test_pred,
    labels=range(len(classes)),
    display_labels=classes,
    cmap="Blues",
    xticks_rotation=45,
    normalize="true",
)

plt.tight_layout()
plt.show()