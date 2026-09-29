import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
from transformers import DataCollatorWithPadding
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay

MODEL_NAME = "deepset/gbert-base"


df_train = pd.read_parquet("data/processed/train.parquet")
df_val = pd.read_parquet("data/processed/val.parquet")
df_test = pd.read_parquet("data/processed/test.parquet")

classes = sorted(df_train["faction_id"].unique())
label2id = {label: i for i, label in enumerate(classes)}
id2label = {i: label for label, i in label2id.items()}

MAX_LENGTH = 256
MAX_CHUNKS_PER_SPEECH = 3
STRIDE = 64

LABEL_COL = "faction_id"
TEXT_COL = "masked_speech"


def select_evenly(items, maximum):
    if len(items) <= maximum:
        return items

    positions = np.linspace(0, len(items) - 1, num=maximum, dtype=int)

    return [items[position] for position in positions]


def dataframe_to_chunks(df):
    df = df.reset_index(drop=True).copy()

    chunk_rows = []
    max_content_length = MAX_LENGTH - tokenizer.num_special_tokens_to_add(pair=False)

    for speech_id, row in df.iterrows():
        text = str(row[TEXT_COL])
        label = label2id[row[LABEL_COL]]

        token_ids = tokenizer(text, add_special_tokens=False, truncation=False)["input_ids"]

        chunks = []

        start = 0
        while start < len(token_ids):
            chunk = token_ids[start:start + max_content_length]

            chunks.append(tokenizer.prepare_for_model(chunk, add_special_tokens=True, truncation=True, max_length=MAX_LENGTH))

            if start + max_content_length >= len(token_ids):
                break

            start += max_content_length - STRIDE

        chunks = select_evenly(chunks, MAX_CHUNKS_PER_SPEECH)

        for chunk_number, chunk in enumerate(chunks):
            chunk_rows.append({
                "speech_id": speech_id,
                "chunk_number": chunk_number,
                "input_ids": chunk["input_ids"],
                "attention_mask": chunk["attention_mask"],
                "labels": label,
            })

    return pd.DataFrame(chunk_rows)



tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_chunks = dataframe_to_chunks(df_train)
validation_chunks = dataframe_to_chunks(df_val)
test_chunks = dataframe_to_chunks(df_test)

print("Training speeches:", len(df_train))
print("Training chunks:", len(train_chunks))
print("Validation speeches:", len(df_val))
print("Validation chunks:", len(validation_chunks))
print("Test speeches:", len(df_test))
print("Test chunks:", len(test_chunks))

assert train_chunks.groupby("speech_id").size().max() <= 3
assert validation_chunks.groupby("speech_id").size().max() <= 3
assert train_chunks["input_ids"].map(len).max() <= MAX_LENGTH
assert validation_chunks["input_ids"].map(len).max() <= MAX_LENGTH



training_speech_ids = train_chunks["speech_id"].to_numpy()
validation_speech_ids = validation_chunks["speech_id"].to_numpy()
test_speech_ids = test_chunks["speech_id"].to_numpy()

train_dataset = Dataset.from_pandas(
    train_chunks.drop(columns=["speech_id", "chunk_number"]),
    preserve_index=False,
)

validation_dataset = Dataset.from_pandas(
    validation_chunks.drop(columns=["speech_id", "chunk_number"]),
    preserve_index=False,
)

test_dataset = Dataset.from_pandas(
    test_chunks.drop(columns=["speech_id", "chunk_number"]),
    preserve_index=False,
)


print(train_dataset)
print(validation_dataset)
print(test_dataset)

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)

print(train_dataset)
print(validation_dataset)
print(test_dataset)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(classes),
    label2id=label2id,
    id2label=id2label
)

from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="models/gbert-full",

    num_train_epochs=4,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,

    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,

    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,

    logging_strategy="steps",
    logging_steps=50,

    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    fp16=True,
    report_to="none",
    seed=11,
)

from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    processing_class=tokenizer,
    data_collator=data_collator,
)

print("Device:", trainer.args.device)

trainer.train()


prediction_output = trainer.predict(validation_dataset)
chunk_logits = prediction_output.predictions

logit_columns = [
    f"logit_{i}" for i in range(len(classes))
]

logits_df = pd.DataFrame(
    chunk_logits,
    columns=logit_columns,
)

logits_df["speech_id"] = validation_speech_ids

speech_logits = (
    logits_df
    .groupby("speech_id")[logit_columns]
    .mean()
    .sort_index()
)

y_pred = speech_logits.to_numpy().argmax(axis=1)


test_prediction_output = trainer.predict(test_dataset)
test_chunk_logits = test_prediction_output.predictions

test_logits_df = pd.DataFrame(
    test_chunk_logits,
    columns=logit_columns,
)

test_logits_df["speech_id"] = test_speech_ids

test_speech_logits = (
    test_logits_df
    .groupby("speech_id")[logit_columns]
    .mean()
    .sort_index()
)

y_test_pred = test_speech_logits.to_numpy().argmax(axis=1)

y_test_true = (
    df_test
    .reset_index(drop=True)[LABEL_COL]
    .map(label2id)
    .to_numpy()
)

test_accuracy = accuracy_score(y_test_true, y_test_pred)


test_macro_f1 = f1_score(
    y_test_true,
    y_test_pred,
    average="macro",
)

print("Transformer test accuracy:", test_accuracy)
print("Transformer test macro-F1:", test_macro_f1)


gbert_f1_per_class = f1_score(
    y_test_true,
    y_test_pred,
    labels=range(len(classes)),
    average=None,
    zero_division=0,
)

for faction, score in zip(classes, gbert_f1_per_class):
    print(f"Transformer test F1 for {faction}: {score:.3f}")



y_true = (
    df_val
    .reset_index(drop=True)[LABEL_COL]
    .map(label2id)
    .to_numpy()
)

validation_accuracy = accuracy_score(y_true, y_pred)

validation_macro_f1 = f1_score(
    y_true,
    y_pred,
    average="macro",
)

print("Transformer validation accuracy:", validation_accuracy)
print("Transformer validation macro-F1:", validation_macro_f1)

ConfusionMatrixDisplay.from_predictions(
    y_true,
    y_pred,
    labels=range(len(classes)),
    display_labels=classes,
    cmap="Blues",
    xticks_rotation=45,
    normalize="true",
)

plt.tight_layout()
plt.show()
