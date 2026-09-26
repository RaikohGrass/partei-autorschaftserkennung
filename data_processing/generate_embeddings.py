import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

embedding_model = SentenceTransformer(MODEL_NAME)
model_tokenizer = embedding_model.tokenizer
max_length = embedding_model.max_seq_length - model_tokenizer.num_special_tokens_to_add(pair=False) # Das ausgewählte Modell fügt zwei spezielle Tokens hinzu. Also 128 - 2 für MiniLM-L12-v2


def split_into_chunks(text):

    token_ids = model_tokenizer(str(text), add_special_tokens=False, truncation=False)["input_ids"]    
    
    chunks = []

    for start in range(0, len(token_ids), max_length):
        chunk_ids = token_ids[start:start + max_length]
        chunk_text = model_tokenizer.decode(chunk_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        chunks.append(chunk_text)
            
    return chunks

def encode_speeches(texts, model):
    speech_embeddings = []

    for text in texts:
        chunks = split_into_chunks(text)
        chunk_embeddings = model.encode(chunks, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
        speech_embedding = chunk_embeddings.mean(axis=0)
        norm = np.linalg.norm(speech_embedding)

        if norm > 0:
            speech_embedding = speech_embedding / norm

        speech_embeddings.append(speech_embedding)

    return np.vstack(speech_embeddings)

def main():
    df_train = pd.read_parquet("data/processed/train.parquet")
    df_val = pd.read_parquet("data/processed/val.parquet")
    df_test = pd.read_parquet("data/processed/test.parquet")

    train_embeddings = encode_speeches(df_train["masked_speech"].tolist(), embedding_model)
    val_embeddings = encode_speeches(df_val["masked_speech"].tolist(), embedding_model)
    test_embeddings = encode_speeches(df_test["masked_speech"].tolist(), embedding_model)

    np.save("data/embeddings/train_embeddings.npy", train_embeddings)
    np.save("data/embeddings/val_embeddings.npy", val_embeddings)
    np.save("data/embeddings/test_embeddings.npy", test_embeddings)

if __name__ == "__main__":
    main()