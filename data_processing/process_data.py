import pandas as pd
import re

PARTY_PATTERN = re.compile(
    r"\b(?:"
    r"CDU/CSU|CDU|CSU|"
    r"Unionsfraktion|Union|"
    r"SPD|"
    r"Sozialdemokrat(?:en|innen)?|"
    r"Sozialdemokratinnen und Sozialdemokraten|"
    r"sozialdemokratisch(?:e|en|er|es|em)?|"
    r"AfD|Alternative für Deutschland|"
    r"Bündnis 90/Die Grünen|Bündnis 90|"
    r"GRÜNE|Grüne|Grünen|Grüner|"
    r"Grünenfraktion|"
    r"DIE LINKE|Die Linke|Linken|Linker|"
    r"Linksfraktion|"
    r"FDP|Freie Demokraten"
    r")\b"
)

def mask_party_names(text, party_pattern=PARTY_PATTERN) -> str:
    return party_pattern.sub("[PARTY]", text)

def main():
    
    df_train = pd.read_feather("data/filtered/train.feather")
    df_val = pd.read_feather("data/filtered/val.feather")
    df_test = pd.read_feather("data/filtered/test.feather")

    for dataset in [df_train, df_val, df_test]:
        dataset["masked_speech"] = (dataset["speech_content"].str.replace("\xa0", " ", regex=False).apply(mask_party_names))

    df_train.to_parquet("data/processed/train.parquet")
    df_val.to_parquet("data/processed/val.parquet")
    df_test.to_parquet("data/processed/test.parquet")

if __name__ == "__main__":
    main()