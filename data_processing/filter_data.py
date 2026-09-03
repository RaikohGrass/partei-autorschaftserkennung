import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def filter_speeches(df_speeches:pd.DataFrame, factions_ids:dict, electoral_term:int, min_words:int) -> pd.DataFrame:
    df_speeches_filtered = df_speeches[(df_speeches["faction_id"].isin(factions_ids.keys())) & (df_speeches["electoral_term"] == electoral_term)]
    # Wir entfernen alle Reden, die keinen Inhalt haben.
    df_speeches_filtered.dropna(subset=["speech_content"], inplace=True)
    
    # Wir entfernen alle Reden, die weniger Wörter als erwünscht haben.
    df_speeches_filtered["word_count"] = df_speeches_filtered["speech_content"].apply(lambda x: len(x.split()))
    df_speeches_filtered = df_speeches_filtered[df_speeches_filtered["word_count"] >= min_words]
    df_speeches_filtered["faction_id"] = df_speeches_filtered["faction_id"].map(factions_ids)

    return df_speeches_filtered

def split_data(df:pd.DataFrame, train_size:float, r_seed:int=11) -> tuple:

    groups = df["politician_id"]

    split_train = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=r_seed)
    train_idx, remaining_idx = next(split_train.split(df, groups=groups))

    df_train = df.iloc[train_idx].copy()
    df_remaining = df.iloc[remaining_idx].copy()

    split_val_test = GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=r_seed)
    val_idx, test_idx = next(split_val_test.split(df_remaining, groups=df_remaining["politician_id"]))

    df_val = df_remaining.iloc[val_idx].copy()
    df_test = df_remaining.iloc[test_idx].copy()

    return df_train, df_val, df_test

def proof_speaker_disjointness(df_train:pd.DataFrame, df_val:pd.DataFrame, df_test:pd.DataFrame) -> None:
    train_speakers = set(df_train["politician_id"].unique())
    val_speakers = set(df_val["politician_id"].unique())
    test_speakers = set(df_test["politician_id"].unique())

    assert len(train_speakers.intersection(val_speakers)) == 0, "Train and Validation sets have overlapping speakers."
    assert len(train_speakers.intersection(test_speakers)) == 0, "Train and Test sets have overlapping speakers."
    assert len(val_speakers.intersection(test_speakers)) == 0, "Validation and Test sets have overlapping speakers."

    print("The three datasets are speaker-disjoint.")

def main():

    # Hier wird es entschieden, welche Wahlperiode und welche Parteien in die Analyse einbezogen werden sollen.
    # Für die Hausarbeit wurde die 19. Wahlperiode und die Parteien CDU/CSU, SPD, Bündnis 90/Die Grünen, DIE LINKE und AfD ausgewählt.
    ELECTORAL_TERM = 19
    PARTIES = ["CDU/CSU", "SPD", "Grüne", "DIE LINKE.", "AfD"]
    SPEECH_MIN_WORD_COUNT = 100
    RANDOM_SEED = 11
    TRAINING_SET_PERCENTAGE = 0.7
    
    df_speeches = pd.read_feather("data/raw/speeches.feather")
    df_politicians = pd.read_feather("data/raw/politicians.feather")
    df_factions = pd.read_feather("data/raw/factions.feather")

    # Erstmal suchen wir die IDs der Parteien, die wir in die Analyse einbeziehen wollen.
    factions_ids = df_factions[df_factions["abbreviation"].isin(PARTIES)][["id", "abbreviation"]].set_index("id").to_dict()["abbreviation"]

    # Danach filtern wir die Reden nach den IDs der Parteien und der Wahlperiode.
    df_speeches_filtered = filter_speeches(df_speeches, factions_ids, ELECTORAL_TERM, SPEECH_MIN_WORD_COUNT)
    politicians_ids = df_speeches_filtered["politician_id"].unique().tolist()

    df_politicians_filtered = df_politicians[df_politicians["id"].isin(politicians_ids)]

    df_train, df_val, df_test = split_data(df_speeches_filtered, TRAINING_SET_PERCENTAGE, r_seed=RANDOM_SEED)

    # folgende Aufteilung nach Fraktion ist im Datensatz enthalten:
    for name, dataset in [
        ("Training", df_train),
        ("Validierung", df_val),
        ("Test", df_test),
    ]:
        print(f"\n{name}:")
        print(dataset["faction_id"].value_counts())

    # da wir nur 1500 Rede pro Fraktion haben wollen, nehmen wir eine Stichprobe vom Datensatz:

    df_train_balanced = (
        df_train
        .groupby("faction_id", group_keys=False)
        .sample(n=350, random_state=RANDOM_SEED)
        .reset_index(drop=True)
    )

    df_val_balanced = (
        df_val
        .groupby("faction_id", group_keys=False)
        .sample(n=75, random_state=RANDOM_SEED)
        .reset_index(drop=True)
    )

    df_test_balanced = (
        df_test
        .groupby("faction_id", group_keys=False)
        .sample(n=75, random_state=RANDOM_SEED)
        .reset_index(drop=True)
    )
    
    # Jetzt müssen wir bestätigen, dass die Redner in den drei Datensätzen disjunkt sind.
    proof_speaker_disjointness(df_train_balanced, df_val_balanced, df_test_balanced)

    df_train_balanced.to_feather("data/filtered/train.feather")
    df_val_balanced.to_feather("data/filtered/val.feather")
    df_test_balanced.to_feather("data/filtered/test.feather")


if __name__ == "__main__":
    main()