# Klassifikation der Fraktionszugehörigkeit anhand von Reden im Deutschen Bundestag

In diesem Repository befinden sich die Skript, die für die Untersuchung im Rahmen einer Hausarbeit des Kurses "Einführung in die Computerlinguistik" durchgeführt wurde.

Verschiedene einzelne anwendungsfertige Skripte sowie ein Jupyter-Notebook, die den ganzen Untersuchungsprozess beschreiben, stehen zur Verfügung.


## Anleitung

### Requirements

Vor der Nutzung der Skripte müssen die Python-Packages von ***"requirements.txt"*** installiert werden. Bei der Untersuchung habe ich Python 3.14.6 verwendet.

Für die Ausführung der Skripte zur Feinabstimmung des GBERT-Modells müssen die extra Packages von ***"requirements_GPU.txt"*** installiert werden. Diese funktionieren aber nur für Linux/Windows-Systeme und werden nur empfohlen, wenn der Rechner einen leistungsfähigen GPU besitzt. 

### Rohdaten

Im Repository steht nur eine Kopie der aufbereiteten Daten zur Verfügung. Um die Verarbeitung der Daten zu überprüfen/replizieren, muss eine Kopie der Rohdaten im Ordner ***./data/raw/**** herunterladen werden.

Folgende Dateien werden benötigt:

- factions.feather
- politicians.feather
- speeches.feather

Alle diese Dateien steht das OpenDiscourse-Projekt zur [Verfügung](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910%2FDVN%2FFIKIBO).

### Einzelne Skripte

Die erste Möglichkeit zur Erkundung der geleisteten Arbeit ist eine Reihe von einzelnen Skripte, die nacheinander ausgeführt/erkundet werden können:

1. Datenaufberaitung (***./data_processing/***)
    1. filter_data.py
    2. process_data.py
    3. generate_embeddings.py
2. Modelldefinierung (***./models/***)
    1. baseline.py
    2. ngram.py
    3. embeddings.py
    4. GBERT_fine_tuning.py (nur mit GPU auszuführen)
3. Auswertung der Modelle (***./validation/***)
    1. embeddings_regularization.py
    2. validation.py
    3. features_analyse.py
    4. evaluation.py
    5. GBERT_evaluation.py (nur mit GPU auszuführen)

### Jupyter Notebook

Die zweite Erkundungsmöglichkeit ist das Jupyter-Notebook. Drinnen werden Stücke von den einzelnen Skripte herbeigerufen und der ganze Untersuchungsprozess wird kurz dabei erklärt.