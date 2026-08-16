from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

RANDOM_SEED = 11

def ngram_model(X_train, y_train):

      word_ngram_model = Pipeline([
      (
            "tfidf",
            TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True, lowercase=True)
      ),
      (
            "classifier",
            LogisticRegression(max_iter=2000, random_state=RANDOM_SEED)
      )
      ])

      word_ngram_model.fit(X_train, y_train)
      
      return word_ngram_model