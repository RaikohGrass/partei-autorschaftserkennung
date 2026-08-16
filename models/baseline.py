from sklearn.dummy import DummyClassifier

def baseline_model(X_train, y_train):
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)

    return dummy