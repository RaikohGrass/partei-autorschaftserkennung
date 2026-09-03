from sklearn.linear_model import LogisticRegression

def embedding_classifier(X_train_embeddings, y_train, regularization = 1.0):
    """
    Train a logistic regression classifier on the provided embeddings and labels.

    Parameters:
    - X_train_embeddings: numpy array of shape (n_samples, n_features)
        The training embeddings.
    - y_train: numpy array of shape (n_samples,)
        The training labels.

    Returns:
    - embedding_classifier: trained LogisticRegression model
    """
    
    embedding_classifier = LogisticRegression(max_iter=2000, random_state=11, C=regularization)

    embedding_classifier.fit(X_train_embeddings, y_train)

    return embedding_classifier
