import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from preprocess import load_data, split_data

def train():
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

    joblib.dump(model, "models/sentiment_model.pkl")
    joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
    print("모델 저장 완료!")

if __name__ == "__main__":
    train()