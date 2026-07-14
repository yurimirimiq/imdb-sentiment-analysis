import joblib
from preprocess import clean_text

model = joblib.load("models/sentiment_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

def predict_sentiment(text: str) -> str:
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    return "긍정 😊" if pred == 1 else "부정 😞"

if __name__ == "__main__":
    print("⚠️  이 모델은 영어(English) 영화 리뷰만 지원합니다.")
    while True:
        review = input("\nEnter your movie review (or 'exit' to quit): ")