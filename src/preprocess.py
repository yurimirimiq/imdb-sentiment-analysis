import re
import pandas as pd
from sklearn.model_selection import train_test_split

def clean_text(text: str) -> str:
    """HTML 태그 제거, 특수문자 제거, 소문자화"""
    text = re.sub(r"<.*?>", " ", text)          # HTML 태그 제거 (예: <br />)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)     # 알파벳 외 문자 제거
    text = re.sub(r"\s+", " ", text).strip()     # 중복 공백 정리
    return text.lower()

def load_data(path: str = "data/IMDB Dataset.csv"):
    df = pd.read_csv(path)
    df["cleaned_review"] = df["review"].apply(clean_text)
    df["label"] = df["sentiment"].map({"positive": 1, "negative": 0})
    return df

def split_data(df, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        df["cleaned_review"], df["label"],
        test_size=test_size, random_state=random_state, stratify=df["label"]
    )
    return X_train, X_test, y_train, y_test