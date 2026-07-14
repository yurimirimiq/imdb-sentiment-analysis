# IMDB Sentiment Analysis

IMDB 영화 리뷰 데이터셋을 활용해 영어 리뷰의 긍정/부정을 예측하는 머신러닝 프로젝트입니다.

> ⚠️ 이 모델은 **영어 리뷰**만 지원합니다. IMDB 데이터셋 자체가 영어로 구성되어 있어, 한국어 등 다른 언어 입력 시 정확한 예측을 보장하지 않습니다.

## 데이터셋

[IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) (Kaggle)

프로젝트 실행 전 위 링크에서 데이터셋을 다운로드하여 `data/IMDB Dataset.csv` 경로에 넣어주세요.

## 프로젝트 구조

imdb-sentiment-analysis/
├── data/                  # 데이터셋 (gitignore 처리)
├── models/                # 학습된 모델 저장
├── notebooks/              # EDA 노트북
├── src/
│   ├── preprocess.py       # 텍스트 전처리
│   ├── train.py             # 모델 학습
│   └── predict.py            # 감성 예측
├── requirements.txt
└── README.md

## 사용 방법

### 1. 환경 설정
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. 모델 학습
```bash
python src/train.py
```

### 3. 감성 예측
```bash
python src/predict.py
```

## 모델 성능

- 알고리즘: TF-IDF + Logistic Regression
- Accuracy: **90.1%**

|          | Precision | Recall | F1-score |
|----------|-----------|--------|----------|
| Negative | 0.91      | 0.89   | 0.90     |
| Positive | 0.90      | 0.91   | 0.90     |

## 사용 기술

- Python, pandas, scikit-learn
- TF-IDF Vectorization
- Logistic Regression

## 향후 개선 방향

- 딥러닝 기반 모델(LSTM, BERT 등)로 확장
- 다국어(한국어 등) 지원
- Streamlit을 활용한 웹 데모 제공