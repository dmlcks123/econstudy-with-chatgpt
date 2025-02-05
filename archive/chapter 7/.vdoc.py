# type: ignore
# flake8: noqa
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# 단어 단위 토큰화
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

from nltk.tokenize import word_tokenize

# 실습 파일을 활용한 토큰화
tokenized_text = [word_tokenize(line) for line in cleaned_data]
print(tokenized_text[:5])
#
#
#
!pip install konlpy
```
#
#
#
#
# 한국어 형태소 단위 토큰화
from konlpy.tag import Okt

okt = Okt()

text = "안녕하세요. 저는 자연어 처리를 공부하고 있습니다."

# 형태소 단위로 토큰화
tokens = okt.morphs(text)

print(tokens)
# 출력: ['안녕하세요', '.', '저', '는', '자연어', '처리', '를', '공부', '하고', '있습니다', '.']
#
#
#
#
#
#
#
#
# 영어 불용어 처리
from nltk.corpus import stopwords
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
filtered_text = [[word for word in line if word not in stop_words] for line in tokenized_text]
print(filtered_text[:5])
```
#
#
#
#
# 한국어 불용어 처리
import nltk
from nltk.tokenize import word_tokenize
import requests

# Download 'punkt_tab' if not already downloaded
nltk.download('punkt_tab')

# 한국어 불용어 리스트 다운로드
url = "https://raw.githubusercontent.com/stopwords-iso/stopwords-ko/master/stopwords-ko.txt"
stopwords_ko = set(requests.get(url).text.split("\n"))

print("불용어 개수:", len(stopwords_ko))
print("일부 불용어:", list(stopwords_ko)[:10])

# 샘플 텍스트
text = "이것은 샘플 문장입니다. 그리고 불용어를 제거할 것입니다."

# 토큰화
tokens = word_tokenize(text)

# 불용어 제거
filtered_tokens = [word for word in tokens if word not in stopwords_ko]

print(filtered_tokens)
```
#
#
#
#
#
#

# 직접 불용어 설정하기
from konlpy.tag import Okt
import re

okt = Okt()

# 원본 텍스트
text = "이것은 예제 문장입니다! 123 This is a Sample Sentence."

# 불용어 제거
stopwords = ['이것은', '예제']
cleaned_text = ' '.join([word for word in text.split() if word not in stopwords])

# 특수문자 제거
cleaned_text = re.sub(r'[^가-힣\s]', '', cleaned_text)

# 대소문자 변환 (영어 텍스트만 해당)
cleaned_text = cleaned_text.lower()

print(cleaned_text)
```
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# sklearn의 CountVectorizer를 불러옴 (BoW 모델 생성)
from sklearn.feature_extraction.text import CountVectorizer

# CountVectorizer 객체 생성 (단어 빈도를 기반으로 벡터화)
vectorizer = CountVectorizer()

# 텍스트 데이터를 BoW로 변환 (cleaned_data는 전처리된 텍스트 리스트)
X = vectorizer.fit_transform(cleaned_data)

# 생성된 단어 사전(Feature Names) 출력
print(vectorizer.get_feature_names_out())

# BoW 표현된 행렬을 배열 형태로 변환하여 출력
print(X.toarray())
#
#
#
#
#
#
#
#
#
#
#
#
# scikit-learn의 TfidfVectorizer 라이브러리를 불러옴 (TF-IDF 모델 생성)
from sklearn.feature_extraction.text import TfidfVectorizer

# TfidfVectorizer 객체 생성 (TF-IDF 기반으로 텍스트를 숫자로 변환하는 도구)
tfidf_vectorizer = TfidfVectorizer()

# 텍스트 데이터를 TF-IDF 방식으로 변환 (cleaned_data는 전처리된 텍스트 리스트)
X_tfidf = tfidf_vectorizer.fit_transform(cleaned_data)

# 변환된 TF-IDF 행렬을 배열 형태로 변환하여 출력
print(X_tfidf.toarray())
```
#
#
#
#
#
#
#
#
# TextBlob 라이브러리 불러오기 (텍스트 감성 분석을 위한 라이브러리)
from textblob import TextBlob

# 감성 분석을 수행할 샘플 텍스트 (영어 문장)
sample_text = "I really love this product! It's amazing."

# TextBlob을 사용하여 감성 점수(polarity) 계산
# sentiment.polarity 값은 -1(부정적) ~ 1(긍정적) 범위의 감성 점수를 반환
sentiment_score = TextBlob(sample_text).sentiment.polarity

# 감성 점수 출력
print(f'Sentiment Score: {sentiment_score}')
```
#
#
#
#
pip install soynlp
#
#
#
from soynlp.normalizer import emoticon_normalize
from soynlp.word import WordExtractor
import pandas as pd

df = pd.DataFrame({'text': text_data})

# 감성 분석을 위한 간단한 스코어링 함수
def simple_sentiment(text):
    pos_words = ["좋아요", "좋네요", "유익", "추천", "최고", "기뻐", "행복"]
    neg_words = ["별로", "싫어요", "최악", "짜증", "후회"]
    
    score = sum([1 for word in pos_words if word in text]) - sum([1 for word in neg_words if word in text])
    
    return score

# 감성 분석 적용
df['sentiment'] = df['text'].apply(simple_sentiment)

print(df)
#
#
#
#
#
#
from pykospacing import Spacing
from konlpy.tag import Okt
import re

# 띄어쓰기 교정
spacing = Spacing()
text = "해당문장은띄어쓰기가필요합니다"
corrected_text = spacing(text)
print("Corrected Text:", corrected_text)

```
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# 데이터 시각화를 위한 라이브러리 불러오기
import matplotlib.pyplot as plt  # 기본적인 그래프 그리기
import seaborn as sns  # 고급 통계 시각화 라이브러리

# 그래프 크기 설정 (가로 6인치, 세로 4인치)
plt.figure(figsize=(6,4))

# 감성 점수(sentiment) 히스토그램 생성
# bins=20 → 20개의 구간으로 나누어 히스토그램을 그림
# kde=True → 커널 밀도 추정(KDE)를 추가하여 분포를 부드럽게 표현
sns.histplot(df['sentiment'], bins=20, kde=True)

# X축 레이블 설정 (Sentiment Score)
plt.xlabel('Sentiment Score')

# Y축 레이블 설정 (Frequency)
plt.ylabel('Frequency')

# 그래프 제목 설정
plt.title('Sentiment Score Distribution')

# 그래프 출력
plt.show()
```
#
#
#
#
!pip install konlpy wordcloud
#
#
#
# 워드클라우드 생성 라이브러리 불러오기
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# 한국어 형태소 분석기(Okt) 불러오기
from konlpy.tag import Okt

# Okt 형태소 분석기 객체 생성
okt = Okt()

# 한국어 명사만 추출하는 함수 정의
def extract_korean_words(text):
    return ' '.join(okt.nouns(text))  # 명사만 추출하여 공백으로 연결

# ✅ 긍정 감성 텍스트에서 한국어 명사 추출
positive_text = ' '.join(df[df['sentiment'] > 0]['text'].apply(extract_korean_words))  # 감성 점수가 0보다 큰 문장의 명사만 추출

# ✅ 긍정 워드클라우드 생성 (단, 데이터가 없으면 경고 메시지 출력)
if not positive_text.strip():  # 만약 긍정적인 단어가 없다면
    print("Warning: No positive words found for wordcloud.")  # 경고 메시지 출력
else:
    wordcloud_positive = WordCloud(
       font_path='/content/NanumGothic.ttf',  # 한글 폰트 지정 (Google Colab 환경 기준)
        background_color='white'  # 배경색: 흰색
    ).generate(positive_text)  # 긍정적인 단어를 바탕으로 워드클라우드 생성

    # ✅ 그래프 설정
    plt.figure(figsize=(12, 6))  # 그래프 크기 설정
    plt.subplot(1, 2, 1)  # 1행 2열 중 첫 번째 서브플롯
    plt.imshow(wordcloud_positive, interpolation='bilinear')  # 워드클라우드 출력
    plt.axis("off")  # 축 표시 제거
    plt.title("Positive Sentiment")  # 그래프 제목 설정

# ✅ 부정 감성 텍스트에서 한국어 명사 추출
negative_text = ' '.join(df[df['sentiment'] <= 0]['text'].apply(extract_korean_words))  # 감성 점수가 0 이하인 문장의 명사만 추출

# ✅ 부정 워드클라우드 생성 (데이터가 없으면 경고 메시지 출력)
if not negative_text.strip():  # 만약 부정적인 단어가 없다면
    print("Warning: No negative words found for wordcloud.")  # 경고 메시지 출력
else:
    wordcloud_negative = WordCloud(
        font_path='/content/NanumGothic.ttf',  # 한글 폰트 지정
        background_color='black'  # 배경색: 검정색
    ).generate(negative_text)  # 부정적인 단어를 바탕으로 워드클라우드 생성

    # ✅ 부정 감성 워드클라우드 시각화
    plt.subplot(1, 2, 2)  # 1행 2열 중 두 번째 서브플롯
    plt.imshow(wordcloud_negative, interpolation='bilinear')  # 워드클라우드 출력
    plt.axis("off")  # 축 표시 제거
    plt.title("Negative Sentiment")  # 그래프 제목 설정

# ✅ 그래프 출력
plt.show()
```
#
#
#
#
#
#
#
#
#
#
#
## 1. 뉴스기사 크롤링

# 크롬 드라이버 설정
options = webdriver.ChromeOptions()
options.add_argument('--headless')  # 크롬창이 뜨지 않게 설정
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
driver = webdriver.Chrome(options=options)

# 연합 뉴스 경제 페이지로 이동
url = 'https://www.yna.co.kr/economy/all'
driver.get(url)

# 페이지 배율 줄이기 (줌 아웃)
driver.execute_script("document.body.style.zoom='50%'")
time.sleep(2)

# 페이지 스크롤 끝까지 내리기
last_height = driver.execute_script("return document.body.scrollHeight")
while True:
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(2)
    new_height = driver.execute_script("return document.body.scrollHeight")
    if new_height == last_height:
        break
    last_height = new_height

# 페이지 소스 가져오기
page_source = driver.page_source
driver.quit()

# BeautifulSoup을 사용하여 페이지 파싱
soup = BeautifulSoup(page_source, 'html.parser')
news_data = []

# 뉴스 기사 요소 찾기
news_elements = soup.select('div.list-type038 ul.list li')

for element in news_elements:
    try:
        # 제목과 링크 수집
        link_element = element.select_one('a.tit-wrap')
        title = link_element.text.strip()
        link = link_element['href']

        # 시간 수집
        time_element = element.select_one('span.txt-time')
        time_text = time_element.text.strip() if time_element else ""

        news_data.append({
            'title': title,
            'link': link,
            'time': time_text
        })
    except Exception as e:
        continue

# 수집한 데이터를 CSV 파일로 저장
current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
file_name = f'{current_time}.csv'

with open(file_name, mode='w', newline='', encoding='utf-8-sig') as file:
    writer = csv.DictWriter(file, fieldnames=['title', 'link', 'time'])
    writer.writeheader()
    writer.writerows(news_data)

print(f'Data saved to {file_name}')
#
#
#
#
#
# 텍스트 전처리 및 토크나이즈 함수 정의
okt = Okt()
spacing = Spacing()

def preprocess_text(text):
    # 특수문자 제거
    text = re.sub(r'[^가-힣\s]', '', text)
    # 띄어쓰기 교정
    text = spacing(text)
    # 불용어 제거
    stopwords = ['에', '에게', '은', '는', '이', '가', '을', '를', '저', '여기', '연합뉴스']
    text = ' '.join([word for word in text.split() if word not in stopwords])
    return text

def tokenize_text(text):
    return okt.morphs(text)

# 기사 내용을 수집하는 함수
def get_article_body(url):
    driver.get(url)
    time.sleep(2)  # 페이지가 로드될 시간을 주기 위해 추가
    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')

    # 기사 본문 내용 추출 (모든 <p> 태그를 대상으로 함)
    content = soup.find_all('p')
    article_text = ""
    for paragraph in content:
        article_text += paragraph.get_text().strip() + " "

    return article_text.strip()

# 크롬 드라이버 설정
options = webdriver.ChromeOptions()
options.add_argument('--headless')  # 크롬창이 뜨지 않게 설정
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
driver = webdriver.Chrome(options=options)

# 기사 내용 수집
articles_body = [get_article_body(article['link']) for article in news_data]

# 전처리 및 토크나이즈 적용
processed_bodies = [preprocess_text(body) for body in articles_body]
tokenized_bodies = [tokenize_text(body) for body in processed_bodies]

# 크롬 드라이버 종료
driver.quit()

# 단어별 언급 횟수를 바탕으로 데이터프레임 생성
word_counts_list = []
for i, tokens in enumerate(tokenized_bodies):
    word_counts = pd.Series(tokens).value_counts()
    word_counts.name = news_data[i]['title']
    word_counts_list.append(word_counts)

df = pd.concat(word_counts_list, axis=1).fillna(0).astype(int)
print(df)

# 데이터프레임을 엑셀 파일로 저장
current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
excel_file_name = f'word_counts_{current_time}.xlsx'
df.to_excel(excel_file_name)
print(f'DataFrame saved to {excel_file_name}')
#
#
#
#
#
## 수집된 기사 요약, 키워드, 감정분석

import pandas as pd
import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, pipeline
from konlpy.tag import Okt
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F


# CSV 파일 불러오기
file_path = '/content/20240718_022815.csv'
df = pd.read_csv(file_path)

# 요약 모델 로드 (PyTorch 사용)
summarizer_model_name = "lcw99/t5-base-korean-text-summary"
summarizer_tokenizer = AutoTokenizer.from_pretrained(summarizer_model_name)
summarizer_model = AutoModelForSeq2SeqLM.from_pretrained(summarizer_model_name)
summarizer = pipeline("summarization", model=summarizer_model, tokenizer=summarizer_tokenizer, framework="pt")

# 감정 분석 모델 로드 (PyTorch 사용)
sentiment_model_name = "WhitePeak/bert-base-cased-Korean-sentiment"
sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)
sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
sentiment_analyzer = pipeline("sentiment-analysis", model=sentiment_model, tokenizer=sentiment_tokenizer)

# 형태소 분석기 로드
okt = Okt()

# 기사 내용을 수집하고 요약 및 키워드 추출, 감정 분석 수행
def fetch_article_content(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        content = ' '.join([para.get_text() for para in paragraphs])
        return content
    except Exception as e:
        return None

def extract_keywords(text, top_n=5):
    nouns = okt.nouns(text)
    freq_dist = pd.Series(nouns).value_counts()
    keywords = freq_dist.head(top_n).index.tolist()
    return keywords

def summarize_text(text):
    try:
        if len(text) < 100:
            return text
        inputs = summarizer_tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
        summary_ids = summarizer_model.generate(inputs, max_length=150, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = summarizer_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    except Exception as e:
        return "요약 실패"

def analyze_sentiment(text):
    # 텍스트를 토큰 단위로 나누기
    max_length = 512
    tokens = sentiment_tokenizer.tokenize(text)

    # 토큰을 512 길이 이하로 나누기
    chunks = [' '.join(tokens[i:i + max_length]) for i in range(0, len(tokens), max_length)]
    sentiments = []
    for chunk in chunks:
        inputs = sentiment_tokenizer(chunk, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
        with torch.no_grad():
            outputs = sentiment_model(**inputs)
        sentiment = torch.argmax(outputs.logits, dim=-1).item()
        sentiments.append(sentiment)

    # 다수결로 최종 감정 결정
    sentiment = max(set(sentiments), key=sentiments.count)
    sentiment_labels = ["매우 부정적", "부정적", "중립적", "긍정적", "매우 긍정적"]
    return sentiment_labels[sentiment]

# 새로운 칼럼 추가
df['summary'] = ''
df['keywords'] = ''
df['sentiment'] = ''

for index, row in df.iterrows():
    article_url = row['link']
    article_content = fetch_article_content(article_url)

    if article_content:
        # 요약
        summary = summarize_text(article_content)
        df.at[index, 'summary'] = summary

        # 키워드 추출
        keywords = extract_keywords(article_content)
        df.at[index, 'keywords'] = ', '.join(keywords)

        # 감정 분석
        sentiment = analyze_sentiment(article_content)
        df.at[index, 'sentiment'] = sentiment

# 결과를 CSV 파일로 저장
output_file_path = '/content/processed_articles.csv'
df.to_csv(output_file_path, index=False, encoding='utf-8-sig')

print(f'Processed data saved to {output_file_path}')

current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
excel_file_name = f'word_counts_sentiments_{current_time}.xlsx'
df.to_excel(excel_file_name)
print(f'DataFrame saved to {excel_file_name}')
#
#
#
#
#
