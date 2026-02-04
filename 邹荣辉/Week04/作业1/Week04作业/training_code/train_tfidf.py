import pandas as pd
import jieba
from joblib import dump
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer

train_data = pd.read_csv('C:/Users/Administrator/Desktop/研究生/ai2026/AI大模型学习/Week04/Week04作业/assets/dataset/Simplified_Chinese_Multi-Emotion_Dialogue_Dataset.csv', sep=',', header=None)

cn_stopwords = pd.read_csv('C:/Users/Administrator/Desktop/研究生/ai2026/AI大模型学习/Week04/Week04作业/assets/dataset/baidu_stopwords.txt', header=None)[0].values

train_data[0] = train_data[0].apply(lambda x: " ".join([x for x in jieba.lcut(x) if x not in cn_stopwords]))

tfidf = TfidfVectorizer(ngram_range = (1,1) )

train_tfidf = tfidf.fit_transform(train_data[0])

model = LinearSVC()
model.fit(train_tfidf, train_data[1])
#变量保存
dump((tfidf, model), "C:/Users/Administrator/Desktop/研究生/ai2026/AI大模型学习/Week04/Week04作业/assets/weights/tfidf_ml.pkl") # pickle 二进制
