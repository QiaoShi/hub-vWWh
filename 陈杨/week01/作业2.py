import jieba # 中文分词用途
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer # 词频统计
from sklearn.neighbors import KNeighborsClassifier # KNN
from sklearn import linear_model # 线性模型模块
from sklearn import tree # 决策树模块
from sklearn import svm #svm
from sklearn.ensemble import RandomForestClassifier  #随机森林
from sklearn.naive_bayes import MultinomialNB #朴素贝叶斯
from sklearn.ensemble import GradientBoostingClassifier #梯度提升

dataset = pd.read_csv("dataset.csv", sep="\t", header=None, nrows=10000)

# 提取 文本的特征 tfidf， dataset[0]
# 构建一个模型 knn， 学习 提取的特征和 标签 dataset[1] 的关系
# 预测，用户输入的一个文本，进行预测结果
input_sententce = dataset[0].apply(lambda x: " ".join(jieba.lcut(x))) # sklearn对中文处理
# print("处理后的分词结果：",input_sententce)
vector = CountVectorizer() # 对文本进行提取特征 默认是使用标点符号分词， 不是模型pip install Scikit-learn==1.5.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
vector.fit(input_sententce.values) # 统计词表
input_feature = vector.transform(input_sententce.values) # 100 * 词表大小

test_query = "我想要导航去重庆"
test_sentence = " ".join(jieba.lcut(test_query))
test_feature = vector.transform([test_sentence])
print("待预测的文本", test_query)

model = KNeighborsClassifier()
model.fit(input_feature, dataset[1].values)
print("KNN模型预测结果: ", model.predict(test_feature))

model = tree.DecisionTreeClassifier()
model.fit(input_feature, dataset[1].values)
print("决策树模型预测结果: ", model.predict(test_feature))

model = linear_model.LogisticRegression(max_iter=1000) # 创建一个逻辑回归模型
model.fit(input_feature, dataset[1].values)
print("逻辑回归模型预测结果: ", model.predict(test_feature))

model = svm.SVC()
model.fit(input_feature, dataset[1].values)
print("SVM模型预测结果: ", model.predict(test_feature))

model = RandomForestClassifier()
model.fit(input_feature, dataset[1].values)
print("随机森林模型预测结果: ", model.predict(test_feature))

model = MultinomialNB()
model.fit(input_feature, dataset[1].values)
print("朴素贝叶斯模型预测结果: ", model.predict(test_feature))

model = GradientBoostingClassifier()
model.fit(input_feature, dataset[1].values)
print("梯度提升模型预测结果: ", model.predict(test_feature))
