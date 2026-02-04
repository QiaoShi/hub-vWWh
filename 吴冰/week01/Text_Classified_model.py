import jieba
import pandas as pd
from fastapi import FastAPI
from openai import OpenAI
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier

dataset = pd.read_csv("dataset.csv", sep='\t', header=None, nrows=10000)
# print(dataset[1].value_counts())

input_sentence_jieba = dataset[0].apply(lambda x: ' '.join(jieba.lcut(x)))
'''
.apply()返回的仍然是 Pandas Series 对象， 包含额外的元数据（索引、名称等）

input_sentence_jieba 是一个 Series，索引是行号，值是分词后的字符串
input_sentence_jieba = [
0    还有 双鸭山 到 淮阴 的 汽车票 吗 13 号 的
1                    从 这里 怎么 回家
2     随便 播放 一首 专辑 阁楼 里 的 佛里 的 歌]

input_sentence_jieba.values 返回的是 NumPy 数组， 没有任何 Pandas 特有的元数据， 确保 scikit-learn 能稳定地处理输入数据
input_sentence_jieba.values = [
    "还有 双鸭山 到 淮阴 的 汽车票 吗 13号 的", "从 这里 怎么 回家", "随便 播放 一首 专辑 阁楼 里 的 佛里 的 歌"]
'''

vector = CountVectorizer()
vector.fit(input_sentence_jieba.values)

'''
CountVectorizer()：词频向量化器（默认按空格/标点分词，统计词频）。
vector.fit()：学习词汇表（建立“字典”：词 → 索引）。
    输入参数：input_sentence_jieba.values = [ "还有 双鸭山 到 淮阴 的 汽车票 吗 13号 的",
                                            "从 这里 怎么 回家", 
                                            "随便 播放 一首 专辑 阁楼 里 的 佛里 的 歌"]
    参数要求: 类数组（array-like）或稀疏矩阵
    处理过程：按空格拆分成词语列表：["从", "这里", "怎么", "回家"]
            去重，按首次出现的顺序编号
    输出字典：{
            '还有': 0, '双鸭山': 1, '到': 2, '淮阴': 3, '的': 4, '汽车票': 5,
            '吗': 6, '13号': 7, '从': 8, '这里': 9, '怎么': 10, '回家': 11,
            '随便': 12, '播放': 13, '一首': 14, '专辑': 15, '阁楼': 16,
            '里': 17, '佛里': 18, '歌': 19} 
 
print("词汇表:", vector.vocabulary_)  # 打印词汇表（词→索引的映射）
print("词汇表大小:", len(vector.vocabulary_))  # 词汇表大小

'''

input_feature = vector.transform(input_sentence_jieba.values)

'''
vector.transform() : 按照学过的词汇表把文本转成数字向量。
    输入参数：input_sentence_jieba.values = [ "还有 双鸭山 到 淮阴 的 汽车票 吗 13号 的",
                                            "从 这里 怎么 回家", 
                                            "随便 播放 一首 专辑 阁楼 里 的 佛里 的 歌"]
    处理过程：统计每个词出现的次数，形成向量（长度20）
            文本1 "还有 双鸭山 到 淮阴 的 汽车票 吗 13号 的"
            词频："还有":1, "双鸭山":1, "到":1, "淮阴":1, "的":2, "汽车票":1, "吗":1, "13号":1，其余 0。
            向量 =[1, 1, 1, 1, 2, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]        
    输出矩阵： [[1 1 1 1 2 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0]
              [0 0 0 0 0 0 0 0 1 1 1 1 0 0 0 0 0 0 0 0]
              [0 0 0 0 2 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1]]                   
                                        
'''
model = KNeighborsClassifier()
model.fit(input_feature, dataset[1].values)

'''
KNeighborsClassifier() : sklearn库中实现的 k-近邻分类算法，属于监督学习中的基于实例的学习方法。
model.fit(特征数据，标签数据):
    输入：input_feature：3×20 的词频矩阵
         dataset[1].values：["Travel-Query", "Travel-Query", "Music-Play"]
    处理过程：只是储存，没有训练
        input_feature：model._fit_X
        dataset[1].values：model._y
'''


client = OpenAI(
    api_key="*********************",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

app = FastAPI()


@app.get("/text-cls/ml")
def text_classify_using_ml(text: str) -> str:
    test_sentence = ' '.join(jieba.lcut(text))
    test_feature = vector.transform([test_sentence])
    return model.predict(test_feature)[0]


'''
model.predict()
    假设来了一条新文本，经过CountVectorizer处理的特征向量 x_new，调用 model.predict([x_new])：
    计算 x_new与 model.X_中所有 3 条训练样本的距离（欧氏距离）。
    按距离从小到大排序，取前 k=3个邻居（这里就是全部训练样本）。
    统计这 3 个邻居的标签：
    如果 x_new与文本1、文本2 更近 → 邻居标签 = ['Travel-Query', 'Travel-Query', 'Music-Play']
    多数票（2 比 1）→ 预测为 Travel-Query。
    返回预测结果。
'''


@app.get("/text_cls/llm")
def text_classify_using_llm(text: str) -> str:
    completion = client.chat.completions.create(
        model="qwen-flash",
        messages=[
            {"role": "user", "content": f"""帮我进行文本分类：{text}
输出的类别只能从如下中进行选择， 除了类别之外下列的类别，请给出最合适的类别。
FilmTele-Play            
Video-Play               
Music-Play              
Radio-Listen           
Alarm-Update        
Travel-Query        
HomeAppliance-Control  
Weather-Query          
Calendar-Query      
TVProgram-Play      
Audio-Play       
Other
"""}
        ]
    )
    return completion.choices[0].message.content


print("这是深度学习的输出结果：", text_classify_using_ml("播放钢琴曲命运交响曲"))
print("这是大语言模型的输出结果：", text_classify_using_llm("播放钢琴曲命运交响曲"))
