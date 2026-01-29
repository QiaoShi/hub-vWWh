## task1 -> 理解rnn、lstm、gru的计算过程

见笔记md文档 , [记录并比较了三种算法](https://github.com/Birchove/ai_learning/blob/main/%E7%8E%8B%E5%85%86%E7%82%AB/week03/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C(nn)%E4%B8%AD%E7%9A%84%E5%BE%AA%E7%8E%AF%E5%B1%82(Recurrent%20layers).md)

## task2 -> 05_LSTM文本分类.py 使用lstm ，使用rnn/ lstm / gru 分别代替原始lstm，进行实验，对比精度

考虑到实际使用rnn , lstm , gru均为调用pytorch中的对应库,并且这几个module到现在也相对完整,所以使用调用实现

详见[三种模型的比较测试](https://github.com/Birchove/ai_learning/blob/main/%E7%8E%8B%E5%85%86%E7%82%AB/week03/RNNBase%E6%96%87%E6%9C%AC%E5%88%86%E7%B1%BB.py)

并且其输出为 : 
```bash
模型 RNN  训练完成，准确率: 0.1334
模型 LSTM 训练完成，准确率: 0.9732
模型 GRU  训练完成，准确率: 0.9876

测试输入: '帮我导航到北京'
[RNN] 预测结果: Video-Play
[LSTM] 预测结果: Travel-Query
[GRU] 预测结果: Travel-Query
```

三个模型训练耗费时间都较长,但是可以看到LSTM以及GRU要明显优于RNN

## task03 -> 阅读项目计划书  + 初步项目代码，写清楚四个模型的优缺点

术语积累 : 

+ 长尾效应 :是指那些原来不受到重视的销量小但种类多的产品或服务由于总量巨大，累积起来的总收益超过主流产品的现象。(图中黄色部分)
<img width="2560" height="1332" alt="image" src="https://github.com/user-attachments/assets/406b19a1-e8fc-4e13-8a0c-32f0ff98de1c" />
在nlp中指较少被用到的词语 , 但是种类又极其多 , 称为长尾词

---

+ 停用词(Stop word) : 指在信息检索、搜索引擎和自然语言处理（NLP）中，为了节省存储空间、提高搜索效率及准确性，而自动过滤的频繁出现的低信息量词汇。它们通常是冠词、介词、连词、助动词或语气词，如“的”、“是”、“on”、“the”等。
---

+ TF-IDF算法 (term frequency–inverse document frequency) :

  TF指词频 , IDF指逆文档频率 , 分别的计算公式为 :
<img width="550" height="155" alt="image" src="https://github.com/user-attachments/assets/64030b35-674e-4859-b377-807fa9eb50e1" />
<img width="550" height="120" alt="image" src="https://github.com/user-attachments/assets/7f3197b0-a92f-4613-8edf-4438754ae6aa" />

最终的结果只需将二者相乘

可以看到TF-IDF值与词频成正比,与文档频率成反比 , 即文章中一个词出现频率越高 , 越重要, 如果词频相同, 那么在语料库中出现越多, 说明这个词越常见 ,即在本文中越不重要



