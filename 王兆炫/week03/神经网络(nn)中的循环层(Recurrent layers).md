# 神经网络(nn)中的循环层(Recurrent layers)

`RNN / LSTM / GRU` 都是在时间上一步步处理序列数据的算子。

三者总体对应于pytorch中的`torch.nn.RNNBase` , RNNBase是框架内部的基类, 封装并且抽象出RNN-family(RNN , LSTM , GRU)的具体行为

它们每一步都用当前输入和上一步的“记忆”（隐藏状态）计算出新的隐藏状态；区别在于 如何控制记忆的保留与更新（LSTM/GRU 用“门”来做控制，RNN 则没有门）。

---
## recurrent neural network：RNN 循环神经网络

把当前输入和上步隐藏状态通过线性变换后加起来,而后再通过激活函数得到新的隐藏状态

$$
h_t = \tanh(W_{ih} x_t + b_{ih} + W_{hh} h_{t-1} + b_{hh})
$$

此处的激活函数选为tanh , $x_t$代表当前输入 , $h_{t-1}$为上一步的隐藏状态,即记忆 ; W,b为对应变换矩阵

### nn.RNN

`nn.RNN` -> 将整个RNN过程封装起来的module,参数如下 :

> torch.nn.RNN(input_size, hidden_size, num_layers=1, nonlinearity='tanh', bias=True, batch_first=False, dropout=0.0, bidirectional=False, device=None, dtype=None)

其中`num_layers`可以选择RNN的层数 , 由于RNN的特性,层数一般不超过3层, 且当层数为3时需搭配Dropout,这么做是为了避免梯度消失


> "Dropout层" : "随机失活层,是对具有深度结构的人工神经网络进行优化的方法，在学习过程中通过将隐含层的部分权重或输出随机归零，降低节点间的相互依赖性（co-dependence ）从而实现神经网络的正则化（regularization），降低其结构风险（structural risk）。
e.g. nn.Dropout(0.5)   ->每次训练时用50%的能力去训练,期望是每个子模型都表现良好"

> "梯度消失": " 在层数较多的神经网络中，根据链式法则，梯度在反向传播时需要连续乘以上一层隐藏层的导数及权重，如果这些数值小于1，会导致梯度乘积趋近于0。表现为训练初期 Loss（损失）下降较快，但随后停滞不前，模型收敛困难，准确率无法提升。"

其余参数可以参考[pytorch官方文档下RNN用法](https://docs.pytorch.org/docs/2.4/generated/torch.nn.RNN.html#torch.nn.RNN)

### nn.RNNcell

`nn.RNNcell` 则是针对单个时间步的RNN单元（输入一对 ($x_t$, $h_{t-1}$)，返回 $h_t$）,只封装单步的权重与运算,不管理num_layers和双向(bidirectional),以及无法执行dropout

> torch.nn.RNNCell(input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype=None)

## LSTM : Long Short-Term Memory 循环神经网络的改进版 

> torch.nn.LSTM(input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0.0, bidirectional=False, proj_size=0, device=None, dtype=None)

### 与RNN的区别 : 

+ RNN 的每步只有一个隐藏态 , 每次更新都是"覆盖式"的 , 新的$h_t$完全由当前输入和上一步隐藏态经过非线性决定，长期记忆容易丢失（梯度消失）
+ 而LSTM给出了一个独立的记忆路径cell state , 每次更新都是加法式的 , 可以视为保留了每一步的信息
+ LSTM具有gates机制 : forget gates ; input gates ; output gates 门是可学习的 `sigmoid` 单元，用来控制读/写/保留多少信息。门的存在使得网络可以学习在什么时候保留长期信息、什么时候写入新信息。

### 同样有nn.LSTMCell , 类比上方RNNcell

## nn.GRU 

> torch.nn.GRU(input_size, hidden_size, num_layers=1, bias=True, batch_first=False, dropout=0.0, bidirectional=False, device=None, dtype=None)

### GRU : Gate Recurrent Unit

相比LSTM，使用GRU能够达到相当的效果，并且相比之下更容易进行训练，能够很大程度上提高训练效率，因此很多时候会更倾向于使用GRU。

GRU把forget和update合并成一个update门 , 不需要单独维护cell , 降低了计算的复杂度(在参数上也能看出来GRU明显减少)
