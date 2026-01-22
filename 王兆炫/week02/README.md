>Task01
调整 09_深度学习文本分类.py 代码中模型的层数和节点个数，对比模型的loss变化。

代码中对应于模型每层节点数的参数为: hidden_dim ; 所以只改变hidden_dim的大小可以查看节点数对模型的影响

因为每次运行都有10次epoch,在整个过程中loss逐渐降低,模型参数实现优化,所以在此处只比较最后一次loss的改变

以下是几次运行结果的对比:

hidden_dim=128 : Epoch [10/10], Loss: 0.5766

hidden_dim=64 : Epoch [10/10], Loss: 0.5866

hidden_dim=256 : Epoch [10/10], Loss: 0.5878

hidden_dim=512 : Epoch [10/10], Loss: 0.5877

可知,在保持模型层数为1层的情况下,结点数的增加或减少对于模型的loss没有很明显的影响

修改模型为两层,此处修改SimpleClassifier中实现,改为多层模型

```python
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=None):
        super(SimpleClassifier, self).__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            if dropout_rate is not None and dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev = h
        layers.append(nn.Linear(prev, output_dim))  # 输出层
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
```

使用三层:
hidden_dims = [256, 128,512]   此时为Epoch [10/10], Loss: 1.1572
可以看到loss反而增加

使用三层相同结点数:
```bash
hidden_dims = [256, 256,256]
```
结果为:
```bash
Epoch [10/10], Loss: 1.5615
输入 '帮我导航到北京' 预测为: 'HomeAppliance-Control'
输入 '查询明天北京的天气' 预测为: 'Calendar-Query'
```

可以看到lose较高,并且两个输出的例子都是有误的

所以可以得出结论:

**当前情况下,受限于当前模型的选择/数据的特点/模型训练方式,更适合的方式是采用浅(单)层训练,单层训练可以使模型结果表现较优,但层数变多时反而由于过拟合导致训练不稳定,结果变差**

>Task02
调整 06_torch线性回归.py 构建一个sin函数，然后通过多层网络拟合sin函数，并进行可视化。

这部分代码在`hw02.py`文件中可查看,列出几个修改部分的说明:

+ 原代码逻辑为`y=ax+b` ,为线性拟合,无法拟合`sinx`这种非线性曲线,所以修改为`sin`
+ 多个线性函数的叠加结果依然是线性函数,所以这里引入了`Tanh`函数,其输出在[-1,1]间的S型曲线,增加函数层数的同时也就意味着需要多层叠加,删去原来的a,b;修改为`torch.nn.Linear(1, 64)`构成的全连接层
+ 由于使用多层网络,所以将optimizer从`SGD`修改为`Adam`,保证在多层网络下拟合结果更优更稳定


同时附拟合结果图片

<img width="1000" height="600" alt="Figure_1" src="https://github.com/user-attachments/assets/f177bf0e-2870-4c9e-a7f4-b5c2a53784a0" />




