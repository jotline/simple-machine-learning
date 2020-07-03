# RNN(SRNN)
## 原理
在NLP领域，RNN的重要性无须多言，在自监督模型风靡之前，NLP领域的任务基本上都被RNNs长期霸榜。

RNN的核心其实极其简单：
>把线性层循环起来，前一刻的输出作为下一刻的输入。

这样序列上每个时间点上的特征也就被依次串了起来，下一刻的序列内容可以看到之前所有的状态。同时为了序列中的token可以同时看到前后的信息，也就诞生了bi-RNNs，但是思路是一样的，只不过将序列倒序排列而已。

RNN的隐状态计算公式为：

$h_{t}=tanh(w_{ih}\times x_{t}+b_{ih}+w_{hh}\times h_{t-1}+b_{hh})$

通俗来说也就是：
>序列的每个时刻，分别对上一步的输出和当前时刻的输入分别进行线性变换，并将它们的和用tanh进行激活。

在3这个时刻，RNN看起来就是：
$h_{3}(h_{2}(h_{1}(h_{0},x1),x_{2}),x_{3})$

很容易看到RNN的几个问题：
- 无法看到长距离的依赖，因为序列太长的话，经过了很多次嵌套，影响程度会越来越小。
- 梯度消失，tanh很容易梯度消失，尤其是多个tanh叠加。（其实Relu也没办法解决梯度在长距离传递上的问题）

## 代码
看一下pytorch中的RNN实现，在pytorch中RNN被用闭包层层封装了起来，为了方便说明原理，只挑选核心代码出来解析。

```python
def Recurrent(inner, reverse=False):
    def forward(input, hidden, weight):
        output = []
        steps = range(input.size(0) - 1, -1, -1) if reverse else range(input.size(0)) # steps=[seq_len-1, ...,1,0] or [0,1,...,seq_len-1]
        for i in steps:
            hidden = inner(input[i], hidden, *weight)
            # hack to handle LSTM
            output.append(hidden[0] if isinstance(hidden, tuple) else hidden)

        if reverse:
            output.reverse()
        output = torch.cat(output, 0).view(input.size(0), *output[0].size())

        return hidden, output

    return forward
```

可以看到非常清晰，会正向或者反向遍历整个序列，依次调用inner方法，将当前的输入input[i]和上一步的输出hidden传入。inner方法就是使用上文的隐状态公式进行计算，代码为：

```python
NNReLUCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
    hy = F.relu(F.linear(input, w_ih, b_ih) + F.linear(hidden, w_hh, b_hh))
    return hy

def RNNTanhCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None):
    hy = F.tanh(F.linear(input, w_ih, b_ih) + F.linear(hidden, w_hh, b_hh))
    return hy
```


