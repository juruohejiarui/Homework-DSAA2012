# CNN

对于序列和空间扫描，而且内容和位置没有太大关系，只和局部趋势（形状）有关的时候，我们可以复用一个小的神经网络，然后组合成一个大的。

分开多层有如下优点，而且可以使用更少的参数：

![Why Distribution](figs/7-whydistribution.png)

卷积神经网络包含 卷积层 (Convolution Layer) 和池化层 (Pooling Layer) ，卷积和池化交替进行。

卷积层的结构如下，包含若干个 kernel 。
在进行卷积之前可能需要进行 padding 操作，将一个 $w\times h$ 矩阵拓展成一个 $(w+2a)\times(h+2b)$ 的矩阵，或者将 长度是 $l$ 的序列变为长度为 $l+2a$ 的序列。具体来说，就是在边框填上指定宽度的 $0$ 。

![Convolution Layer](figs/7-convlayer.png)

池化层有多种选项，Max Pooling / Mean Pooling 比较常见。此外，还有 $p$-norm : $y=\left(\frac{1}{w_k h_k} \sum_{i,j} x_{i,j}^p\right)^{\frac{1}{p}}$ ，或者是一个小的神经网络。

Downsampling （下采样）可以通过一个步长为 $S$ 的卷积/池化操作实现。

Upsampling 就是对于每个点直接添加 $S-1$ 个空行/空列，或者设置一个 $S<1$ 

# RNN

之前提到的神经网络都属于 Finite Response System，而 Infinite Response System 的定义如下：

![Infinite Response System](figs/7-infressys.png)

## NARX

抽取一下关键点：每一个输入都会对之后的输出有影响。NARX 的每一个输入包含完整的过去的信息。注意，输出的 $Y_{1\cdots t-1}$ 也会被用于 $Y_t$ 的计算。

NARX 结构图如下：

![NARX](figs/7-narx.png)

NARX 可以用于
- 天气预报
- 股票市场
- 追踪系统的替换建模
- 语言模型

PS: “memory” of the past is in the output itself, and not in the network .

分析一下这个网络要计算的信息：

$$
\begin{aligned}
m_t &= r(y_{t-1}, h_{t-1}, m_{t-1}) \\
h_t &= f(x_t,  m_t) \\
y_t &= g(h_t)
\end{aligned}
$$

其中 $m_t$ 是关于过去的 "memory" ，然后使用过去的记忆和当前的输入获取一些信息，然后再根据信息进行推理，得到 $y_t$ .

## Jordan Network

使用一个固定的计算方式来获取 "memory" （不可学习）

![Jordan Network](figs/7-jordannet.png)

## Single Hidden Layer RNN

$$
\begin{aligned}
h_t &= f(x_t, h_{t-1}) \\
y_t &= g(h_t)
\end{aligned}
$$

或者使用如下的方式：

$$
\begin{aligned}
\mathbf{h}^{(1)}(t)&=f_1\left(W^{(1)}\mathbf{x}(t) + W^{(11)} \mathbf{h}^{(1)}(t-1) + \mathbf{b}^{(1)}\right) \\
\mathbf{y}(t)&=f_2\left(W^{(2)}\mathbf{h}^{(1)}(t) + \mathbf{b}^{(2)}\right) \\
\end{aligned}
$$

其中 $h^{(1)}(-1)$ 可以作为模型参数的一部分。

### Training of Single Hidden Layer RNN

现在假设 $\mathrm{div}$/$\mathrm{Loss}$ 是一个输入为 $\mathrm{y}_{1\dots T}$ ，输出为实数的函数。

令 $\mathbf{z}^{(1)}(t)=W^{(1)}\mathbf{x}(t) + W^{(11)} \mathbf{h}^{(1)}(t-1) + \mathbf{b}^{(1)}$, $\mathbf{z}^{(2)}(t)=W^{(2)}\mathbf{h}^{(1)}(t) + \mathbf{b}^{(2)}$ 。

那么有:

$$
\begin{aligned}
\nabla_{\mathbf{z}^{(2)}(t)}\mathrm{Loss} &= \nabla_{\mathbf{y}(t)}\mathrm{Loss} \nabla_{\mathbf{z}^{(2)}(t)} \mathbf{y}(t) \\
\nabla_{\mathbf{h}^{(1)}(t)}\mathrm{Loss} &= \nabla_{\mathbf{z}^{(2)}(t)}\mathrm{Loss}\times W^{(2)} + \nabla_{\mathbf{z}^{(1)}(t+1)}\mathrm{Loss}\times W^{(11)} \\
\nabla_{\mathbf{z}^{(1)}(t)}\mathrm{Loss} &= \nabla_{\mathbf{h}^{(1)}(t)}\mathrm{Loss} \nabla_{\mathbf{z}^{(1)}(t)}\mathbf{h}^{(1)}(t) \\
\nabla_{W^{(2)}}\mathrm{Loss} &\xleftarrow {+} \mathbf{h}^{(1)}(t) \nabla_{\mathbf{z}^{(2)}(t)}\mathrm{Loss} \\
\nabla_{W^{(11)}}\mathrm{Loss} &\xleftarrow {+} \mathbf{h}^{(1)}(t-1) \nabla_{\mathbf{z}^{(1)}(t)}\mathrm{Loss} \\
\nabla_{W^{(1)}}\mathrm{Loss} &\xleftarrow {+} \mathbf{x}(t) \nabla_{\mathbf{z}^{(1)}(t)}\mathrm{Loss} \\
\nabla_{h^{(1)}(-1)}\mathrm{Loss} &= \nabla_{\mathbf{z}^{(1)}(0)}\mathrm{Loss} \times W^{(11)}
\end{aligned}
$$

## Multiple Recurrent Layer RNN

最简单也是最常用的模型：

$$
\begin{aligned}
\mathbf{h}^{(0)}(t) &= \mathbf{x}(t) \\
\mathbf{h}^{(i)}(t)&=f_1\left(W^{(i)}\mathbf{h}^{(i-1)}(t) + W^{(ii)} \mathbf{h}^{(i)}(t-1) + \mathbf{b}^{(i)}\right) \\
\mathbf{y}(t)&=f_2\left(W^{(n+1)}\mathbf{h}^{(n)}(t) + \mathbf{b}^{(n+1)}\right) \\
\end{aligned}
$$

这里假设 RNN 有 $n$ 个隐藏层。

## Bidirectional RNN

一个正向，一个反向。当整个序列都可见的时候比较实用。有正反交错和独自计算两种：

![Bidirectional RNN](figs/7-birnn.png)

对于正反都是单层的，首先从 $T\rightarrow 0$ 计算 $\mathbf{h}_f$ 的梯度，然后从 $0\rightarrow T$ 计算 $\mathbf{h}_b$ 的梯度，当然两者可以互换。

## Long Short-Term Memory (LSTM)

设置一个特别的 LSTM Cell，这个 Cell 接受三个输入：1. $\mathbf{x}(t)$; 2. $\mathbf{h}(t-1)$, state; 3. $\mathbf{c}(t-1)$ linear history ，输出为 $\mathbf{h}(t)$, $\mathbf{c}(t)$ 。使用 Sigmoid 作为 Gate Function $\sigma$.

每一个 Gate 都有对应的名称：
- $f_t$ ： "forget" gate，需要携带多少的 history
- $i_t$ ： "input" gate，表示新的输入是否值得被记忆
- $o_t$ ： "output" gate，表示形成的记忆有多少需要展示到输出中

计算式如下：

$$
\begin{aligned}
\mathbf{i}_t &= \sigma (W_i[\mathbf{c}_{t-1};\mathbf{h}_{t-1};\mathbf{x}_t] + \mathbf{b}_i) \\
\mathbf{f}_t &= \sigma (W_f[\mathbf{c}_{t-1};\mathbf{h}_{t-1};\mathbf{x}_t] + \mathbf{b}_f) \\
\mathbf{o}_t &= \sigma (W_o[\mathbf{c}_t; \mathbf{h}_{t-1};\mathbf{x}_t] + \mathbf{b}_o) \\
\tilde{\mathbf{c}}_t &= \tanh (W_c[\mathbf{h}_{t-1};\mathbf{x}_t] + \mathbf{b}_c) \\
\mathbf{c}_t &= \mathbf{f}_t\odot \mathbf{c}_{t-1} + \mathbf{i}_t \odot \tilde{\mathbf{c}}_t \\
\mathbf{h}_t &= \mathbf{o}_t \odot \tanh (\mathbf{c}_t)
\end{aligned}
$$
![LSTM Cell](figs/7-lstmcell.png)