# Image Processing
## Vision Architecture Basics: Vision Transformer (ViT)

将图像转换为若干个向量。

ViT 工作流程：$\text{Image}\rightarrow\text{patches}\rightarrow\text{vectors}$

![ViT convert images to vectors](figs/9-vitimg2vec.png)

然后将 patch embedding 输入到 transformers encoder 中，得到向量表示 $z_1,\dots,z_n$ 

网络深度越大，模型关注的区域越大。

## Learning image representations: Language-Image Pre-training (CLIP)

公用一个 embedding space 来让一个模型来同时学习图像和文本表示。

使用两个 encoder $f_I(x)\rightarrow z_I,f_T(x)\rightarrow z_T$ 分别对图像和文本进行编码。

- 成对的图像和文本的向量应该很相近
- 不成对的就应该很远

$$
L=-\frac{1}{2}\sum_{n=1}^N\left[ \log\frac{\exp\left(f_I(x_n)^\top f_T(y_n)\right)}{\sum_{j=1}^N\exp\left(f_I(x_j)^\top f_T(y_n)\right)} + \log\frac{\exp\left(f_I(x_n)^\top f_T(y_n)\right)}{\sum_{j=1}^N\exp\left(f_I(x_n)^\top f_T(y_j)\right)}\right]
$$

就是一个 cross-entropy loss 。

# Multi-to-text

## Combining with a language model: Llava

使用 CLIP/ViT 将图像转换为向量序列之后，将向量序列处理为 token/embedding ，然后和普通的文本 token 一起处理。

可以对 CLIP/ViT 的输出使用线性变换，转换到 embedding space 中。

e.g. MOLMO (AI2)

# Multi-to-Image

使用模型生成图片是一种 sampling 操作。（嗯嗯，不知道为啥这样写）

## Autoregressive

将图像视为像素序列

任务变成：

$$
\begin{aligned}
&x_{\text{img}}\in \mathbb{R}^{H\times W\times C}\rightarrow x_1, \dots, x_T \\
\mathcal{L}_{\text{MLE}}&=\sum_{t=1}^T -\log p_{\theta}(x_t|x_{<t})
\end{aligned}
$$

challenges: 序列长度

e.g. 
1. PixelRNN：一个RNN，使用先前的所有像素来生成当前像素，可以直接使用大量的图像进行训练，不需要标记。
2. Image Transformer：需要使用一种记忆力效率更到的注意力模式

e.x. WaveNet for Audio

![WaveNet](figs/9-wavenet.png)

## Variational Auto-encoder (VAE)

### Auto-encoder

一个最简单的 Autoencoder (使用线性激活函数，只有一个向量)，所有的点被映射到一条直线上。

![Simplest Autoencoder](figs/9-simpleae.png)

如果含有多个 $w$，也依然是一种 PCA 。

在中间添加非线性激活函数，可以获得一个 nonlinear PCA 。而增加隐藏层可以捕捉更复杂的非线性 manifold 。

当 hidden state 的维度小于输入，也被称为 bottleneck 网络。

而 decoder 部分只能生成位于训练学到的 manifold 上的数据，任何被输入到 hidden state 的数据都会被映射到和训练集中的点很相似的位置上。因此可以当成一个 数据集上的分布的生成器。

![Autoencoder](figs/9-ae.png)

回到生成问题上来，我们需要模型学会某类图像的 distribution ，然后从这个 distribution 中采样一张图。
- 这里基于假设：数据的分布在高维空间中是一个 non-linear/linear manifold
- The principal components of all instances of the target class of data lie on this manifold

那我们可以使用上面的 decoder, 给他输入任意东西，就会输出一张看起来很合理的图。

但是有一个问题：对于在训练集上的数据，ae会表现得不错，但是如果不在训练集上，那么输出就会很奇怪，而且不可预测。

![Problem of AE](figs/9-aeproblem.png)

这需要让输入的 $z$ 符合 $z$ 的自然分布 (natural distribution)，也就是让 $z$ 是 typical 的，但我们并不知道 $z$ 的分布。

但是，我们可以通过模型、训练数据、学习过程来定义 $z$ ，因此可以直接施加一个分布给 $z$ 。

当我们训练 AE 的时候，可以直接给一个限制 $z$ ，说 $z$ 符合某个分布，典型的比如 $\mathcal{N}(0,I)$ ，一个标准正态分布。高斯分布是 least informative distribution ，因此相对于同样方差的其他分布，有更高的 entropy 。那么应当如何训练模型，使得这个 $z$ 符合分布？

考虑到高斯分布的特殊性：
1. 每个变量独立
2. negative log probability 和 变量的 square norm $\Vert \mathbb{x}\Vert^2$ 成正比。

因此最小化一个 KL 散度就可以了，或者说最小化 $z$ 的 square norm 就可以了

$$
\begin{aligned}
\min_{\theta,\phi} \sum_{x} |x-\hat{x}|^2+\lambda \Vert\mathbb{E}[x;\theta]\Vert^2
\end{aligned}
$$

但是有个问题，这个计算不能完全捕捉到方差，只能输出在一个 manifold 上。如果高维空间中的 $z$ 代表一个二维高斯分布，那么所有 $z$ 就代表一个平面——一个平面上的高斯分布。解码器将此平面映射到高维空间中的某种 manifold。任何输入到解码器的样本都只能位于这个平面上，无法离开这个平面。实际数据的维度就不得而知了。

而且这个 manifold 不能捕捉到所有我们需要生成的图像的信息。也就是说，即使我们成功得知了 manifold 的维度，自然数据中也依然包含一些相对于这个 manifold 的 variation （也可以说是噪声）。自然数据不会完全落在 manifold 上，只能说很接近 (lie close)

一种改进方式是：在 Decoder 输出后面添加一个噪声。

然后模型变成了这样子：生成一个 $z$ ，扔到 Decoder 里面，生成了一个东西，然后添加一个你设置的 noise 。

$$
\begin{aligned}
&e\sim P(e) \\
&z\sim\mathcal{N}(0,I)\rightarrow D(z;\phi)\rightarrow \hat{x} \\
&\hat{x}+e\rightarrow x 
\end{aligned}
$$

这里假设 noise 是一个 $\mathcal{N}(0,C)$ ，而一个经典假设是 $C=c\cdot \mathrm{diag}(\mathbb{1})$ 并且和 $z$ 无关。

训练目标变成了：

$$
\arg\min_{\phi}\mathbb{E}\left[\Vert D(z;\phi)-x\Vert^2\right]
$$

训练时候，我们需要一个 $z$，但是我们不知道 $z$ 是什么。这时候可以使用 encoder 来帮助估计 $z$ 

![AE-training](figs/9-aetraining.png)

因为噪声的存在，对于给定的 $x$ ，不能确定唯一的 $z$ ，但是对于给定的 $x$ ，$z$ 满足一个分布，我们要 encoder 做的就是学习每个 $x$ 对应的 $z$ 的分布，让输出的 $z$ 最有可能通过 decoder 和 noise 获得 $x$ 。这个过程应用了 Variational Lower Bound 的最大值。

## Structure of The Variational Autio-encoder (VAE)

![VAE Structure](figs/9-vaestruct.png)

- Encoder 计算给定的点 $x$ 的 $z$ 的概率分布： $P(z|x)$ 
- Decoder 尝试从 $P(z|x)$ 中得到的 $z$ 的期望值转换为 $x$ 

### Encoder–Decoder 的联合训练目标
VAE 中 encoder 与 decoder 需要同时训练：
- **Encoder**：对每个输入 $x$，输出一个潜变量的条件分布 $Q(z|x)$，用于近似真实后验 $P(z|x)$。
- **Decoder**：从潜变量 $z$ 生成数据 $x$，即建模 $P(x|z)$。

与普通 auto-encoder 不同，encoder 并不输出一个确定的 latent code，而是输出一个 **分布**，从而允许不确定性与随机性。

---

### 带噪声的生成模型直观
假设 decoder 的输出并非精确落在低维流形上，而是：

$$
x = D(z) + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, C),
$$

其中 $C$ 是全秩但幅度较小的协方差矩阵。

这意味着：
- 数据分布集中在某个低维流形附近
- 但在高维空间中仍然是连续的、可建模的概率分布

---

### 变分推断与 ELBO
我们希望最小化 encoder 分布与真实后验之间的 KL 散度：

$$
\mathrm{KL}\big(Q(z|x)\,\|\,P(z|x)\big).
$$

利用贝叶斯公式：

$$
P(z|x) = \frac{P(x|z)P(z)}{P(x)},
$$

并展开 KL 散度，可以得到（忽略与参数无关的 $\log P(x)$）：

$$
\mathcal{L}(x)
=
\mathbb{E}_{z\sim Q(z|x)}[-\log P(x|z)]
+
\mathrm{KL}\big(Q(z|x)\,\|\,P(z)\big).
$$

这一定义了 VAE 的 **训练损失函数**（负的 ELBO）：
- 第一项：**重建误差**（训练 decoder）
- 第二项：**KL 正则项**（约束 encoder 输出分布）

---

### KL 项的意义
KL 正则项

$$
\mathrm{KL}\big(Q(z|x)\,\|\,P(z)\big)
$$

鼓励 encoder 输出的分布 $Q(z|x)$ 接近先验分布 $P(z)$。这样可以保证：
- 不同样本对应的 latent 分布不会彼此孤立
- 整体 latent space 结构良好、可采样

---

### 先验分布 $P(z)$ 的选择
通常选择

$$
P(z) = \mathcal{N}(0, I),
$$

原因包括：
- 各向同性、无偏
- 在固定方差下熵最大（least informative）
- 数学形式简单，KL 项可解析计算
- 对应的负对数概率与 $||z||^2$ 成正比

---

### Encoder 的参数化形式
真实后验 $P(z|x)$ 通常难以计算，因此假设近似后验为对角高斯：

$$
Q(z|x) = \mathcal{N}\big(z;\,\mu(x), \mathrm{diag}(\sigma^2(x))\big).
$$

Encoder 网络输出：
- 均值向量 $\mu(x)$
- 方差向量 $\sigma^2(x)$（或 $\log \sigma^2(x)$）

这将学习一个复杂分布的问题，转化为学习有限维网络参数的问题。

---

### Decoder 的角色
Decoder 定义条件分布 $P(x|z)$，形式取决于数据类型：
- 连续数据：高斯分布
- 二值图像：Bernoulli 分布

Decoder 本质上是一个从 latent space 到 data space 的生成网络。

---

### 训练过程（单个样本）
对每个输入 $x$：
1. Encoder 输出 $\mu(x), \sigma(x)$，构造 $Q(z|x)$  
2. 从 $Q(z|x)$ 中采样 $z$  
3. 计算重建误差 $-\log P(x|z)$  
4. 计算 KL 正则项 $\mathrm{KL}(Q(z|x)\|P(z))$  
5. 最小化两项之和，更新 encoder 与 decoder 参数  

（实现中通常使用 reparameterization trick 以支持反向传播）

---

### 生成阶段
训练完成后：
- **丢弃 encoder**
- 从先验分布采样 $z \sim \mathcal{N}(0, I)$
- 通过 decoder 生成样本：
- 
$$
\hat{x} \sim P(x|z)
$$

这使得 VAE 成为一个真正的 **生成模型**。

---

### 潜变量空间的直观理解
由于：
- encoder 输出的是分布而非点
- KL 项将所有 $Q(z|x)$ 拉向统一先验

latent space 会形成一个：
- 连续
- 连通
- 易于插值和采样

的表示空间，从而保证随机采样的 $z$ 大概率对应“合理”的生成样本。

---

### 小结
VAE 的核心思想可以概括为：
- 用 encoder 学习 $Q(z|x)$ 近似后验
- 用 decoder 学习 $P(x|z)$ 进行生成
- 用 KL 正则保证 latent space 结构良好

其损失函数为：

$$
\mathbb{E}_{z\sim Q(z|x)}[-\log P(x|z)]
+
\mathrm{KL}\big(Q(z|x)\,\|\,P(z)\big),
$$

从而实现 **可训练、可采样的概率生成模型**。