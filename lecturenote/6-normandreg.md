- Noramlization 主要是为了提升 training
- Regularization 主要是为了 generalization

# Normalization

一般来说，我们想要让 Loss 函数在远离最优解的地方陡峭 (steep)，在接近最优解的地方平缓 (flat)，从而让优化算法更快收敛。

对于 real-valued 任务，我们最终输出会使用 identity function $f(x)=x$ 以及 $\mathcal{L}_2$ ，对于 classification 任务，我们使用 softmax 函数和 KL 散度。这样对于不同的任务，都会有 $\nabla_{\mathbf{z}} \mathrm{Loss}(\hat{\mathbf{y}}, \mathbf{y})=(\hat{\mathbf{y}}-\mathbf{y})^\top$ 其中 $\mathbf{z}$ 是模型的最后一层输出前的仿射值。 

## Batch Normalization

mini-batch 基于一个假设：mini-batch 内数据的分布和整体数据的分布类似。但是实际上每个 mini-batch 都都会有不一样的分布，也就是带有一个 "covariate shift"。这个 shift 可能会很大，从而影响训练效果。

要做的是将 mini-batch 的数据“移动到某个一样的位置”，从而减少 covariate shift。

具体而言，我们要让均值 $\mu=0$，方差为 $\sigma^2=1$。然后再移动到某一个位置，对于一个 mini-batch $\mathcal{B}=\{\mathbf{z}_1, \mathbf{z}_2, \ldots, \mathbf{z}_m\}$，我们计算：
$$
\begin{aligned}
\mu_{\mathcal{B}} &= \frac{1}{m} \sum_{i=1}^m \mathbf{z}_i \\
\sigma^2_{\mathcal{B}} &= \frac{1}{m} \sum_{i=1}^m (\mathbf{z}_i - \mu_{\mathcal{B}})^2 \\
\mathbf{u}_i &= \frac{\mathbf{z}_i - \mu_{\mathcal{B}}}{\sqrt{\sigma^2_{\mathcal{B}} + \epsilon}} \\
\hat{\mathbf{z}}_i &= \gamma \mathbf{u}_i + \vec\beta
\end{aligned}
$$

一般来说，Batch Normalization 会放在仿射变换之后，激活函数之前。是一个 unit-specific（不知道怎么翻译） 的操作。

所有的 $\mathbf{z}$ 对所有的 $\mathbf{u}$ 都有影响，但是 $\mathbf{u}$ 只会影响各自的 $\hat{\mathbf{z}}$。$\gamma, \vec\beta$ 是可学习的参数，用来控制最终的均值和方差。

$$
\frac{\mathrm{d}~\mathrm{Loss}}{\mathrm{d}\vec{\beta}} = \frac{\mathrm{d}~\mathrm{Loss}}{\mathrm{d}\hat{\mathbf{z}}}, \quad
\frac{\mathrm{d}~\mathrm{Loss}}{\mathrm{d}\gamma} = \mathbf{u} \frac{\mathrm{d}~\mathrm{Loss}}{\mathrm{d}\hat{\mathbf{z}}}
$$

这样会让 Loss 函数的输入除了每个 item 本身的输入，还有 $\mu_{\mathcal{B}}, \sigma^2_{\mathcal{B}}$，从而使 mini-batch 的 Loss 变为：

$$
\mathrm{Loss}(\mathcal{B}) = \frac{1}{m}\sum_{i=1}^m \mathrm{Loss}(f(\mathbf{x}_i, \mu_{\mathcal{B}}, \sigma^2_{\mathcal{B}}), \mathbf{y}_i)
$$

接下来推导一下 $\frac{\mathrm{d}~\mathrm{Loss}}{\mathrm{d}\mathbf{z}_i}$：

$$
\begin{aligned}
\frac{\mathrm{d}~\mathrm{Loss}}{\mathrm{d}\mathbf{z}_i} &= 
	\sum_{j=1}^m
	\frac{\mathrm{d}~\mathrm{Loss}}{\mathrm{d}\mathbf{u}_j}
	\begin{cases}
		\frac{\partial \mathrm{u}_j}{\partial \mu_{\mathcal{B}}} \frac{\partial \mu_{\mathcal{B}}}{\partial \mathbf{z}_i} +
		\frac{\partial \mathrm{u}_j}{\partial \sigma^2_{\mathcal{B}}} \frac{\partial \sigma^2_{\mathcal{B}}}{\partial \mathbf{z}_i} +
		\frac{\partial \mathrm{u}_j}{\partial \mathbf{z}_i} & j = i \\
		\frac{\partial \mathrm{u}_j}{\partial \mu_{\mathcal{B}}} \frac{\partial \mu_{\mathcal{B}}}{\partial \mathbf{z}_i} +
		\frac{\partial \mathrm{u}_j}{\partial \sigma^2_{\mathcal{B}}} \frac{\partial \sigma^2_{\mathcal{B}}}{\partial \mathbf{z}_i} & j \neq i
	\end{cases} \\
	&= \sum_{j=1}^m \frac{\mathrm{d}~\mathrm{Loss}}{\mathrm{d}\hat{\mathbf{z}}_j} \gamma
	\left(
		-\frac{1}{m \sqrt{\sigma^2_{\mathcal{B}} + \epsilon}}
		-\frac{(\mathbf{z}_j - \mu_{\mathcal{B}})^\top}{m(\sigma^2_{\mathcal{B}} + \epsilon)^{3/2}} (\mathbf{z}_i - \mu_{\mathcal{B}})
		+ \frac{\mathbb{I}(i=j)}{m \sqrt{\sigma^2_{\mathcal{B}} + \epsilon}}
	\right) \\
	&= \frac{\gamma}{\sqrt{\sigma^2_{\mathcal{B}} + \epsilon}}
	\left(
		\frac{\mathrm{d}~\mathrm{Loss}}{\mathrm{d}\hat{\mathbf{z}}_i}
		- \frac{1}{m}\sum_{j=1}^m \frac{\mathrm{d}~\mathrm{Loss}}{\mathrm{d}\hat{\mathbf{z}}_j}
		- \frac{(\mathbf{z}_i - \mu_{\mathcal{B}})^\top}{\sigma^2_{\mathcal{B}} + \epsilon} \cdot \frac{1}{m} \sum_{j=1}^m
		\frac{\mathrm{d}~\mathrm{Loss}}{\mathrm{d}\hat{\mathbf{z}}_j} (\mathbf{z}_j - \mu_{\mathcal{B}})
	\right)
\end{aligned}
$$

可以发现，当 mini-batch 中的样本非常相似或者相同的时候，Batch Normalization 的梯度会变得很小。

在最后推理的时候，我们需要使用训练集的均值和方差，作为上面的 $\mu_{\mathcal{B}}, \sigma^2_{\mathcal{B}}$。具体如下:

$$
\mu_{\text{BN}} = \frac{1}{N_{\text{mini-batch}}} \sum_{\mathcal{B}} \mu_{\mathcal{B}}, \quad
\sigma^2_{\text{BN}} = \frac{m}{(m-1)N_{\text{mini-batch}}} \sum_{\mathcal{B}} \sigma^2_{\mathcal{B}}
$$

Batch normalization 只可以放置在某些层，甚至某些单元上。可以提升收敛速度和神经网络的性能。可以减少 dropout 的使用。

需要更高的初始学习率，学习率衰减和更好的训练数据的随机重排。

# Regularization

一般来说，Training Sample 只能是训练空间的一个很小一部分，模型如果过于复杂，就会过拟合训练数据，从而在测试数据上表现不好。Regularization 的目标就是通过 Smoothness 来提高泛化能力。

具体操作方式是，在 Loss 函数中加入一个正则项 (regularization term)：

$$
\mathrm{Loss}_{\text{total}} = \mathrm{Loss}_{\text{original}} + \lambda \sum_{k} \left\Vert W_k \right\Vert_2^2
$$

更新参数的方法更改为：

$$
W_k^{(t)} = W_k^{(t-1)}
 - \lambda W_k^{(t-1)}
 - \eta \left(\nabla_{W_k} \mathrm{Loss}_{\text{original}}\left(W_k^{(t-1)}\right)\right)^\top
$$

这种正则项实际上是 $\mathcal{L}_2$ 正则化 (也叫 weight decay)，可以防止参数过大，从而防止过拟合。

# Bagging

Bagging 是 Bootstrap Aggregating 的简称。Bagging 的目标是通过集成多个模型来减少方差 (variance)，从而提升泛化能力。

取出一部分训练数据 (有放回采样)，训练多个模型，然后将多个模型的预测结果进行平均 (回归任务) 或投票 (分类任务)。

Bagging 可以减少模型的方差，从而提升泛化能力。适用于高方差模型 (如决策树)。

# Dropout

设置一个 probability $p$，对于每个神经元，在训练过程中，有 $1-p$ 的概率被“丢弃”，即不参与前向传播和反向传播。实际是，如果一个 $p$-Bernoulli 变量 $d$ 为 0，则该神经元被丢弃；如果为 1，则该神经元被保留。

通过Dropout，整个模型变成了一个集成了多个子模型的平均值，有类似 Bagging 的效果。有减少过拟合的效果。

Dropout 强制让神经元学习 "rich" 且 "redundant" 的表示，从而提升泛化能力。