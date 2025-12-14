# Learning

一个模型实际上一个 parametric function $f(x; \theta)$ ，用于拟合一个函数 $g(x)$ ，拟合的过程就是 Learning 。

拟合的最终目的则是：

$$
\begin{aligned}
\text{totalerr}(\theta) &= \int_{-\infty}^{\infty} \mathrm{error}(f(X; \theta), g(x)) \mathrm{d} X \\
\hat{\theta} &= \arg\min_{\theta} \{\text{totalerr}(\theta)\}
\end{aligned}
$$

但是一般来说我们拿不到整个函数的图像，只有一些 samples, 所以退而求其次，使用如下函数：

$$
\text{EmpiricalError}(\theta) = \frac{1}{N} \sum_{i=1}^N \mathrm{error}(f(X_i;\theta), d_i)
$$

最终目标是使得 $\text{EmpiricalError}(\theta)=0$ 

## Error Function

### Classification Error

$$
\text{error}(a,b) = \mathbb{I}(a\ne b)
$$

## Perceptron Learning Algorithm

给定 $N$ 个 samples, $(X_1, y_1), \dots, (X_N, y_N)$ ，其中 $y_i\in \{-1, 1\} ~\forall i\in \{1, \dots, N\}$ .

进行如下操作：

$$
\begin{aligned}
&\text{Initialize}~\theta \\
&\text{cycle until}~\textbf{no error} \\
&~~ \text{for}~i\leftarrow 1\dots N \\
&~~ ~~ O(X_i)=\mathrm{sign}(\theta^\top X_i) \\
&~~ ~~ \text{if}~O(X_i)\ne y_i \\
&~~ ~~ ~~ W\leftarrow W + y_i X_i
\end{aligned} 
$$

这个方法只适用于 linear seperable 的数据集。

此后，我们使用 differential 的 error function 和 activative function.

这里规定一下，$\nabla_X f(X)$ 形状和 $X^T$ 相同。

- **turning point**: $\nabla_X f(X) = 0$ 且 $\nabla^2_X f(X)>0$

用 $\mathrm{div}(a,b)$ 表示 $a$, $b$ 之间的 divergence. 那么学习的目标就是 ：

$$
\hat{\theta}=\arg\min_{\theta} \int_{-\infty}^{\infty} \mathrm{div}(f(X;\theta), y)P(y)\mathrm{d} X
$$

但是实际上有

$$
\begin{aligned}
\mathbb{E}[\mathrm{div}(f(X;\theta), g(X))] &= \int_X \mathrm{div}(f(X;\theta),g(X))P(X)\mathrm{d} X \\
&\approx \frac{1}{N}\sum_{i=1}^N \mathrm{div}(f(X_i;\theta), d_i)
\end{aligned}
$$

让 $\mathrm{Loss}(\theta) = \frac{1}{N}\sum_{i=1}^N \mathrm{div}(f(X_i;\theta), d_i)$ ，那么要做的事情就是：$\hat{\theta} = \arg\min_{\theta} \mathrm{Loss}(\theta)$

## Gradient Descent

下面的算法用于达到最小值。

$$
\begin{aligned}
&\text{Initialize}~x^{(0)} \\
&\text{cycle until}~\textbf{no error} \\
&\text{for}~t\leftarrow 1\dots \infty~\text{until meet criteria}\\
&~~ x^{(t+1)}\leftarrow x^{(t)} - \eta^{(t)}\nabla_x f\left(x^{(t)}\right) \\
\end{aligned} 
$$

**criteria**:
1. $\left|f\left(x^{(t+1)}\right) - f\left(x^{(t)}\right)\right|\leq \epsilon_1$ or 
2.  $\left|\nabla_x f\left(x^{(t)}\right)\right|\leq \epsilon_2$

一般来说 $f=\text{Loss}$ .