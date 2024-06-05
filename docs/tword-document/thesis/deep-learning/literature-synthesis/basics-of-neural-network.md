---
tags:
  - thesis
  - background
---
```
Dan Mark Orapa
University of the Philippines Diliman
Neural Network and Deep Learning 

01. Basics of Neural Network

| yay my first note in neural network adfhfdahadkfhad
```

`01. NEURON and NEURAL NETWORK` $N$ is a single output of size n that corresponds to the sum of every input variable $x_i$ of the same size with the following equation
$$N = \sigma\left(\sum_{i=1}^nx_iw_i + b\right) = \sigma(x^Tw + b)$$
$b$ is the bias that is determined and updated every after backpropagations, and $w$ is the weights for each input $x_i$. In a simple neural network, $\sigma : \mathbb{R} \rightarrow \mathbb{R}$ i.e., the inputs $(x_i,w,b)$ are scalar and so is the output $(N)$. 

```tikz
\usetikzlibrary{positioning, arrows.meta}

\begin{document}

\begin{tikzpicture}[
    % Define styles for the different components
    input/.style={draw, circle, minimum size=8mm, inner sep=0pt},
    neuron/.style={draw, circle, minimum size=8mm, inner sep=0pt},
    output/.style={draw, circle, minimum size=8mm, inner sep=0pt},
    arrow/.style={->, >=stealth, shorten >=1pt, thick},
]

% Input nodes
\node[input] (input1) at (0, 2) {$x_1$};

% Neuron nodes
\node[neuron] (neuron1) at (3, 2) {$N_1$};

% Output node
\node[output] (output) at (6, 2) {$O$};

% Connect the nodes with arrows
\foreach \i in {1}
    \foreach \j in {1}
        \draw[arrow] (input\i) -- node[midway, above] {$w_{1\i}^2$} (neuron\j);

\foreach \i in {1}
    \draw[arrow] (neuron\i) -- node[midway, above] {$w_{1\i}^3$} (output);

\end{tikzpicture}

\end{document}
```

Now, don't be confused. $x$ as a vector is composed of $(x_1,x_2,\ldots, x_n)$ and so is $w = (w_1,w_2,\ldots, w_n)$. We denote these vectors as elements of a field (set of numbers that can be added, multiplied, and divided), with notation $x,w\in \mathbb{R}^n$. So that $x^Tw$ means a dot product of $x$ and $w$ which is a scalar. 

Bias $b^{\ell}$ indicates the bias for each layer $\ell$. The weight $w^{\ell}_i$ indicates the weight of node whose layer it goes at is $\ell$ (i.e., the node on the left attached to the layer neuron $\ell$.) 

```tikz
\usetikzlibrary{positioning, arrows.meta}

\begin{document}

\begin{tikzpicture}[
    % Define styles for the different components
    input/.style={draw, circle, minimum size=8mm, inner sep=0pt},
    neuron/.style={draw, circle, minimum size=8mm, inner sep=0pt},
    output/.style={draw, circle, minimum size=8mm, inner sep=0pt},
    arrow/.style={->, >=stealth, shorten >=1pt, thick},
]

% Input nodes
\node[input] (input1) at (0, 2) {$x_1$};
\node[input] (input2) at (0, 0) {$x_2$};
\node[input] (input3) at (0, -2) {$x_3$};

% Neuron nodes
\node[neuron] (neuron1) at (3, 2) {$N_1$};
\node[neuron] (neuron2) at (3, 0) {$N_2$};
\node[neuron] (neuron3) at (3, -2) {$N_3$};

% Output node
\node[output] (output) at (6, 0) {$O$};

% Connect the nodes with arrows
\foreach \i in {1,2,3}
    \foreach \j in {1,2,3}
        \draw[arrow] (input\i) --  (neuron\j);

\foreach \i in {1,2,3}
    \draw[arrow] (neuron\i) -- node[midway, above] {$w_{1\i}^3$} (output);

\end{tikzpicture}

\end{document}
```

Here, we have the input $x = (x_1,x_2,x_3)$ and the neurons $N_1,N_2,N_3$ considered at the 2nd layer of the network. Moreover, $O$ is the output which is the weighted sum of each neuron with respective weights $w_1^3,w_2^3,w^3_{3}$. That is, going in the third (superscript) layer from the first (subscript) neuron of the second layer.

We represent the weights in between layers 1 and 2 (the arrow web in between $x$ and $N$s) by a matrix called the weight matrix. That is, 

$$
W^{\ell}
= 
\begin{bmatrix} w_{11}^{\ell} &w_{12}^{\ell}&w_{13}^{\ell} \\ w_{21}^{\ell} & w_{22}^{\ell} & w_{23}^{\ell}  \\ w_{31}^{\ell} & w_{32}^{\ell} & w_{33}^\ell \end{bmatrix}
$$

by which $w_{11}^{\ell}$ is the weight from the first input variable $x_1$ going into the first neuron $N_1$, $w_{23}^{\ell}$ is the weight from the third input variable $x_3$ going into the second neuron $N_2$. So one can see that the first column vector of $W^{\ell}$ denoted by $w^{\ell}_{i1}$ is a vector of all the weights coming from the first input.  

We also represent the "collective" output of activation function as a vector. That is, the neuron vector $N = (N_1,N_2,N_3)$ (in our case diagram above) is an output of processing the input $x$ to the activation function $\sigma$. This is sometimes denoted as $A^{\ell}$ which is the activation output of layer $\ell$. To visualize, we have
$$\sigma(z) = \sigma\left(
	\begin{matrix}
	 z_{1} = x^{T}w_{1j}^{\ell} + b_1 \\
	z_{2} =x^{T}w_{2j}^{\ell} + b_2  \\ 
	z_{3} = x^{T}w_{3j}^{\ell} + b_3
	\end{matrix}
	
\right)
= A^{\ell}.
$$
And so, in a general fashion,
$$
A^{1}=\sigma(A^{0}W^{1} + b^{1})

$$
where $A^{0}= x = (x_1,x_2,\ldots,x_n)$ or the input vector. 
An example of activation function is $ReLU$ which is a piecewise function 
$$\sigma_{ReLU}(z) = \max\{0,z\} = \begin{cases} 0 & z < 0  \\
z & z \geq 0 \end{cases}$$
`02. COST FUNCTIONS` The cost functions are used to determine whether the output values are considered "close" enough to the actual values. That is, numerically it *verdicts* the outputs of the neural network.

The most common is the Mean Square Error (MSE) that, in its namesake, determines the *difference* between the output of the neural network and the actual data values. This *difference* does not directly mean difference, but the gap which can be determine by
$$
MSE = \dfrac{1}{k}\sum\limits_{i=1}^m(\hat{y}-y)^2
$$
where $k = f(m)$ where $f$ is a linear function (for instance $f(m) = 2m$). It depends on the user's preference actually. Take MIT Professor Winston, who doesn't like the how performance function looks like if $k > 0$. It looks like this

```tikz
\usetikzlibrary{positioning, arrows.meta}
\begin{document}
\begin{tikzpicture}
  % Axes
  \draw[->] (-3,0) -- (3,0) node[right] {$x$};
  \draw[->] (0,-1) -- (0,5) node[above] {$y$};

  % Quadratic function
  \draw[black, thick, domain=0:2, samples=100] plot (\x,{\x^2});
	\draw[black, thick, domain=-2:0, samples=100] plot (\x,{-\x^2});
\end{tikzpicture}
\end{document}

```
which does not capture (for him) if a neural network performs well or not. Since if the neural network performs well, then the MSE should approach 0, in this graph it goes down (which is like negatively perceived) so have instead a $k < 0$ so that we have this

```tikz
\usetikzlibrary{positioning, arrows.meta}
\begin{document}
\begin{tikzpicture}
  % Axes
  \draw[->] (-3,0) -- (3,0) node[right] {$x$};
  \draw[->] (0,-5) -- (0,1) node[above] {$y$};

  % Quadratic function
  \draw[black, thick, domain=0:2, samples=100] plot (\x,{-\x^2});
	\draw[black, thick, domain=-2:0, samples=100] plot (\x,{\x^2});
\end{tikzpicture}
\end{document}

```
Now, we shall see that if a neural network performs well, then it goes up there to 0 (lmao). 

Cost functions are also called *Performance* function to determine how well the neural network is learning the data, sort of like its "performance" **(bad-dum-tss)**. 

`03. JACOBIAN` The *Jacobian* is a function that takes in a matrix (of function inputs) and outputs another matrix which is the partial derivatives of such functions. 
$$
J(A) = \begin{pmatrix}
\partial f_1/\partial v_1 & \partial f_1/\partial v_2 &\cdots& \partial f_1/\partial v_n \\
\partial f_2/\partial v_1 & \partial f_2/\partial v_2 &\cdots& \partial f_2/\partial v_n \\
\vdots &\ddots& & \vdots \\
\vdots & & \ddots &  \vdots \\
\partial f_n/\partial v_1 & \cdots & & \partial f_n/\partial v_n
\end{pmatrix}
$$
where
$$
A = \begin{pmatrix}
f_1(v_1,\ldots, v_n) \\
f_2(v_1,\ldots,v_n) \\
\vdots \\
f_n(v_1,\ldots,v_n)
\end{pmatrix}
$$

Remark. The Jacobian of all element-wise functions is a diagonal matrix. We denote it by
$$
J = diag(\gamma_1,\gamma_2,\ldots,\gamma_n)
$$
`04. HADAMARD PRODUCT` One can see that the *Hadamard Product* is the freshman's dream for matrices. That to multiply matrices, a rookie would mistakenly multiply element-wise, which is what Hadamard product does and not the usual matrix multiplication. 

Denote Hadamard Product by $\circ$ with
$$
AB = \begin{pmatrix}
a_1 \\ a_2\\\vdots \\ a_n 
\end{pmatrix}
\circ
\begin{pmatrix}
b_1  \\ b_2  \\ \vdots  \\ b_n
\end{pmatrix}
=
\begin{pmatrix}a_1b_1  \\ a_2b_2 \\ \vdots  \\ a_nb_n\end{pmatrix}
$$

`06. SCALAR EXPANSION` 
Now, we take a scalar $\alpha$ to be multiplied to a matrix, say $A$ as defined above. We have
$$
\alpha A = \alpha I \circ A = \begin{pmatrix}
\alpha a_1 \\ \alpha a_2\\\vdots \\ \alpha a_n 
\end{pmatrix}
$$
where $I = ones(n,1)$. 

`07. NEURON DERIVATIVES` This is a prerequisite to understand how the activation function works (which is to optimize the weights and biases with respect to the neural layer $A^{\ell}$.)

`08. ACTIVATION FUNCTION` 
The activation function works as a function that "activates" a certain neuron whenever it reaches a threshold that depends on the case of which the neural network is applied into. 

Take for example, in traffic congestion. Say, the congestion propagates on a road, but a neural network cannot judge whether a picture of a road is congested or not. Using the activation function, it can numerically determine which road has the congestion propagated into by having a threshold of (thesis ko na to sorry lmao) $\hat{c} = s_i/\overline{s}_i < 0.6$ which is the ratio of actual average speed of vehicles on a road for time $t$ and the average speed for the whole month, say. Now, if $\hat{c}$ reaches $0.6$ or lower, then the activation function activates the neuron that carries this road section, and thus conclude that the specific road section is congested. 

The *ReLU* as discussed above is a that threshold step function which is a problem in gradient descent given that it is discontinuous. 

Another activation function is the *Sigmoid* function defined by
$$
S(z) = \dfrac{1}{1 + e^{-z}}
$$
which should look like this
```tikz
\usetikzlibrary{positioning, arrows.meta}

\begin{document}
\begin{tikzpicture}
  % Axes
  \draw[->] (-5,0) -- (5,0) node[right] {$z$};
  \draw[->] (0,-0.2) -- (0,1.2) node[above] {$\sigma(z)$};

  % Sigmoid function
  \draw[black, smooth, thick, domain=-5:5, samples=100] plot (\x, {1/(1 + exp(-\x))});
\end{tikzpicture}
\end{document}


```
as the input goes large, the value of $e^{-z}$ goes small and hence, the function approaches $1$. Otherwise, the function approaches $0$. The Sigmoid function answers the problem in discontinuity for gradient descent or derivatives alike.

`09. NEURAL NETS` 
We think of training a neural network as finding the *most suitable* neural weights for each input variable so that the neural network, as a whole, behaves and captures what one wants it to do.

```tikz

\usetikzlibrary{positioning, arrows.meta}

\begin{document}
\begin{tikzpicture}

% Nodes for x_i
\foreach \i in {1,2,3} {
    \node[circle, draw, minimum size=8mm] (x\i) at (0, -\i) {$x_\i$};
}

% Box with w_i floating inside
\node[draw, minimum width=2cm, minimum height=2.5cm, anchor=west] (weights) at (3, -2.5) {};
\foreach \i in {1,2,3} {
    \node[anchor=center] (w\i) at (4,-\i-0.5) {$w_\i$};
}

% Nodes for z_i
\foreach \i in {1,2,3} {
    \node[circle, draw, minimum size=8mm] (z\i) at (6, -\i) {$z_\i$};
}

% Arrows from x_i to weights
\foreach \i in {1,2,3} {
    \draw[->] (x\i) -- (weights);
}

% Arrows from weights to z_i
\foreach \i in {1,2,3} {
    \draw[->] (weights) -- (z\i);
}

\end{tikzpicture}
\end{document}

```
where $z = f(x,w,\ldots)$. 

`The BASIC Neural Network`
Going back to this figure from above, we have

```tikz
\usetikzlibrary{positioning, arrows.meta}

\begin{document}

\begin{tikzpicture}[
    % Define styles for the different components
    input/.style={draw, circle, minimum size=8mm, inner sep=0pt},
    neuron/.style={draw, circle, minimum size=8mm, inner sep=0pt},
    output/.style={draw, circle, minimum size=8mm, inner sep=0pt},
    arrow/.style={->, >=stealth, shorten >=1pt, thick},
]

% Input nodes
\node[input] (input1) at (0, 2) {$x_1$};

% Neuron nodes
\node[neuron] (neuron1) at (3, 2) {$N_1$};

% Output node
\node[output] (output) at (6, 2) {$O$};

% Connect the nodes with arrows
\foreach \i in {1}
    \foreach \j in {1}
        \draw[arrow] (input\i) -- node[midway, above] {$w_{1\i}^2$} (neuron\j);

\foreach \i in {1}
    \draw[arrow] (neuron\i) -- node[midway, above] {$w_{1\i}^3$} (output);

\end{tikzpicture}

\end{document}
```

Now, we introduce the activation functions and the performance function. We have


```tikz
\usetikzlibrary{positioning, arrows.meta}

\begin{document}
\begin{tikzpicture}[
    % Define styles for the different components
    input/.style={draw, circle, minimum size=8mm, inner sep=0pt},
    neuron/.style={draw, circle, minimum size=8mm, inner sep=0pt},
    output/.style={draw, circle, minimum size=8mm, inner sep=0pt},
    process/.style={draw, rectangle, minimum size=8mm, inner sep=4pt},
    arrow/.style={->, >=stealth, shorten >=1pt, thick},
]

% Input nodes
\node[input] (input1) at (0, 2) {$x_1$};

% Process node (S)
\node[process] (S1) at (3, 2) {$S$};
\node[process] (S2) at (9, 2) {$S$};

% Neuron nodes
\node[neuron] (input2) at (6, 2) {$y$};

% Output node
\node[output] (output) at (10.5, 2) {$z$};

\draw[arrow] (input1) -- node[midway, above] {$w_{1\i}^{2}\quad p_1$} (S1);

\draw[arrow] (S1) -- (input2);

\draw[arrow] (input2) -- node[midway, above] {$w_{1\i}^{3}\quad p_2$} (S2);

\draw[arrow] (S2) -- (output);
    
\end{tikzpicture}
\end{document}

```

An input $x_1$ goes into the network with a random $w_{11}^2$ weight then goes to the sigmoid activation function, outputs $N_1$ then goes to $z$ which is determined whether the neural network performed well using performance function (in this case, MSE.)

`The Partial Derivatives`
Consider the same neural network


```tikz
\usetikzlibrary{positioning, arrows.meta}

\begin{document}
\begin{tikzpicture}[
    % Define styles for the different components
    input/.style={draw, circle, minimum size=8mm, inner sep=0pt},
    neuron/.style={draw, circle, minimum size=8mm, inner sep=0pt},
    output/.style={draw, circle, minimum size=8mm, inner sep=0pt},
    process/.style={draw, rectangle, minimum size=8mm, inner sep=4pt},
    arrow/.style={->, >=stealth, shorten >=1pt, thick},
]

% Input nodes
\node[input] (input1) at (0, 2) {$x_1$};

% Process node (S)
\node[process] (S1) at (3, 2) {$S$};
\node[process] (S2) at (9, 2) {$S$};

% Neuron nodes
\node[neuron] (input2) at (6, 2) {$y$};

% Output node
\node[output] (output) at (10.5, 2) {$z$};

\draw[arrow] (input1) -- node[midway, above] {$w_{1\i}^{2}\quad p_1$} (S1);

\draw[arrow] (S1) -- (input2);

\draw[arrow] (input2) -- node[midway, above] {$w_{1\i}^{3}\quad p_2$} (S2);

\draw[arrow] (S2) -- (output);
    
\end{tikzpicture}
\end{document}

```


then we have the partial derivatives to optimize the performance function $P$. We have
$$
\dfrac{\partial P}{\partial w_2} = \dfrac{\partial p_2}{\partial w_2}\dfrac{\partial z}{\partial p_2}\dfrac{\partial P}{\partial z}
$$
which determines the how the performance is going to improve with respect to the second weight $w_2$. The same thing goes for $w_1$. We have

$$
\dfrac{\partial P}{\partial w_1} = \dfrac{\partial p_1}{\partial w_1}\dfrac{\partial y}{\partial p_1}\dfrac{\partial p_2}{\partial y}\dfrac{\partial z}{\partial p_2}\dfrac{\partial P}{\partial z}
$$
`Neural "NETWORK"`
Now, expand the neuron we have above to more neurons, and let them interact. Then, the number of weights to determine blows up exponentially. This is how neural network works. 

