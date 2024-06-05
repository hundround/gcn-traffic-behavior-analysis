
---
Here is an example of a graph, of a simple social network:
```tikz

```

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
\node[input] (input1) at (0, 2) {$a$};
\node[input] (input2) at (0, 0) {$b$};
\node[input] (input3) at (0, -2) {$c$};

% Neuron nodes
\node[neuron] (neuron1) at (3, 2) {$d$};
\node[neuron] (neuron2) at (3, 0) {$e$};
\node[neuron] (neuron3) at (3, -2) {$f$};

% Output node
\node[output] (output) at (6, 0) {$g$};

% Connect the nodes with arrows
\foreach \i in {1,2,3}
    \foreach \j in {1,2,3}
        \draw[arrow] (input\i) --  (neuron\j);

\foreach \i in {1,2,3}
   \draw[arrow] (neuron\i) -- (output);

\end{tikzpicture}

\end{document}
```

Let the adjacency matrix be $A$ where
$$A = [x_{ij}]$$ where $x_{ij}$ is the (i,j)-th entry of the matrix. $i,j=a,b,c,d,e,f,g$. 
To convert this graph to its corresponding adjacency matrix, we have the following: 
		a. Node $e$ is a "destination" node of $a,b,c$. Then, we have the entries $x_{ea,eb,ec}=1$ so as $x_{da,db,dc}$ and $x_{fa,fb,fc}$. 
		b. Nodes $x_{aj}$, $x_{bj}$, and $x_{cj}$ are all 0 since they are not the destination nodes of any possible source nodes. 

Hence, we have the adjacency matrix:
$$A = 
\begin{pmatrix}
0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 \\
1 & 1 & 1 & 0 & 0 & 0 & 0 \\
1 & 1 & 1 & 0 & 0 & 0 & 0 \\
1 & 1 & 1 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 1 & 1 & 1 & 0
\end{pmatrix}
$$
Note: too big. 