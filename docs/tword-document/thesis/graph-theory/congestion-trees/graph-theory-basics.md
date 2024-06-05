---
tags:
  - thesis
  - graph-theory
  - congestion-trees
---
---
This section discusses the graph-theoric definitions for the prerequisites of *Congestion Trees*

```tikz
\usetikzlibrary{positioning}

\begin{document}
\begin{tikzpicture}[>=stealth, dot/.style={circle, fill, inner sep=2pt}]

    % Nodes
    \node[dot] (v1) at (0,0) {};
    \node[dot] (v2) at (2,1.5) {};
    
    % Edge
    \draw[-] (v1) -- (v2) node[midway, right] {};
    
    % Labels for vertices
    

\end{tikzpicture}
\end{document}

```

`Definition 01.`  graph, denoted by $G = (V,\varepsilon)$ where V is the set of vertices $\{m_i\}_{i=1}^n$ and $\varepsilon$ is the set of edges $\{l_j\}_{j=1}^{n'}$ , is consists of finite set of vertices and edges joining different pairs of distinct vertices.

Above is a figure of a graph with a pair of vertices and an edge joining them.

`Definition 02.` A subgraph $A$  of a graph $B$ is a graph whose set of vertices and edges belong on the set of vertices and edges of $B$. 

`Definition 03.` We say that a graph is *connected* if any vertices can be reached from all the other vertices on the graph.

```tikz
\usetikzlibrary{positioning}

\begin{document}
\begin{tikzpicture}[>=stealth, dot/.style={circle, fill, inner sep=1.5pt}]

    % Nodes
    \node[dot] (v1) at (0,0) {};
    \node[dot] (v2) at (2,1.5) {};
    \node[dot] (v3) at (4,0) {};
    
    % Edges
    \draw[-] (v1) -- (v2) node[midway, above] {};
    \draw[-] (v2) -- (v3) node[midway, above] {};

\end{tikzpicture}
\end{document}

```

The figure above is an example of a connected graph. Any vertex can be reached from any other vertices.

`Definition 05.` We say a graph is complete if every pair of vertices is adjacent to each other. 

```tikz
\usetikzlibrary{positioning}

\begin{document}
\begin{tikzpicture}[dot/.style={circle, fill, inner sep=1.5pt}]

    % Nodes
    \node[dot] (v1) at (0,0) {};
    \node[dot] (v2) at (2,1.5) {};
    \node[dot] (v3) at (4,0) {};
    
    % Edges
    \draw (v1) -- (v2);
    \draw (v2) -- (v3);
    \draw (v1) -- (v3);

\end{tikzpicture}
\end{document}

```

The *connected* graph example from before is not complete since the two endpoints are not adjacent to each other (in other words, no edge joins them.)

The triangle as shown is a complete graph. We see that every pair of vertices is *jointed* if you will. 

`Definition 06. Tree` 
	As one of the main definitions of this section, we say that a graph is a tree if for a pair of vertices $V_1$, there is a unique edge associated to $V_1$. That is, a tree is an acyclic connected graph. 

The example shown for the complete graph is a tree since there is a unique edge that connects every pair of vertices. An example below is a graph that is not a tree since no edge is associated with the pair of vertices from the lower side of the graph.


```tikz
\usetikzlibrary{positioning}

\begin{document}
\begin{tikzpicture}[dot/.style={circle, fill, inner sep=1.5pt}]

    % Nodes
    \node[dot] (v1) at (0,0) {};
    \node[dot] (v2) at (1,1.5) {};
    \node[dot] (v3) at (2,0) {};
    \node[dot] (v4) at (1,-1.5) {};
    
    % Edges
    \draw (v1) -- (v2);
    \draw (v2) -- (v3);
    \draw (v3) -- (v1);
    \draw (v1) -- (v4);

\end{tikzpicture}
\end{document}

```

`Definition 06.1` A tree of graph $G = (V, \varepsilon)$ may contain edge that is not present in the graph $G$. 

`Definition 07.` A spanning tree of a connected graph $G = (V,\varepsilon)$ is a tree such that $T = (V,\varepsilon')$ with $\varepsilon' \subseteq \varepsilon$. 

`Definition 08.` The *degree* $d_v$ of a vertex $v\in V$ is the number of edges incident (or associated) to $v$.

`Definition 09.` The *indegree* of a vertex $v\in V$ is the number of edges directed **to** $v$. 

For a directed edge, the destination is called a children node or vertex while the starting node is called its parent. 




