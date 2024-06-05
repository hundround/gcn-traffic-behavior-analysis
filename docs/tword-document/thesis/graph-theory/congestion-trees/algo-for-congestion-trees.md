---
tags:
  - thesis
  - congestion-trees
  - background
  - graph-theory
---
---
```
Input: congestion source R^t_s; set of road location information R〈O,D〉
Output: congestion trees, CTrees

Let CTrees = [R^t_s], flag = 1 
Do{ 
	for each road section r in R: 
		find the r that meets: (R_s).O = = r.D 
	if r^(t+1) is in congestion: 
		add r^(t+1) to CTrees 
		Rt s = r^(t+1) 
		flag = 1 
	else: 
		flag = 0 
}while (flag = = 0) 
Return CTrees
```

`Algorithm 01. Congestion Trees Building Process` 
This algorithm is used to construct trees to illustrate graph-theoric congestion propagation for each congestion source on the data given by the user log inputs.

The algorithm requires the inputs R^t_s, the congestion source where the recurrent or incident congestion has happened, and the set of road location information (where the road is located node-wise).

The output is the congestion tree stored in CTrees, a matrix of 1 and 0s.



