# Traffic Behavior Analysis using Graph Convolutional Networks

This repository is under the documentation requirement for my undergraduate study titled "Traffic Behavior Analysis using Graph Convolutional Networks", an overview of the
literature of the graph convolutional network (GCN) with respect to the spectral and spatial-based architecture of the method. 

The codes mainly used in this study are adapted from [1] which are _train.py, test.py, dbgcn_utils.py,_ and _gcns.py_. The modified _train.py_ and _dbgcn_utils.py_ can be found under [src/training](https://github.com/hundround/gcn-traffic-behavior-analysis/tree/main/src/training). The remaining GCN codes are unedited, hence
left cited from the [luansen/congestion_propagation_analysis](https://github.com/luansenda/congestion_propagation_inference) repo. The public data used in this study can be found in [zhu/diffusion_attack](https://github.com/LYZ98/diffusion_attack) repo.  

Some the sample codes implementing convolution with a kernel, specifically, Sobel filter, can be found in [miguelmota/sobel](https://github.com/miguelmota/sobel). In [chokkan/deeplearning/blob/master/notebook/convolution](https://github.com/chokkan/deeplearning/blob/master/notebook/convolution.ipynb), distinct convolution feature map outputs are demonstrated by varying kernel matrices.

(This repository is currently under editing by the author.)

### Reference:

[1] Luan, Sen, et al. "Traffic congestion propagation inference using dynamic Bayesian graph convolution network." _Transportation research part C: emerging technologies 135_ (2022): 103526.  
[2] Zhu, Lyuyi, et al. "Adversarial diffusion attacks on graph-based traffic prediction models." _IEEE Internet of Things Journal_ (2023).
