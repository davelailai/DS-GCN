# SSRead: Structural Semantic Readout Layer for Graph Neural Networks

- This is the author code of ["Learnable Structural Semantic Readout for Graph Classification (ICDM 2021)"](https://arxiv.org/abs/2111.11523).

## Overview

<p align="center">
<img src="./figure/example.png" width="1000">	
</p>

Visualization of our structural semantic alignment (K=4) and other hierarchical graph poolings (N'=4). The structural positions and node-clusters (or selected nodes) are represented in different colors.


## Run the codes

- python
- torch (GPU version only)
- torch_geometric

```
python main.py --dataset <tu-dataset-name> --device <gpu-device-idx>
```
