This project is generated based on the the pyskl (https://github.com/kennymckormick/pyskl)

installation:

git clone https://github.com/davelailai/DS-GCN.git

cd DS-GCN

conda env create -f pyskl.yaml

conda activate pyskl

pip install -e .

Run our method :


bash tools/dist_test.sh  configs/dsstgcn/ntu60_xsub_3dkp/j.py

please cite the paper: 
Xie, J., Meng, Y., Zhao, Y., Nguyen, A., Yang, X., & Zheng, Y. (2024). Dynamic Semantic-Based Spatial Graph Convolution Network for Skeleton-Based Human Action Recognition. Proceedings of the AAAI Conference on Artificial Intelligence, 38(6), 6225-6233. https://doi.org/10.1609/aaai.v38i6.28440

