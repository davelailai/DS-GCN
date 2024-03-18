import torch
import torch.nn.functional as F
from torch_geometric.nn.inits import uniform

from conv import GNN_node
from gread import GlobalReadout
from sread import SemanticReadout

class GNN(torch.nn.Module):

    def __init__(self, gnn_type, num_classes, num_nodefeats, num_edgefeats,
                    num_layer = 5, emb_dim = 300, drop_ratio = 0.5,  
                    read_op = 'sum', num_position = 4, gamma = 0.01):
        super(GNN, self).__init__()

        self.num_classes = num_classes
        self.num_nodefeats = num_nodefeats
        self.num_edgefeats = num_edgefeats        

        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.emb_dim = emb_dim
        self.rep_dim = emb_dim * num_position 

        self.read_op = read_op
        self.num_position = num_position
        self.gamma = gamma

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        ### GNN to generate node-level representations
        self.gnn_node = GNN_node(num_nodefeats, num_edgefeats, num_layer, emb_dim, drop_ratio = drop_ratio, gnn_type = gnn_type)

        ### Readout layer to generate graph-level representations
        self.read = SemanticReadout(self.emb_dim, read_op=self.read_op, num_position=self.num_position, gamma=self.gamma)
        
        if read_op == 'set2set': self.rep_dim *= 2
        self.graph_pred_linear = torch.nn.Linear(self.rep_dim, self.num_classes)

    def forward(self, batched_data):
        h_node = self.gnn_node(batched_data)
        h_graph = self.read(h_node, batched_data.batch)
        return self.graph_pred_linear(h_graph)

    def get_embedding(self, batched_data):
        h_node = self.gnn_node(batched_data)
        return self.read(h_node, batched_data.batch)

    def get_alignment(self, batched_data):
        h_node = self.gnn_node(batched_data)
        return self.read.get_alignment(h_node)

    def get_aligncost(self, batched_data):
        h_node = self.gnn_node(batched_data)
        return self.read.get_aligncost(h_node, batched_data.batch)

if __name__ == '__main__':
    GNN(num_classes = 10)
