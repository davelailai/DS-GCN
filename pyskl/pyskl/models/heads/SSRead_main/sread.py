import torch
import torch.nn.functional as F

from gread import global_add_pool, global_mean_pool, global_max_pool
from gread import GlobalAttention, Set2Set

class SemanticReadout(torch.nn.Module):
    def __init__(self, emb_dim = 300, read_op = 'sum', num_position = 4, gamma = 0.01):

        super(SemanticReadout, self).__init__()
        
        self.emb_dim = emb_dim
        self.read_op = read_op
        self.num_position = num_position
        self.gamma = gamma

        self.protos = torch.nn.Parameter(torch.zeros(num_position, emb_dim), requires_grad=True)
        torch.nn.init.xavier_normal_(self.protos.data)
    
        if read_op == 'sum':
            self.gread = global_add_pool
        elif read_op == 'mean':
            self.gread = global_mean_pool 
        elif read_op == 'max':
            self.gread = global_max_pool 
        elif read_op == 'attention':
            self.gread = GlobalAttention(gate_nn = torch.nn.Linear(emb_dim, 1))
        elif read_op == 'set2set':
            self.gread = Set2Set(emb_dim, processing_steps = 2) 
        else:
            raise ValueError("Invalid graph readout type.")

    def init_protos(self, protos):
        self.protos.data = protos

    def get_alignment(self, x):
        D = self._compute_distance_matrix(x, self.protos)
        A = torch.zeros_like(D).scatter_(1, torch.argmin(D, dim=1, keepdim=True), 1.)
        return A 

    def get_aligncost(self, x, batch):
        D = self._compute_distance_matrix(x, self.protos)
        if self.gamma == 0:
            D = torch.min(D, dim=1)[0]
        else:
            D = -self.gamma * torch.log(torch.sum(torch.exp(-D/self.gamma), dim=1) + 1e-12)
       
        N = torch.zeros(D.shape[0], batch.max().item() + 1, device=D.device).scatter_(1, batch.unsqueeze(dim=1), 1.)
        N /= N.sum(dim=0, keepdim=True)
        return torch.sum(D / N.sum(dim=1), dim=0)
        
    def forward(self, x, batch):
        size = batch.max().item() + 1

        A = self.get_alignment(x)
        sbatch = self.num_position * batch + torch.max(A, dim=1)[1]
        ssize = self.num_position * (batch.max().item() + 1)

        x = self.gread(x, sbatch, size=ssize)
        x = x.reshape(size, self.num_position, -1)
        return x.reshape(size, -1)

    def _compute_distance_matrix(self, h, p):
        h_ = torch.pow(torch.pow(h, 2).sum(1, keepdim=True), 0.5)
        p_ = torch.pow(torch.pow(p, 2).sum(1, keepdim=True), 0.5)
        hp_ = torch.matmul(h_, p_.transpose(0, 1))
        hp = torch.matmul(h, p.transpose(0, 1))
        return 1 - hp / (hp_ + 1e-12)

