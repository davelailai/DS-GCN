_base_ = ['../_init_/lr_schedual.py']
model = dict(
    type='RecognizerGCN',
    backbone=dict(
		type='CTRGCN',
        gcn_type = 'unit_ctrhgcn', # 'unit_ctrgcn', 'unit_ctrhgcn'

        gcn_node_attention = True,
        gcn_edge_attention = True,
        gcn_add_type = False,
        gcn_ada = True,
        gcn_num_types=5,
        gcn_rel_reduction=8,
        gcn_edge_num=15,
    
        num_stages=10,
        inflate_stages=[5, 8],
        down_stages=[5, 8],
        tcn_type='msmlp', # 'unit_tcn', 'mstcn', 'unit_tcnedge', 'unitmlp', 'msmlp'
        tcn_add_tcn=True,
        tcn_merge_after=True,
        # semantic_stage=[1, 2, 3, 4],
        graph_cfg=dict(layout='nturgb+d', mode='random', num_filter=3, init_off=.04, init_std=.02)),
        # graph_cfg=dict(layout='nturgb+d', mode='spatial')),
    cls_head=dict(type='GCNHead', num_classes=60, in_channels=256))