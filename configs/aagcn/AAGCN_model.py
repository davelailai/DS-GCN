_base_ = ['../_init_/lr_schedual.py']
model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='AAGCN',
        gcn_attention = False,
        gcn_type= 'unit_aahgcn', # 'unit_aagcn', 'unit_aahgcn'
        # gcn_node_att = True,
        # gcn_edge_att = True,
        gcn_adaptive=True,  # init, offset, importance
    
        tcn_type='unitmlp', # 'unit_tcn', 'mstcn', 'unit_tcnedge', 'unitmlp', 'msmlp'
        tcn_add_tcn=True,
        tcn_merge_after=True,
        graph_cfg=dict(layout='nturgb+d', mode='random', num_filter=3, init_off=.04, init_std=.02)),
        
    cls_head=dict(type='GCNHead', num_classes=60, in_channels=256))