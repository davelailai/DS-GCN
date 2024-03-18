_base_ = ['../_init_/lr_schedual.py']
model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='STGCN',
        gcn_adaptive='init',
        tcn_type='unitmlp', # 'unit_tcn', 'mstcn', 'unit_tcnedge', 'unitmlp', 'msmlp'
        tcn_add_tcn=True,
        tcn_merge_after=True,
        # gcn_num_types=5,
        # gcn_reduce=8,
        # gcn_edge_num=15,
        # # graph_cfg=dict(layout='nturgb+d', mode='stgcn_spatial'),
        # num_stages=10,
        # inflate_stages=[5, 8],
        # down_stages=[5, 8],
        graph_cfg=dict(layout='nturgb+d', mode='random', num_filter=3, init_off=.04, init_std=.02),
        # graph_cfg=dict(layout='nturgb+d', mode='stgcn_spatial'),
        # tcn_ms_cfg=[(3, 1), (3, 2), (3, 3), (3, 4), ('max', 3), '1x1']
        ),
    cls_head=dict(type='GCNHead', num_classes=60, in_channels=256))
    # neck = dict(type='PretrainNeck', in_channels=256, read_op='attention', num_position=25),
    # neck = dict(type='SimpleNeck', in_channels=256, mode='GCN'),
    # cls_head=dict(
    #     type='ClsHead', num_classes=60, in_channels=256,
    #     #loss_cls=[dict(type='CrossEntropyLoss',loss_weight=1.0)]
    #     ))
#load_from='./work_dirs_pretrain/stgcn_read/stgcn_pyskl_ntu60_xsub_3dkp/j/epoch_20.pth'
