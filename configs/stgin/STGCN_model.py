
_base_ = ['../_init_/lr_schedual.py']
model = dict(
    type='RecognizerGCNR',
    backbone=dict(
        type='STGIN',
        gcn_type='unit_gcnedge',
        tcn_type='unit_tcnedge',
        graph_cfg=dict(layout='nturgb+d', mode='stgcn_spatial'),
        in_channels = 3),
    # neck = dict(type='PretrainNeck', in_channels=256, read_op='attention', num_position=25),
    neck = dict(type='SimpleNeck', in_channels=256, mode='GCN'),
    cls_head=dict(
        type='ClsHead', num_classes=60, in_channels=256,
        # loss_cls=[dict(type='CrossEntropyLoss',loss_weight=1.0)]
        ))
# load_from = './work_dirs_pretrain/stgcn_read/stgcn_pyskl_ntu60_xsub_3dkp/j/epoch_20.pth'