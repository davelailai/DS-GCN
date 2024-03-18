_base_ = ['../_init_/lr_schedual.py']
model = dict(
    type='RecognizerGCNPre',
    backbone=dict(
        type='STGCN',
        graph_cfg=dict(layout='nturgb+d', mode='stgcn_spatial'),
        in_channels=4),
    neck = dict(type='PretrainNeck', in_channels=256, read_op='attention', num_position=25),
    # neck = dict(type='SimpleNeck', in_channels=256, mode='GCN'),
    cls_head=dict(
        type='ClsHead', num_classes=60, in_channels=256,
        loss_cls=[dict(type='CrossEntropyLoss',loss_weight=1.0)]))
dataset_type = 'PoseDataset'
ann_file = 'data/nturgbd/ntu60_3danno.pkl'
optimizer = dict(module=dict(type='SGD', lr=0.5, momentum=0.9, weight_decay=0.0005, nesterov=True))
train_pipeline = [
    dict(type='PreNormalize3D'),
    dict(type='GenSkeFeat', dataset='nturgb+d', feats=['j']),
    dict(type='UniformSample', clip_len=100),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint']) 
]
val_pipeline = [
    dict(type='PreNormalize3D'),
    dict(type='GenSkeFeat', dataset='nturgb+d', feats=['j']),
    dict(type='UniformSample', clip_len=100, num_clips=1, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
test_pipeline = [
    dict(type='PreNormalize3D'),
    dict(type='GenSkeFeat', dataset='nturgb+d', feats=['j']),
    dict(type='UniformSample', clip_len=100, num_clips=1, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]

data = dict(
    train=dict(
        dataset=dict(type=dataset_type, ann_file=ann_file, pipeline=train_pipeline, split='xsub_train')),
    val=dict(type=dataset_type, ann_file=ann_file, pipeline=val_pipeline, split='xsub_val'),
    test=dict(type=dataset_type, ann_file=ann_file, pipeline=test_pipeline, split='xsub_val'))
total_epochs = 80
# load_from = './work_dirs_pretrain/stgcn_read/stgcn_pyskl_ntu60_xsub_3dkp/j/epoch_20.pth'
work_dir = './work_dirs_pretrain/stgcn_read/stgcn_pyskl_ntu60_xsub_3dkp/j'
