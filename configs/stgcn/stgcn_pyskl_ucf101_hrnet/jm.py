_base_ = ['../STGCN_model.py']

model = dict(
    backbone=dict(
        graph_cfg=dict(layout='coco', mode='stgcn_spatial')),
    cls_head=dict(type='GCNHead', num_classes=101, in_channels=256))

dataset_type = 'PoseDataset'
ann_file = 'data/ucf101/ucf101_hrnet.pkl'
train_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['jm']),
    dict(type='UniformSample', clip_len=100),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint']) 
]
val_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['jm']),
    dict(type='UniformSample', clip_len=100, num_clips=1, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
test_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['jm']),
    dict(type='UniformSample', clip_len=100, num_clips=1, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]

data = dict(
    train=dict(
        dataset=dict(type=dataset_type, ann_file=ann_file, pipeline=train_pipeline, split='train1')),
    val=dict(type=dataset_type, ann_file=ann_file, pipeline=val_pipeline, split='test1'),
    test=dict(type=dataset_type, ann_file=ann_file, pipeline=test_pipeline, split='test1'))

work_dir = './work_dirs/stgcn/stgcn_pyskl_ucf101_hrnet/jm'
