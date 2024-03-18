_base_ = ['../STGCN_model.py']

model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='STGCN',
        graph_cfg=dict(layout='coco', mode='stgcn_spatial')))

dataset_type = 'PoseDataset'
ann_file = 'data/nturgbd/ntu60_hrnet.pkl'
train_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['j']),
    dict(type='UniformSample', clip_len=100),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint']) 
]
val_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['j']),
    dict(type='UniformSample', clip_len=100, num_clips=1, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
test_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=['j']),
    dict(type='UniformSample', clip_len=100, num_clips=1, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]

data = dict(
    train=dict(
        dataset=dict(type=dataset_type, ann_file=ann_file, pipeline=train_pipeline, split='xview_train')),
    val=dict(type=dataset_type, ann_file=ann_file, pipeline=val_pipeline, split='xview_val'),
    test=dict(type=dataset_type, ann_file=ann_file, pipeline=test_pipeline, split='xview_val'))
    
work_dir = './work_dirs/stgcn/stgcn_pyskl_ntu60_xview_hrnet/j'
