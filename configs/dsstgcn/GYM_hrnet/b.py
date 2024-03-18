_base_ = ['../DGSTGCN_model.py']
modality = 'b'
clip_len = 100
graph = 'coco'
model = dict(
    # backbone=dict(
    #     graph_cfg=dict(layout=graph, mode='stgcn_spatial')),
    cls_head=dict(type='GCNHead', num_classes=99, in_channels=256))

work_dir = f'./work_dirs_TMM9/dgstgcn3/gym_hrnet/b3'

dataset_type = 'PoseDataset'
ann_file = 'data/gym/gym_hrnet.pkl'
train_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=[modality]),
    dict(type='UniformSample', clip_len=100),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
val_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=[modality]),
    dict(type='UniformSample', clip_len=100, num_clips=1, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
test_pipeline = [
    dict(type='PreNormalize2D'),
    dict(type='GenSkeFeat', dataset='coco', feats=[modality]),
    dict(type='UniformSample', clip_len=100, num_clips=10, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
data = dict(
    train=dict(
        dataset=dict(type=dataset_type, ann_file=ann_file, pipeline=train_pipeline, split='train')),
    val=dict(type=dataset_type, ann_file=ann_file, pipeline=val_pipeline, split='val'),
    test=dict(type=dataset_type, ann_file=ann_file, pipeline=test_pipeline, split='xval'))

