_base_ = ['../STGCN_model.py']

model = dict(
    type='RecognizerGCN',
    cls_head=dict(type='GCNHead', num_classes=120, in_channels=256))

dataset_type = 'PoseDataset'
ann_file = 'data/nturgbd/ntu120_3danno.pkl'
train_pipeline = [
    dict(type='PreNormalize3D'),
    dict(type='GenSkeFeat', dataset='nturgb+d', feats=['jm']),
    dict(type='UniformSample', clip_len=100),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
val_pipeline = [
    dict(type='PreNormalize3D'),
    dict(type='GenSkeFeat', dataset='nturgb+d', feats=['jm']),
    dict(type='UniformSample', clip_len=100, num_clips=1, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
test_pipeline = [
    dict(type='PreNormalize3D'),
    dict(type='GenSkeFeat', dataset='nturgb+d', feats=['jm']),
    dict(type='UniformSample', clip_len=100, num_clips=1, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
data = dict(
    train=dict(
        dataset=dict(type=dataset_type, ann_file=ann_file, pipeline=train_pipeline, split='xset_train')),
    val=dict(type=dataset_type, ann_file=ann_file, pipeline=val_pipeline, split='xset_val'),
    test=dict(type=dataset_type, ann_file=ann_file, pipeline=test_pipeline, split='xset_val'))


work_dir = './work_dirs/stgcn/stgcn_pyskl_ntu120_xset_3dkp/jm'
