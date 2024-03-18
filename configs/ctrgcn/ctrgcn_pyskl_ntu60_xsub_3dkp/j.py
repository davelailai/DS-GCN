_base_ = ['../CTRGCN_sparse.py']
dataset_type = 'PoseDataset'
ann_file = 'data/nturgbd/ntu60_3danno.pkl'

work_dir = '/users/lailai/sharedscratch/openmmlab/pyskl_latest/work_dirs_sparse/ctrgcn_new/ctrgcn_pyskl_ntu60_xsub_3dkp/j_sparse0.2'
load_from = '.work_dirs_sparse/ctrgcn_new/ctrgcn_pyskl_ntu60_xsub_3dkp/j_full/latest.pth'
train_pipeline = [
    dict(type='PreNormalize3D'),
    dict(type='RandomRot', theta=0.2),
    dict(type='GenSkeFeat', dataset='nturgb+d', feats=['j']),
    dict(type='UniformSample', clip_len=60),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
val_pipeline = [
    dict(type='PreNormalize3D'),
    dict(type='GenSkeFeat', dataset='nturgb+d', feats=['j']),
    dict(type='UniformSample', clip_len=60, num_clips=1, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
test_pipeline = [
    dict(type='PreNormalize3D'),
    dict(type='GenSkeFeat', dataset='nturgb+d', feats=['j']),
    dict(type='UniformSample', clip_len=60, num_clips=10, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=2),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
data = dict(
    train=dict(
        dataset=dict(type=dataset_type, ann_file=ann_file, pipeline=train_pipeline, split='xsub_train')),
    val=dict(type=dataset_type, ann_file=ann_file, pipeline=val_pipeline, split='xsub_val'),
    test=dict(type=dataset_type, ann_file=ann_file, pipeline=test_pipeline, split='xsub_val'))

