_base_ = ['../CTRGCN_sparse.py']
work_dir = './work_dirs_sparse/ctrgcn_new/ctrgcn_pyskl_ntu60_xview_3dkp/j_sparse0.99'
load_from = '.work_dirs_sparse/ctrgcn_new/ctrgcn_pyskl_ntu60_xview_3dkp/j_full/latest.pth'
dataset_type = 'PoseDataset'
ann_file = 'data/nturgbd/ntu60_3danno.pkl'
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
        dataset=dict(type=dataset_type, ann_file=ann_file, pipeline=train_pipeline, split='xview_train')),
    val=dict(type=dataset_type, ann_file=ann_file, pipeline=val_pipeline, split='xview_val'),
    test=dict(type=dataset_type, ann_file=ann_file, pipeline=test_pipeline, split='xview_val'))


# runtime settings

