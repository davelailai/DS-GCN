_base_ = ['../STGCN_sparse.py']

modality = 'j'
dataset_type = 'PoseDataset'
ann_file = 'data/nturgbd/ntu60_3danno.pkl'
work_dir = './work_dirs_sparse/stgcn_new/stgcn_pyskl_ntu60_xview_3dkp/j_sparse0.2'
load_from='./work_dirs_sparse/stgcn/stgcn_pyskl_ntu60_xview_3dkp/j_full/latest.pth'
train_pipeline = [
    dict(type='PreNormalize3D', align_spine=False),
    dict(type='RandomRot', theta=0.2),
    dict(type='GenSkeFeat', feats=[modality]),
    dict(type='UniformSample', clip_len=60),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput'),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
val_pipeline = [
    dict(type='PreNormalize3D', align_spine=False),
    dict(type='GenSkeFeat', feats=[modality]),
    dict(type='UniformSample', clip_len=60, num_clips=1),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput'),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
test_pipeline = [
    dict(type='PreNormalize3D', align_spine=False),
    dict(type='GenSkeFeat', feats=[modality]),
    dict(type='UniformSample', clip_len=60, num_clips=10),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput'),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]

data = dict(
    train=dict(
        dataset=dict(type=dataset_type, ann_file=ann_file, pipeline=train_pipeline, split='xview_train')),
    val=dict(type=dataset_type, ann_file=ann_file, pipeline=val_pipeline, split='xview_val'),
    test=dict(type=dataset_type, ann_file=ann_file, pipeline=test_pipeline, split='xview_val'))
    
