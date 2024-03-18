_base_ = ['../AAGCN_model.py']

dataset_type = 'PoseDataset'
ann_file = 'data/nturgbd/ntu60_3danno.pkl'
modality = 'j'
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

# load_from ='/home/jbridge/Jianyang/pyskl/work_dirs/stgcn/stgcn_pyskl_ntu120_xsub_3dkp/j/epoch_80.pth'
work_dir = './work_dirs/aagcn/aagcn_pyskl_ntu60_xview_3dkp/j_TT_mlp_add'
