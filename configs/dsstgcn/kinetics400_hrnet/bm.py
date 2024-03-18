_base_ = ['../DGSTGCN_model.py']
modality = 'bm'
clip_len = 100
graph = 'coco'
work_dir = './work_dirs_TMM9/dgstgcn8/kinectics_hrnet/bm8'
model = dict(
    # backbone=dict(
    #     graph_cfg=dict(layout=graph, mode='stgcn_spatial')),
    cls_head=dict(type='GCNHead', num_classes=400, in_channels=256))

memcached = True
mc_cfg = ('localhost', 11211)
dataset_type = 'PoseDataset'
ann_file = 'data/k400/k400_hrnet.pkl'
left_kp = [1, 3, 5, 7, 9, 11, 13, 15]
right_kp = [2, 4, 6, 8, 10, 12, 14, 16]
skeletons = [[0, 5], [0, 6], [5, 7], [7, 9], [6, 8], [8, 10], [5, 11],
             [11, 13], [13, 15], [6, 12], [12, 14], [14, 16], [0, 1], [0, 2],
             [1, 3], [2, 4], [11, 12]]
left_limb = [0, 2, 3, 6, 7, 8, 12, 14]
right_limb = [1, 4, 5, 9, 10, 11, 13, 15]
box_thr = 0.5
valid_ratio = 0.0

train_pipeline = [
    dict(type='DecompressPose', squeeze=True),
    dict(type='UniformSampleFrames', clip_len=clip_len),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    # dict(type='Resize', scale=(-1, 64)),
    # dict(type='RandomResizedCrop', area_range=(0.56, 1.0)),
    # dict(type='Resize', scale=(56, 56), keep_ratio=False),
    # dict(type='Flip', flip_ratio=0.5, left_kp=left_kp, right_kp=right_kp),
    dict(type='GenSkeFeat', dataset=graph, feats=[modality]),
    # dict(type='GeneratePoseTarget', with_kp=False, with_limb=True, skeletons=skeletons),
    dict(type='FormatGCNInput', num_person=2),
    # dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint']) 
]
val_pipeline = [
   dict(type='DecompressPose', squeeze=True),
    dict(type='UniformSampleFrames', clip_len=clip_len),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    # dict(type='Resize', scale=(-1, 64)),
    # dict(type='RandomResizedCrop', area_range=(0.56, 1.0)),
    # dict(type='Resize', scale=(56, 56), keep_ratio=False),
    # dict(type='Flip', flip_ratio=0.5, left_kp=left_kp, right_kp=right_kp),
    dict(type='GenSkeFeat', dataset=graph, feats=[modality]),
    # dict(type='GeneratePoseTarget', with_kp=False, with_limb=True, skeletons=skeletons),
    dict(type='FormatGCNInput', num_person=2),
    # dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint']) 
]
test_pipeline = [
    dict(type='DecompressPose', squeeze=True),
    dict(type='UniformSampleFrames', clip_len=clip_len, num_clips=10),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    # dict(type='Resize', scale=(-1, 64)),
    # dict(type='RandomResizedCrop', area_range=(0.56, 1.0)),
    # dict(type='Resize', scale=(56, 56), keep_ratio=False),
    # dict(type='Flip', flip_ratio=0.5, left_kp=left_kp, right_kp=right_kp),
    dict(type='GenSkeFeat', dataset=graph, feats=[modality]),
    # dict(type='GeneratePoseTarget', with_kp=False, with_limb=True, skeletons=skeletons),
    dict(type='FormatGCNInput', num_person=2),
    # dict(type='FormatShape', input_format='NCTHW'),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint']) 
]
data = dict(
    train=dict(
        dataset=dict(
            type=dataset_type,
            ann_file=ann_file,
            split='train',
            pipeline=train_pipeline,
            box_thr=box_thr,
            valid_ratio=valid_ratio,
            memcached=memcached,
            mc_cfg=mc_cfg)),
    val=dict(
        type=dataset_type,
        ann_file=ann_file,
        split='val',
        pipeline=val_pipeline,
        box_thr=box_thr,
        memcached=memcached,
        mc_cfg=mc_cfg),
    test=dict(
        type=dataset_type,
        ann_file=ann_file,
        split='val',
        pipeline=test_pipeline,
        box_thr=box_thr,
        memcached=memcached,
        mc_cfg=mc_cfg))

