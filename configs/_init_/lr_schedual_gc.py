data = dict(
    videos_per_gpu=128,
    workers_per_gpu=8,
    test_dataloader=dict(videos_per_gpu=32),
    train=dict(
    type='RepeatDataset',
    times=1)
    )

# optimizer
# optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0005, nesterov=True)
# optimizer_config = dict(grad_clip=None)
# optimizer = dict(
#             main=dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0005, nesterov=True, sparse='normal'),
#             mask=dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0005, nesterov=True, sparse='score_only'))
optimizer = dict(
            pool=dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0005, nesterov=True, Causal='pool'),
            SE=dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0005, nesterov=True, Causal='SE'))
# optimizer = dict(main=dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0005, nesterov=True, sparse='normal'))
# optimizer = dict(
#             module=dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0005, nesterov=True))
            # neck=dict(type='SGD', lr=0.05, momentum=0.9, weight_decay=0.0005, nesterov=True))
# optimizer_config = dict(type='MultiModuleHook', grad_clip=None)
# optimizer_config = dict(grad_clip=dict(max_norm=45, norm_type=2))
# learning policy
# lr_config = dict(policy='Cyclic', by_epoch=False)
lr_config = dict(policy='CosineAnnealing', min_lr=0, by_epoch=True)
# lr_config = dict(policy='OneCycle', max_lr=[0.5])
total_epochs = 200
checkpoint_config = dict(interval=5)
# evaluation = dict(interval=1, metrics=['top_k_accuracy','mean_class_accuracy'])
evaluation = dict(  # 训练期间做验证的设置
    interval=1,  # 执行验证的间隔
    metrics=['top_k_accuracy', 'mean_class_accuracy'])  # 设置 `top_k_accuracy` 作为指示器，用于存储最好的模型权重文件
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook'),dict(type='TensorboardLoggerHook')])


# runtime settings
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
