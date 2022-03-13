_base_="base.py"


classes = ("text", "title", "list", "table", "figure")
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=2,
    train=dict(

        sup=dict(

            type="CocoDataset",
            classes=classes,
            ann_file="data/coco/annotations/semi_supervised/instances_train2017.${fold}@${percent}.json",
            img_prefix="data/coco/train2017/" ),
        unsup=dict(

           type="CocoDataset",
            classes=classes,
            ann_file="data/coco/annotations/semi_supervised/instances_train2017.${fold}@${percent}-unlabeled.json",
            img_prefix="data/coco/train2017/",
        ),
    ),
    sampler=dict(
        train=dict(
            sample_ratio=[1, 1],
        )
    ),
)


work_dir = "work_dirs/${cfg_name}/${percent}/${fold}"
semi_wrapper = dict(
    train_cfg=dict(
        unsup_weight=2.0,
    )
)

lr_config = dict(step=[120000 * 4, 160000 * 4])
runner = dict(_delete_=True, type="IterBasedRunner", max_iters=150000)

