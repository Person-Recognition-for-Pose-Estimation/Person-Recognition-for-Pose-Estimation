from ultralytics.data.converter import convert_coco

convert_coco(
    labels_dir="/home/ubuntu/coco/annotations",
    save_dir="/home/ubuntu/coco/labels",
    cls91to80=False,
)