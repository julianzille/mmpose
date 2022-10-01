_base_ = [
    '_base_/models/faster_rcnn_r50_fpn-acino.py',
    '_base_/datasets/coco_detection_acino.py',
    '_base_/schedules/schedule_2x.py', '_base_/default_runtime.py'
]
work_dir=''
log_file=''