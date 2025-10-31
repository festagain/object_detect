from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
""" Использовал для вычислений метрик"""
coco_gt = COCO("instances_scaled.json")
coco_dt = coco_gt.loadRes("results_val.json")

coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()
