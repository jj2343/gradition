import os
from yolox.data.datasets import COCODataset

class COCOVehicleDataset(COCODataset):
    def __init__(
        self,
        data_dir=None,
        json_file="instances_train2017.json",
        name="train2017",
        img_size=(416, 416),
        preproc=None,
        cache=False,
    ):
        super().__init__(data_dir, json_file, name, img_size, preproc, cache)
        
        # 只保留车辆类别: car(2), bus(5), truck(7)
        self.valid_ids = [2, 5, 7]
        
        # 过滤annotations
        self.coco = self._filter_annotations(self.coco)
    
    def _filter_annotations(self, coco):
        # 创建一个新的COCO对象，只包含我们需要的类别
        new_coco = {
            "images": coco.dataset["images"],
            "categories": [cat for cat in coco.dataset["categories"] if cat["id"] in self.valid_ids],
            "annotations": [ann for ann in coco.dataset["annotations"] if ann["category_id"] in self.valid_ids]
        }
        return new_coco