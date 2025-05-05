from yolox.exp import Exp as MyExp

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = "yolox_s_coco_vehicle"
        
        # 定义数据集
        self.num_classes = 3  # car, truck, bus
        self.data_dir = "datasets/COCO"
        self.train_ann = "instances_train2017.json"
        self.val_ann = "instances_val2017.json"
        self.test_ann = "instances_val2017.json"
        
        self.max_epoch = 300
        self.data_num_workers = 4
        self.eval_interval = 1