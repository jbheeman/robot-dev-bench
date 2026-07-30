from mmengine.config import Config
from mmpose.apis import init_model
import torch

config_file = '/home/andrew/miniconda3/lib/python3.13/site-packages/mmpose/.mim/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-huge_8xb64-210e_coco-256x192.py'
checkpoint = 'https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-huge_8xb64-210e_coco-256x192-e32adcd4_20230314.pth'

cfg = Config.fromfile(config_file)
# Remove init_cfg to prevent downloading MAE pre-trained weights and causing the square pos_embed bug
if 'init_cfg' in cfg.model.backbone:
    del cfg.model.backbone.init_cfg

model = init_model(cfg, checkpoint, device='cpu')
print("Model built successfully!")
