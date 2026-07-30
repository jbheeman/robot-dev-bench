from mmengine.config import Config
from mmpose.models.builder import build_pose_estimator
cfg = Config.fromfile('/home/andrew/miniconda3/lib/python3.13/site-packages/mmpose/.mim/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-huge_8xb64-210e_coco-256x192.py')
model = build_pose_estimator(cfg.model)
print("Model pos_embed shape:", model.backbone.pos_embed.shape)
