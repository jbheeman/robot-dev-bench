import os
import argparse
import logging
from mmengine.config import Config
from mmengine.runner import Runner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Fine-tune ViTPose on a custom generalized humanoid dataset.")
    parser.add_argument("--data-root", default="data/humanoid_dataset", help="Path to the dataset root")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size per GPU")
    parser.add_argument("--work-dir", default="work_dirs/vitpose_humanoid", help="Directory to save checkpoints")
    args = parser.parse_args()

    # Base config path for ViTPose-small (assuming mmpose is installed)
    try:
        import mmpose
    except ImportError:
        logger.error("mmpose is not installed. Please install it to run fine-tuning.")
        return

    mmpose_path = os.path.dirname(mmpose.__file__)
    # MMPose stores configs in .mim/configs/
    base_cfg_path = os.path.join(mmpose_path, ".mim", "configs", "body_2d_keypoint", "topdown_heatmap", "coco", "td-hm_ViTPose-small_8xb64-210e_coco-256x192.py")
    
    if not os.path.exists(base_cfg_path):
        # Fallback to local configs directory
        base_cfg_path = os.path.join(mmpose_path, "configs", "body_2d_keypoint", "topdown_heatmap", "coco", "td-hm_ViTPose-small_8xb64-210e_coco-256x192.py")

    logger.info(f"Loading base config from: {base_cfg_path}")
    cfg = Config.fromfile(base_cfg_path)

    # Overwrite dataset paths
    cfg.data_root = args.data_root
    
    # Train dataloader
    cfg.train_dataloader.batch_size = args.batch_size
    cfg.train_dataloader.dataset.data_root = args.data_root
    cfg.train_dataloader.dataset.ann_file = 'annotations/train.json'
    cfg.train_dataloader.dataset.data_prefix = dict(img='images/')
    
    # Remove validation and test since xtcocotools evaluation is broken
    for key in ['val_dataloader', 'val_evaluator', 'val_cfg', 'test_dataloader', 'test_evaluator', 'test_cfg']:
        if key in cfg:
            cfg[key] = None

    # Modify training epochs
    cfg.train_cfg.max_epochs = args.epochs
    cfg.param_scheduler[0].end = args.epochs
    
    # Set work directory
    cfg.work_dir = args.work_dir
    
    # Use pre-trained weights for fine-tuning
    cfg.load_from = "https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-small_8xb64-210e_coco-256x192-62d7a712_20230314.pth"

    logger.info("Initializing Runner...")
    runner = Runner.from_cfg(cfg)
    
    logger.info("Starting fine-tuning...")
    runner.train()
    
    logger.info(f"Training complete! Checkpoints saved in {args.work_dir}")

if __name__ == "__main__":
    main()
