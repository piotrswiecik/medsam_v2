from enum import Enum
from pydantic import BaseModel


class Mode(Enum):
    TRAIN = "train"
    PREDICT = "predict"

class Config(BaseModel):
    yolo_model_path: str
    arcade_syntax_base_dir: str
    yolo_cache_dir: str
    medsam_workdir: str
    medsam_base: str
    yolo_confidence_threshold: float
    image_size: int
    medsam_num_epochs: int
    medsam_val_freq: int
    medsam_checkpoint_freq: int
    medsam_batch_size: int
    medsam_num_workers: int