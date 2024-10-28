"""
Quick wrapper for Segment Anything Model
"""

from dataclasses import dataclass, field
from typing import Type, Union, Literal

import torch
import numpy as np
from transformers import pipeline

from PIL import Image

from nerfstudio.configs import base_config as cfg
from sam2.build_sam import build_sam2_video_predictor

@dataclass
class ImgGroupModelConfig(cfg.InstantiateConfig):
    _target: Type = field(default_factory=lambda: SamGroupModel)
    """target class to instantiate"""
    model_type: Literal["sam_fb", "sam_hf", "maskformer", "sam_2"] = "sam_hf"
    """
    Currently supports:
     - "sam_fb": Original SAM model (from facebook github)
     - "sam_hf": SAM model from huggingface
     - "maskformer": MaskFormer model from huggingface (experimental)
     - "sam_2"
    """

    sam_model_type: str = ""
    sam_model_ckpt: str = ""
    sam_kwargs: dict = field(default_factory=lambda: {})
    "Arguments for SAM model (fb)."

    # # Settings used for the paper:
    # model_type="sam_fb",  
    # sam_model_type="vit_h",
    # sam_model_ckpt="models/sam_vit_h_4b8939.pth",
    # sam_kwargs={
    #     "points_per_side": 32,  # 32 in original
    #     "pred_iou_thresh": 0.90,
    #     "stability_score_thresh": 0.90,
    # },

    device: Union[torch.device, str] = ("cpu",)

@dataclass
class SamGroupModelConfig(cfg.InstantiateConfig):
    _target: Type = field(default_factory=lambda: SamGroupModel)
    """target class to instantiate"""
    model_type: Literal["sam_fb", "sam_hf", "maskformer", "sam_2"] = "sam_2"
    """
    Currently supports:
     - "sam_fb": Original SAM model (from facebook github)
     - "sam_hf": SAM model from huggingface
     - "maskformer": MaskFormer model from huggingface (experimental)
     - "sam_2"
    """

    sam_model_type: str = ""
    sam_model_ckpt: str = ""
    sam_kwargs: dict = field(default_factory=lambda: {})
    "Arguments for SAM model (fb)."

    # # Settings used for the paper:
    # model_type="sam_fb",  
    # sam_model_type="vit_h",
    # sam_model_ckpt="models/sam_vit_h_4b8939.pth",
    # sam_kwargs={
    #     "points_per_side": 32,  # 32 in original
    #     "pred_iou_thresh": 0.90,
    #     "stability_score_thresh": 0.90,
    # },

    device: Union[torch.device, str] = ("cpu",)

class ImgGroupModel:
    """
    Wrapper for 2D image segmentation models (e.g. MaskFormer, SAM)
    Original paper uses SAM, but we can use any model that outputs masks.
    The code currently assumes that every image has at least one group/mask.
    """
    def __init__(self, config: ImgGroupModelConfig, **kwargs):
        self.config = config
        self.kwargs = kwargs
        self.device = self.config.device = self.kwargs["device"]
        self.model = None
        self.video_segments = None

        # also, assert that model_type doesn't have a "/" in it! Will mess with h5df.
        assert "/" not in self.config.model_type, "model_type cannot have a '/' in it!"

    def __call__(self, img: np.ndarray):
        # takes in range 0-255... HxWx3
        # For using huggingface transformer's SAM model
        if self.config.model_type == "sam_hf":
            if self.model is None:
                self.model = pipeline("mask-generation", model="facebook/sam-vit-huge", device=self.device)
            img = Image.fromarray(img)
            masks = self.model(img, points_per_side=32, pred_iou_thresh=0.90, stability_score_thresh=0.90)
            masks = masks['masks']
            masks = sorted(masks, key=lambda x: x.sum())
            return masks
        
        elif self.config.model_type == "sam_fb":
            # For using the original SAM model
            if self.model is None:
                from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
                registry = sam_model_registry[self.config.sam_model_type]
                model = registry(checkpoint=self.config.sam_model_ckpt)
                model = model.to(device=self.config.device)
                self.model = SamAutomaticMaskGenerator(
                    model=model, **self.config.sam_kwargs
                )
            masks = self.model.generate(img)
            masks = [m['segmentation'] for m in masks] # already as bool
            masks = sorted(masks, key=lambda x: x.sum())

            return masks
        
        elif self.config.model_type == "maskformer":
            # For using another model (e.g., MaskFormer)
            if self.model is None:
                self.model = pipeline(model="facebook/maskformer-swin-large-coco", device=self.device)
            img = Image.fromarray(img)
            masks = self.model(img)
            masks = [
                (np.array(m['mask']) != 0)
                for m in masks
            ]
            masks = sorted(masks, key=lambda x: x.sum())
            return masks

        raise NotImplementedError(f"Model type {self.config.model_type} not implemented")
    
class SamGroupModel:
    def __init__(self, config: ImgGroupModelConfig, **kwargs):
        self.config = config
        self.kwargs = kwargs
        self.device = self.config.device = self.kwargs["device"]
        self.model = None
        self.video_segments = None

        # also, assert that model_type doesn't have a "/" in it! Will mess with h5df.
        assert "/" not in self.config.model_type, "model_type cannot have a '/' in it!"

    def __call__(self, video_dir):
        sam2_checkpoint = "/home/ehliang/nerfstudio/sam2/checkpoints/sam2.1_hiera_large.pt"
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

        predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=self.device)
        inference_state = predictor.init_state(video_path=video_dir)

        ann_frame_idx_1 = 0
        ann_obj_id_1 = 1
        points = np.array([[300, 500]], dtype=np.float32)
        labels = np.array([1], np.int32)
        # for labels, `1` means positive click and `0` means negative click

        # `add_new_points_or_box` returns masks for all objects added so far on this interacted frame
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx_1,
            obj_id=ann_obj_id_1,
            points=points,
            labels=labels,
        )

        ann_obj_id_2 = 2  # give a unique id to each object we interact with (it can be any integers)
        points = np.array([[600, 400]], dtype=np.float32)
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx_1,
            obj_id=ann_obj_id_2,
            points=points,
            labels=labels,
        )
        
        ann_obj_id_3 = 3
        points = np.array([[100, 100]], dtype=np.float32)
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=ann_frame_idx_1,
            obj_id=ann_obj_id_3,
            points=points,
            labels=labels,
        )


        video_segments = {}  # video_segments contains the per-frame segmentation results
        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
            background = torch.zeros(inference_state["video_height"], inference_state["video_width"], dtype=torch.bool).cpu().numpy()
            video_segments[out_frame_idx] = {}
            for i, out_obj_id in enumerate(out_obj_ids):
                video_segments[out_frame_idx][out_obj_id] = (out_mask_logits[i] > 0.0).cpu().numpy()

                background = background | video_segments[out_frame_idx][out_obj_id]
            
            video_segments[out_frame_idx][-1] = ~background



        return video_segments
    