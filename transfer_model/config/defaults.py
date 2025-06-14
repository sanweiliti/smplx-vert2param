# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2020 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: Vassilis Choutas, vassilis.choutas@tuebingen.mpg.de

# from typing import Tuple, Optional
# from copy import deepcopy
#  from yacs.config import CfgNode as CN
from dataclasses import dataclass
from omegaconf import OmegaConf

from .loss_defaults import conf as loss_cfg, LossConfig
# from .dataset_defaults import conf as dataset_cfg, DatasetConfig
from .optim_defaults import conf as optim_cfg, OptimConfig
from .body_model_defaults import conf as body_model_cfg, BodyModelConfig


@dataclass
class EdgeFitting:
    reduction: str = 'mean'


@dataclass
class VertexFitting:
    reduction: str = 'mean'
    type: str = 'l2'


@dataclass
class Config:
    use_cuda: bool = True
    device_id: int = 0
    log_file: str = '/tmp/logs'
    output_folder: str = 'output'
    summary_steps: int = 5
    float_type: str = 'float'
    logger_level: str = 'INFO'
    interactive: bool = True

    optim: OptimConfig = optim_cfg
    losses: LossConfig = loss_cfg
    body_model: BodyModelConfig = body_model_cfg

    # dataset config
    data_folder: str = ''
    num_workers: int = 0
    batch_size: int = 1

    shape_fitting: bool = False
    load_shape: bool = False
    shape_dict_path: str = ''

    edge_fitting: EdgeFitting = EdgeFitting()


conf = OmegaConf.structured(Config)
