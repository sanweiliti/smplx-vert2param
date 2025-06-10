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

import os
import os.path as osp
import sys
import pickle

import numpy as np
import open3d as o3d
import torch
from loguru import logger
from tqdm import tqdm
import smplx
import torch.utils.data as dutils


from transfer_model.config import parse_args
from transfer_model.transfer_model_chain import run_fitting_chain
from transfer_model.data.datasets.dataloader_chain import MeshFileChain
from transfer_model.data.datasets.dataloader import MeshFolder
from transfer_model.utils import dist_utils

def main() -> None:
    exp_cfg = parse_args()

    ############### set GPU/CPU device
    if torch.cuda.is_available() and exp_cfg["use_cuda"]:
        dist_utils.setup_dist(exp_cfg.device_id)
        device = dist_utils.dev()
        logger.info(f'using cuda device {device}')
    else:
        device = torch.device('cpu')
        if exp_cfg["use_cuda"]:
            if input("use_cuda=True and GPU is not available, using CPU instead,"
                     " would you like to continue? (y/n)") != "y":
                sys.exit(3)

    ############### set logger
    logger.remove()
    logger.add(
        lambda x: tqdm.write(x, end=''), level=exp_cfg.logger_level.upper(),
        colorize=True)

    ############### set output path
    output_folder = osp.expanduser(osp.expandvars(exp_cfg.output_folder))
    logger.info(f'Saving output to: {output_folder}')
    os.makedirs(output_folder, exist_ok=True)

    ############### load smplx body model
    model_path = exp_cfg.body_model.folder
    body_model = smplx.create(model_path, flat_hand_mean=True, use_pca=False, **exp_cfg.body_model)
    logger.info(body_model)
    body_model = body_model.to(device=dist_utils.dev())

    ############### shape fitting config
    shape_fitting = exp_cfg.get('shape_fitting', True)
    load_shape = exp_cfg.get('load_shape', True)
    logger.info(f'shape_fitting: {shape_fitting}')
    logger.info(f'load_shape: {load_shape}')
    shape_dict_path = exp_cfg.shape_dict_path
    shape_dict_loaded = None
    if load_shape:
        if not os.path.exists(shape_dict_path):
            print('precomputed shape parameter file {} does not exist!'.format(shape_dict_path))
            exit()
        else:
            with open(shape_dict_path, 'rb') as f:
                shape_dict_loaded = pickle.load(f)
            logger.info(f'shape parameters loaded from: {shape_dict_path}')


    ############### set vertex mask to exclude from fitting loss
    smplx_flame_corr = np.load('smplx_data/SMPL-X__FLAME_vertex_ids.npy')  # [5023] smplx vert ids tha belong to flame
    trinity2flame_face_ids = np.load('smplx_data/trinity2flame_face_ids.npy')  # exclude scalp, eyeball, inner mouth vertices for fitting
    mask_ids_face = smplx_flame_corr[trinity2flame_face_ids]
    mask_ids_body = np.asarray(list(set(range(10475)) - set(smplx_flame_corr)))
    mask_ids = np.asarray(list(set(list(mask_ids_body) + list(mask_ids_face))))
    # to tensor
    mask_ids_face = torch.from_numpy(mask_ids_face).to(device=dist_utils.dev())
    mask_ids_body = torch.from_numpy(mask_ids_body).to(device=dist_utils.dev())
    mask_ids = torch.from_numpy(mask_ids).to(device=dist_utils.dev())

    ############### set dataloader
    logger.info(f'Creating dataloader with B={exp_cfg.batch_size}, workers={exp_cfg.num_workers}')
    if shape_fitting:
        dataset = MeshFolder(data_folder=exp_cfg.data_folder, exts='obj')
    else:
        dataset = MeshFileChain(data_folder=exp_cfg.data_folder, batch_size=exp_cfg.batch_size)
    dataloader = dutils.DataLoader(dataset,
                                   batch_size=exp_cfg.batch_size,
                                   num_workers=exp_cfg.num_workers,
                                   shuffle=False)

    ############### run fitting
    for ii, batch in enumerate(tqdm(dataloader)):
        for key in batch:
            if torch.is_tensor(batch[key]):
                batch[key] = batch[key].to(device=dist_utils.dev())
        batch_size = len(batch['vertices'])

        # shape fitting for one subject with multiple frames
        if shape_fitting:
            param_dict_loaded = None
            is_first_batch = True
        # load from previous fitted frames to accelerate fitting if shape_fitting is False
        else:
            if ii == 0:
                param_dict_loaded = None
                is_first_batch = True
            else:
                is_first_batch = False
                param_dict_loaded = {}
                for key in ['transl', 'global_orient', 'body_pose', 'betas', 'left_hand_pose', 'right_hand_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'expression']:
                    param_dict_loaded[key] = []
                for bs in range(batch_size):
                    param_dict_path = previous_output_path_list[bs]
                    with open(param_dict_path, 'rb') as f:
                        param_cur = pickle.load(f)
                        for key in param_cur.keys():
                            param_dict_loaded[key].append(param_cur[key])
                for key in param_dict_loaded.keys():
                    param_dict_loaded[key] = np.asarray(param_dict_loaded[key])

        ############## run fitting
        var_dict = run_fitting_chain(
            exp_cfg, batch, body_model, mask_ids, mask_ids_body, mask_ids_face, shape_fitting, load_shape, shape_dict_loaded, param_dict_loaded, is_first_batch)

        ############## save fitting results
        previous_output_path_list = []
        for j in range(batch_size):
            if shape_fitting:
                fname = batch['paths'][j].split('/')[-1].split('.')[0]
            else:
                fname = f"{batch['mesh_frame_id'][j]:06d}"

            # ####### save mesh
            # output_path = osp.join(
            #     output_folder, f'{fname}.obj')
            # mesh = np_mesh_to_o3d(
            #     var_dict['vertices'][j], var_dict['faces'])
            # o3d.io.write_triangle_mesh(output_path, mesh)

            ####### save params as pkl
            output_path = osp.join(
                output_folder, f'{fname}.pkl')
            save_params = {
                'transl': var_dict['transl'][j].detach().cpu().numpy(),
                'global_orient': var_dict['global_orient'][j].detach().cpu().numpy(),
                'body_pose': var_dict['body_pose'][j].detach().cpu().numpy(),
                'betas': var_dict['betas'][j].detach().cpu().numpy(),
                'left_hand_pose': var_dict['left_hand_pose'][j].detach().cpu().numpy(),
                'right_hand_pose': var_dict['right_hand_pose'][j].detach().cpu().numpy(),
                'jaw_pose': var_dict['jaw_pose'][j].detach().cpu().numpy(),
                'leye_pose': var_dict['leye_pose'][j].detach().cpu().numpy(),
                'reye_pose': var_dict['reye_pose'][j].detach().cpu().numpy(),
                'expression': var_dict['expression'][j].detach().cpu().numpy(),
            }
            with open(output_path, 'wb') as f:
                pickle.dump(save_params, f)
            previous_output_path_list.append(output_path)

        if shape_fitting:
            shape_face_params = {
                'betas': var_dict['betas'][0].detach().cpu().numpy(),
                'jaw_pose': var_dict['jaw_pose'][0].detach().cpu().numpy(),
                'leye_pose': var_dict['leye_pose'][0].detach().cpu().numpy(),
                'reye_pose': var_dict['reye_pose'][0].detach().cpu().numpy(),
                'expression': var_dict['expression'][0].detach().cpu().numpy(),
            }
            parent_folder = ('/').join(output_folder.split('/')[0:-1])
            with open(os.path.join(parent_folder, 'shape_face_fitting.pkl'), 'wb') as f:
                pickle.dump(shape_face_params, f)


if __name__ == '__main__':
    main()
