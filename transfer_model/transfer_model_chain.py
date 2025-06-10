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

from typing import Optional, Dict, Callable
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch3d_chamfer_distance import chamfer_distance

from tqdm import tqdm

from loguru import logger
from .utils import get_vertices_per_edge

from .optimizers import build_optimizer, minimize
from .utils import Tensor
from .losses import build_loss


def summary_closure(gt_vertices, var_dict, body_model, mask_ids=None):
    param_dict = {}
    for key, var in var_dict.items():
        if key in ['betas', 'leye_pose', 'reye_pose', 'jaw_pose', 'expression'] and var.shape[0] == 1 and var_dict['transl'].shape[0] > 1:
            batch_size = var_dict['transl'].shape[0]
            param_dict[key] = var.repeat(batch_size, 1)
        else:
            param_dict[key] = var
    body_model_output = body_model(return_full_pose=True, get_skin=True, **param_dict)
    est_vertices = body_model_output.vertices
    if mask_ids is not None:
        est_vertices = est_vertices[:, mask_ids]
        gt_vertices = gt_vertices[:, mask_ids]
    v2v = (est_vertices - gt_vertices).pow(2).sum(dim=-1).sqrt().mean()
    return {'Vertex-to-Vertex': v2v * 1000}


def build_model_forward_closure(
    body_model: nn.Module,
    var_dict: Dict[str, Tensor],
) -> Callable:
    def model_forward():
        param_dict = {}
        for key, var in var_dict.items():
            if key in ['betas', 'leye_pose', 'reye_pose', 'jaw_pose', 'expression'] and var.shape[0] == 1 and var_dict['transl'].shape[0] > 1:
                batch_size = var_dict['transl'].shape[0]
                param_dict[key] = var.repeat(batch_size, 1)
            else:
                param_dict[key] = var
        return body_model(return_full_pose=True, get_skin=True, **param_dict)
    return model_forward


def build_edge_closure(
    body_model: nn.Module,
    var_dict: Dict[str, Tensor],
    edge_loss: nn.Module,
    optimizer_dict,
    gt_vertices: Tensor,
) -> Callable:
    ''' Builds the closure for the edge objective
    '''
    optimizer = optimizer_dict['optimizer']
    create_graph = optimizer_dict['create_graph']

    params_to_opt = [p for key, p in var_dict.items() if 'pose' in key]
    model_forward = build_model_forward_closure(body_model, var_dict,)

    def closure(backward=True):
        if backward:
            optimizer.zero_grad()

        body_model_output = model_forward()
        est_vertices = body_model_output.vertices
        loss = edge_loss(est_vertices, gt_vertices)
        if backward:
            if create_graph:
                # Use this instead of .backward to avoid GPU memory leaks
                grads = torch.autograd.grad(
                    loss, params_to_opt, create_graph=True)
                torch.autograd.backward(
                    params_to_opt, grads, create_graph=True)
            else:
                loss.backward()

        return loss
    return closure


def build_vertex_closure(
    body_model: nn.Module,
    var_dict: Dict[str, Tensor],
    optimizer_dict,
    gt_vertices: Tensor,
    vertex_loss: nn.Module,
    mask_ids=None,
    params_to_opt: Optional[Tensor] = None,
) -> Callable:
    ''' Builds the closure for the vertex objective
    '''
    optimizer = optimizer_dict['optimizer']
    create_graph = optimizer_dict['create_graph']

    model_forward = build_model_forward_closure(
        body_model, var_dict,
    )

    if params_to_opt is None:
        params_to_opt = [p for key, p in var_dict.items() if p.requires_grad]

    def closure(backward=True):
        if backward:
            optimizer.zero_grad()

        body_model_output = model_forward()
        est_vertices = body_model_output.vertices  # est_vertices = body_model_output['vertices']

        device = next(body_model.buffers()).device
        loss = torch.tensor(0.0).to(device)

        v2v_weight = 1.0
        loss = vertex_loss(
            est_vertices[:, mask_ids] if mask_ids is not None else est_vertices,
            gt_vertices[:, mask_ids] if mask_ids is not None else gt_vertices)
        # print('v2v:', loss.item())
        loss += loss * v2v_weight

        shape_weight = 0.1
        loss_shape_prior = torch.mean(var_dict['betas'] ** 2)
        # print('shape prior:', loss_shape_prior.item())
        loss += loss_shape_prior * shape_weight

        hand_pose_weight = 1.0
        loss_hand_prior = (torch.mean(var_dict['left_hand_pose'] ** 2) + torch.mean(var_dict['right_hand_pose'] ** 2)) / 2
        # print('hand pose prior:', loss_hand_prior.item())
        loss += loss_hand_prior * hand_pose_weight

        if backward:
            if create_graph:
                # Use this instead of .backward to avoid GPU memory leaks
                grads = torch.autograd.grad(
                    loss, params_to_opt, create_graph=True)
                torch.autograd.backward(
                    params_to_opt, grads, create_graph=True)
            else:
                loss.backward()

        return loss
    return closure


def build_vertex_s2m_closure(
    body_model: nn.Module,
    var_dict: Dict[str, Tensor],
    optimizer_dict,
    gt_vertices: Tensor,
    vertex_loss: nn.Module,
    mask_ids=None,
    mask_ids_body=None,
    mask_ids_face=None,
    params_to_opt: Optional[Tensor] = None,
) -> Callable:
    ''' Builds the closure for the vertex objective + scan2mesh loss
    '''
    optimizer = optimizer_dict['optimizer']
    create_graph = optimizer_dict['create_graph']

    model_forward = build_model_forward_closure(
        body_model, var_dict,
    )

    if params_to_opt is None:
        params_to_opt = [p for key, p in var_dict.items() if p.requires_grad]

    def closure(backward=True):
        if backward:
            optimizer.zero_grad()

        body_model_output = model_forward()
        est_vertices = body_model_output.vertices  # est_vertices = body_model_output['vertices']

        device = next(body_model.buffers()).device
        loss = torch.tensor(0.0).to(device)

        weight_v2v_body = 1.0
        loss_v2v_body = vertex_loss(est_vertices[:, mask_ids_body], gt_vertices[:, mask_ids_body])
        # print('v2v body:', loss_v2v_body.item())
        loss += loss_v2v_body * weight_v2v_body

        weight_v2v_face = 1.0  # 200.0
        loss_v2v_face = vertex_loss(est_vertices[:, mask_ids_face], gt_vertices[:, mask_ids_face])
        # print('v2v face:', loss_v2v_face.item())
        loss += loss_v2v_face * weight_v2v_face

        shape_weight = 0.1  # 0.1 -->  shape looks normal, 0.01 --> slight under-smoothed shape
        loss_shape_prior = torch.mean(var_dict['betas'] ** 2)
        # print('shape prior:', loss_shape_prior.item())
        loss += loss_shape_prior * shape_weight

        hand_pose_weight = 1.0
        loss_hand_prior = (torch.mean(var_dict['left_hand_pose'] ** 2) + torch.mean(var_dict['right_hand_pose'] ** 2)) / 2
        # print('hand pose prior:', loss_hand_prior.item())
        loss += loss_hand_prior * hand_pose_weight

        expr_weight = 0.1
        loss_expr_prior = torch.mean(var_dict['expression'] ** 2)
        # print('expression prior:', loss_expr_prior.item())
        loss += loss_expr_prior * expr_weight

        s2m_weight = 1e5
        loss_s2m, _, _ = chamfer_distance(
            gt_vertices[:, mask_ids_face].contiguous(),
            est_vertices[:, mask_ids_face].contiguous())
        loss_s2m = loss_s2m.mean()
        # print('s2m:', loss_s2m.item())
        loss += loss_s2m * s2m_weight

        if backward:
            if create_graph:
                # Use this instead of .backward to avoid GPU memory leaks
                grads = torch.autograd.grad(
                    loss, params_to_opt, create_graph=True)
                torch.autograd.backward(
                    params_to_opt, grads, create_graph=True)
            else:
                loss.backward()

        return loss
    return closure



def get_variables(
    batch_size: int,
    body_model: nn.Module,
shape_fitting: bool = False,
    dtype: torch.dtype = torch.float32
) -> Dict[str, Tensor]:
    var_dict = {}
    device = next(body_model.buffers()).device

    # main body parameters
    var_dict.update({
        'transl': torch.zeros([batch_size, 3], device=device, dtype=dtype),
        'global_orient': torch.zeros([batch_size, 3], device=device, dtype=dtype),
        'body_pose': torch.zeros([batch_size, body_model.NUM_BODY_JOINTS * 3], device=device, dtype=dtype),
        'betas': torch.zeros([1 if shape_fitting else batch_size, body_model.num_betas], dtype=dtype, device=device),
    })
    # hand pose parameters
    var_dict.update(
        left_hand_pose=torch.zeros([batch_size, body_model.NUM_HAND_JOINTS * 3], device=device, dtype=dtype),
        right_hand_pose=torch.zeros([batch_size, body_model.NUM_HAND_JOINTS * 3], device=device, dtype=dtype),
    )
    # face parameters
    var_dict.update(
        jaw_pose=torch.zeros([1 if shape_fitting else batch_size, 3], device=device, dtype=dtype),
        leye_pose=torch.zeros([1 if shape_fitting else batch_size, 3], device=device, dtype=dtype),
        reye_pose=torch.zeros([1 if shape_fitting else batch_size, 3], device=device, dtype=dtype),
        expression=torch.zeros([1 if shape_fitting else batch_size, body_model.num_expression_coeffs], device=device, dtype=dtype),
    )

    for key, val in var_dict.items():
        val.requires_grad_(True)

    return var_dict


def run_fitting_chain(
    exp_cfg,
    batch: Dict[str, Tensor],
    body_model: nn.Module,
    # def_matrix: Tensor,
    # is_first_batch: bool,
    mask_ids: Optional = None,
    mask_ids_body: Optional = None,
    mask_ids_face: Optional = None,
    shape_fitting: bool = False,
    load_shape: bool = False,
    shape_dict_loaded: Optional[Tensor] = None,
    param_dict_loaded: Optional[Dict] = None,
    is_first_batch: Optional[bool] = True,
) -> Dict[str, Tensor]:
    ''' Runs fitting
    '''
    vertices = batch['vertices']
    batch_size = len(vertices)
    dtype, device = vertices.dtype, vertices.device
    summary_steps = exp_cfg.get('summary_steps')
    interactive = exp_cfg.get('interactive')

    ############# initialize smplx parameters
    var_dict = get_variables(batch_size, body_model, shape_fitting)

    ############# load pre-computed shape and face parameters
    if load_shape:
        for key in var_dict.keys():
            if key in ['betas', 'expression', 'leye_pose', 'reye_pose', 'jaw_pose']:
                var = torch.from_numpy((shape_dict_loaded[key])).float().unsqueeze(0).to(device=device)
                var_dict[key] = var.repeat(batch_size, 1)
                var_dict[key].requires_grad_(False)

    ############# load parameters from the previous frame/batch for faster convergence
    if param_dict_loaded is not None:
        for key in var_dict.keys():
            var_dict[key] = torch.from_numpy((param_dict_loaded[key])).float().to(device=device)
            var_dict[key].requires_grad_(True)
            if key in ['betas', 'expression', 'leye_pose', 'reye_pose', 'jaw_pose']:
                var_dict[key].requires_grad_(False)

    ############# Build the optimizer object for the current batch
    optim_cfg = exp_cfg.get('optim', {})
    def_vertices = vertices

    if mask_ids is None:
        f_sel = np.ones_like(body_model.faces[:, 0], dtype=np.bool_)
    else:
        f_per_v = [[] for _ in range(body_model.get_num_verts())]
        [f_per_v[vv].append(iff) for iff, ff in enumerate(body_model.faces) for vv in ff]
        f_sel = list(set(tuple(sum([f_per_v[vv] for vv in mask_ids], []))))
    vpe = get_vertices_per_edge(body_model.v_template.detach().cpu().numpy(), body_model.faces[f_sel])

    def log_closure():
        return summary_closure(def_vertices, var_dict, body_model,
                               mask_ids=mask_ids)

    ############## load edge and v2v loss configs
    edge_fitting_cfg = exp_cfg.get('edge_fitting', {})
    edge_loss = build_loss(type='vertex-edge', gt_edges=vpe, est_edges=vpe,
                           **edge_fitting_cfg)
    edge_loss = edge_loss.to(device=device)

    vertex_fitting_cfg = exp_cfg.get('vertex_fitting', {})
    vertex_loss = build_loss(**vertex_fitting_cfg)
    vertex_loss = vertex_loss.to(device=device)

    ####################### optimize with edge loss
    if is_first_batch:
        logger.info(f'[Stage 1]: fitting with edge loss...')
        optimizer_dict = build_optimizer(list(var_dict.values()), optim_cfg)
        closure = build_edge_closure(
            body_model, var_dict, edge_loss, optimizer_dict,
            def_vertices,
        )
        minimize(optimizer_dict['optimizer'], closure,
                 params=var_dict.values(),
                 summary_closure=log_closure,
                 summary_steps=summary_steps,
                 interactive=interactive,
                 **optim_cfg)

    ######################## Optimize all model parameters with V2V loss + shape/pose priors
    if is_first_batch:
        logger.info(f'[Stage 2]: fitting with V2V loss and shape/pose priors...')
        optimizer_dict = build_optimizer(list(var_dict.values()), optim_cfg)
        closure = build_vertex_closure(
            body_model, var_dict, optimizer_dict, def_vertices,
            vertex_loss=vertex_loss, mask_ids=mask_ids,
        )
        minimize(optimizer_dict['optimizer'], closure,
                 params=list(var_dict.values()),
                 summary_closure=log_closure,
                 summary_steps=summary_steps,
                 interactive=interactive,
                 **optim_cfg)

    #######################  optimize for face with V2V and scan2mesh loss (to fit shape)
    if shape_fitting:
        adam_lr_list = [0.01, 0.001]
        ftol_list = [1e-6, 1e-7]
        for stage in range(len(adam_lr_list)):
            logger.info(f'[Stage {stage+3}]: fitting with V2V and face scan2mesh loss (Adam lr={adam_lr_list[stage]})...')
            optimizer = optim.Adam([var_dict['betas'], var_dict['expression'], var_dict['jaw_pose'], var_dict['leye_pose'], var_dict['reye_pose']], lr=adam_lr_list[stage])
            optimizer_dict = {'optimizer': optimizer, 'create_graph': False}
            closure = build_vertex_s2m_closure(
                body_model, var_dict, optimizer_dict, def_vertices,
                vertex_loss=vertex_loss, mask_ids=mask_ids, mask_ids_body=mask_ids_body, mask_ids_face=mask_ids_face,
            )
            minimize(optimizer_dict['optimizer'], closure,
                     params=list(var_dict.values()),
                     summary_closure=log_closure,
                     summary_steps=summary_steps,
                     interactive=interactive,
                     maxiters=1000,
                     ftol=ftol_list[stage],
                     gtol=optim_cfg['gtol'],
                     )

    ######################## optimize with ADAM for faster convergence
    if not shape_fitting:
        # logger.info(f'[Stage 3]: fitting with V2V and pose priors (Adam)')
        adam_lr_list = [0.01, 0.001, 0.0001]
        ftol_list = [1e-6, 1e-7, 1e-7]
        for stage in range(len(adam_lr_list)):
            logger.info(f'[Stage {stage + 3}]: fitting with V2V and pose priors (Adam lr={adam_lr_list[stage]})...')
            optimizer = optim.Adam(list(var_dict.values()), lr=adam_lr_list[stage])
            optimizer_dict = {'optimizer': optimizer, 'create_graph': False}
            closure = build_vertex_closure(
                body_model, var_dict, optimizer_dict, def_vertices,
                vertex_loss=vertex_loss, mask_ids=mask_ids,
            )
            minimize(optimizer=optimizer_dict['optimizer'], closure=closure,
                          params=list(var_dict.values()),
                          summary_closure=log_closure,
                          summary_steps=summary_steps,
                          interactive=interactive,
                          maxiters=4000,
                          ftol=ftol_list[stage],
                          gtol=optim_cfg['gtol'],
                          )

    # ########## visualize
    # import open3d as o3d
    # input_mesh_o3d = o3d.geometry.TriangleMesh()
    # input_mesh_o3d.vertices = o3d.utility.Vector3dVector(def_vertices[frameid].detach().cpu().numpy())
    # input_mesh_o3d.triangles = o3d.utility.Vector3iVector(body_model.faces)
    # input_mesh_o3d.compute_vertex_normals()
    # o3d.visualization.draw_geometries([input_mesh_o3d])
    #
    # param_dict = {}
    # for key, var in var_dict.items():
    #     if key in ['betas', 'leye_pose', 'reye_pose', 'jaw_pose', 'expression'] and var.shape[0] == 1 and var_dict['transl'].shape[0] > 1:
    #         batch_size = var_dict['transl'].shape[0]
    #         param_dict[key] = var.repeat(batch_size, 1)
    #     else:
    #         param_dict[key] = var
    # body_model_output = body_model(return_full_pose=True, get_skin=True, **param_dict)
    # output_mesh_o3d = o3d.geometry.TriangleMesh()
    # output_mesh_o3d.vertices = o3d.utility.Vector3dVector(body_model_output.vertices[frameid].detach().cpu().numpy())
    # output_mesh_o3d.triangles = o3d.utility.Vector3iVector(body_model.faces)
    # output_mesh_o3d.compute_vertex_normals()
    # output_mesh_o3d.paint_uniform_color(np.array([160 / 255, 160 / 255, 1]))
    # o3d.visualization.draw_geometries([output_mesh_o3d])
    # o3d.visualization.draw_geometries([input_mesh_o3d, output_mesh_o3d])

    ############## save result
    param_dict = {}
    for key, var in var_dict.items():
        if key in ['betas', 'leye_pose', 'reye_pose', 'jaw_pose', 'expression'] and var.shape[0] == 1 and var_dict['transl'].shape[0] > 1:
            batch_size = var_dict['transl'].shape[0]
            param_dict[key] = var.repeat(batch_size, 1)
        else:
            param_dict[key] = var
    body_model_output = body_model(return_full_pose=True, get_skin=True, **param_dict)
    param_dict['vertices'] = body_model_output.vertices
    param_dict['faces'] = body_model.faces

    return param_dict
