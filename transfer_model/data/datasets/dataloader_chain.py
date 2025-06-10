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

from typing import Optional, Tuple

import sys
import os
import os.path as osp

import numpy as np
from psbody.mesh import Mesh
import trimesh
import glob

import torch
from torch.utils.data import Dataset
from loguru import logger


class MeshFileChain(Dataset):
    def __init__(
        self,
        data_folder: str,
        batch_size: int,
        # exts: Optional[Tuple] = None
    ) -> None:
        ''' Dataset similar to ImageFolder that reads meshes with the same
            topology
        '''
        # self.data_folder = osp.expandvars(data_folder)

        self.data_folder = data_folder
        logger.info(
            f'Building mesh folder dataset for folder: {self.data_folder}')

        ################ reorder indices in the sequence
        npy_file = glob.glob(os.path.join(data_folder, '*.npy'))
        if len(npy_file) > 1:
            print('more than one npy files in {}!'.format(data_folder))
            exit()
        elif len(npy_file) == 0:
            print('no npy files in {}!'.format(data_folder))
            exit()
        self.seq_verts = np.load(npy_file[0])  # [T, 10475, 3]
        self.seq_len = len(self.seq_verts)

        ######## create reordered sequence
        # clip_num = batch_size
        if self.seq_len > batch_size:
            clip_len = self.seq_len // batch_size
            remain_len = self.seq_len - clip_len * batch_size
            clip_len = clip_len if remain_len == 0 else clip_len + 1
            self.seq_len_pad = batch_size * clip_len
            self.seq_verts_reorder_pad = np.zeros([self.seq_len_pad, self.seq_verts.shape[-2], self.seq_verts.shape[-1]], dtype=np.float32)
            self.frame_id_list_reorder_pad = -np.ones(self.seq_len_pad).astype(np.int)
            for i in range(clip_len):
                for j in range(batch_size):
                    idx_old = j * clip_len + i
                    idx_new = i * batch_size + j
                    if idx_old < self.seq_len and idx_new < self.seq_len_pad:
                        # self.seq_verts_reorder_pad[idx_new] = self.seq_verts_pad[idx_old]
                        self.frame_id_list_reorder_pad[idx_new] = idx_old
            self.frame_id_list_reorder_pad[np.where(self.frame_id_list_reorder_pad==-1)[0]] = self.seq_len - 1
            self.seq_verts_reorder_pad = self.seq_verts[list(self.frame_id_list_reorder_pad)]
        else:
            self.frame_id_list_reorder_pad = np.arange(self.seq_len)
            self.seq_verts_reorder_pad = self.seq_verts

            # print(self.frame_id_list_reorder_pad)
        # for k in range(clip_len):
        #     print(self.frame_id_list_reorder_pad[batch_size * k: (k + 1) * batch_size])


    def __len__(self) -> int:
        return self.seq_len_pad

    def __getitem__(self, index):
        mesh_verts = self.seq_verts_reorder_pad[index]
        mesh_frame_id = self.frame_id_list_reorder_pad[index]
        return{
            'vertices': np.asarray(mesh_verts, dtype=np.float32),
            'indices': index,
            'mesh_frame_id': mesh_frame_id,
        }

        # mesh_path = self.data_paths[index]
        # # Load the mesh
        # mesh = trimesh.load(mesh_path, process=False)
        # return {
        #     'vertices': np.asarray(mesh.vertices, dtype=np.float32),
        #     'faces': np.asarray(mesh.faces, dtype=np.int32),
        #     'indices': index,
        #     'paths': mesh_path,
        # }
