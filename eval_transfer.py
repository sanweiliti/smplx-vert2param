import copy
import torch
import os
import trimesh
import glob
import pyrender
import time
import json
import smplx
import argparse
import numpy as np
import open3d as o3d
import pickle as pkl
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default='data')
parser.add_argument('--fitting_stage', type=str, default='pose', choices=['shape', 'pose'])
parser.add_argument('--sub_id', type=str, default='sub_0')
parser.add_argument('--seq_name', type=str, default='smplx_N_CON_charades')
parser.add_argument('--vis_frame', default='False', type=lambda x: x.lower() in ['true', '1'])
parser.add_argument('--vis_interval', default=1000, type=int)
parser.add_argument('--vis_seq', default='False', type=lambda x: x.lower() in ['true', '1'])
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    print('[INFO] fitting_stage: ', args.fitting_stage)
    print('[INFO] sub_id: ', args.sub_id)
    if args.fitting_stage == 'pose':
        print('[INFO] seq_name: ', args.seq_name)

    if args.fitting_stage == 'pose':
        intput_npy_path = os.path.join(args.data_root, 'input_npy', args.sub_id, args.seq_name, args.seq_name + '.npy')
        intput_npy = np.load(intput_npy_path)
        output_pkl_dir = os.path.join(args.data_root, 'fitting_results', args.sub_id, args.seq_name)
        # output_pkl_dir = '/mnt/ssd/trinity_smplx_seq/output_mesh_chain_bs100/smplx_N_CON_charades'
    elif args.fitting_stage == 'shape':
        intput_obj_dir = os.path.join(args.data_root, 'fitting_shape', args.sub_id, 'input_obj')
        output_pkl_dir = os.path.join(args.data_root, 'fitting_shape', args.sub_id, 'fitting_params')
    else:
        print('[ERROR] {} not in [pose, shape]!'.format(args.fitting_stage))
        exit()

    frame_list = glob.glob(os.path.join(output_pkl_dir, '*.pkl'))
    frame_list = sorted(frame_list)
    print('[INFO] get {} frames in {}'.format(len(frame_list), output_pkl_dir))

    ############### get body part seg mask for smplx
    with open('/local/home/szhang/motion_body_hand/data_loaders/smplx_vert_segmentation.json') as file:
        verts_idx_dict = json.load(file)
    smplx_hand_mask = list(verts_idx_dict['leftHand']) + list(verts_idx_dict['rightHand']) + list(verts_idx_dict['leftHandIndex1']) + list(verts_idx_dict['rightHandIndex1'])
    smplx_head_mask = list(verts_idx_dict['head'])
    smplx_body_mask = []
    for part_name in verts_idx_dict.keys():
        if part_name not in ['head', 'eyeballs', 'leftEye', 'rightEye', 'leftHand', 'rightHand', 'leftHandIndex1', 'rightHandIndex1']:
            smplx_body_mask += list(verts_idx_dict[part_name])

    ############### load smplx model
    MODEL_PATH = 'smplx_data/smplx_models_lockedhead'
    smplx_model = smplx.create(
            MODEL_PATH,
            model_type='smplx',
            gender='neutral',
            flat_hand_mean=True,
            num_betas=300,
            num_expression_coeffs=100,
            use_pca=False,
            batch_size=1
        ).to(device)



    ############### visualization setup: pyrender
    if args.vis_seq:
        axis_node = pyrender.Node(mesh=pyrender.Mesh.from_trimesh(trimesh.creation.axis(), smooth=False), name='axis')
        scene = pyrender.Scene()
        viewer = pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread=True, record=False)
        viewer.render_lock.acquire()
        scene.add_node(axis_node)
        viewer.render_lock.release()
        time.sleep(5)  # to adjust the camera view

    ############### metrics
    v2v_error_hand_list = []
    v2v_error_body_list = []
    v2v_error_head_list = []
    source_vert_list, result_vert_list = [], []
    # source_vert_list = np.zeros([len(frame_list), 10475, 3], dtype=np.float32)
    # result_vert_list = np.zeros([len(frame_list), 10475, 3], dtype=np.float32)

    ############### evaluation / visualization
    print('[INFO] evaluating...')
    for t, frame_name in tqdm(enumerate(frame_list[6000:])):
        frame_name = frame_name.split('/')[-1].split('.')[0]

        ############## read data and get vertices
        if args.fitting_stage == 'pose':
            frame_id = int(frame_name)
            source_verts = intput_npy[frame_id]
            source_mesh = trimesh.Trimesh(vertices=source_verts, faces=smplx_model.faces)
        elif args.fitting_stage == 'shape':
            source_mesh_path = os.path.join(intput_obj_dir, frame_name + '.obj')
            source_mesh = trimesh.load(source_mesh_path, process=False)
            source_verts = np.asarray(source_mesh.vertices)

        result_pkl_path = os.path.join(output_pkl_dir, frame_name + '.pkl')
        with open(result_pkl_path, 'rb') as f:
            load_params = pkl.load(f)
        for key in load_params:
            load_params[key] = torch.from_numpy(load_params[key]).float().unsqueeze(0).to(device)
        result_verts = smplx_model(**load_params).vertices[0].detach().cpu().numpy()
        result_mesh = trimesh.Trimesh(vertices=result_verts, faces=smplx_model.faces)

        # result_mesh = trimesh.load(result_mesh_path, process=False)
        # source_verts = np.asarray(source_mesh.vertices)
        # result_verts = np.asarray(result_mesh.vertices)

        if args.vis_frame:
            if t % args.vis_interval == 0:
                # mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
                source_mesh_o3d = o3d.geometry.TriangleMesh()
                source_mesh_o3d.vertices = o3d.utility.Vector3dVector(source_verts)
                source_mesh_o3d.triangles = o3d.utility.Vector3iVector(smplx_model.faces)
                source_mesh_o3d.compute_vertex_normals()
                o3d.visualization.draw_geometries([source_mesh_o3d])

                result_mesh_o3d = o3d.geometry.TriangleMesh()
                result_mesh_o3d.vertices = o3d.utility.Vector3dVector(result_verts)
                result_mesh_o3d.triangles = o3d.utility.Vector3iVector(smplx_model.faces)
                result_mesh_o3d.compute_vertex_normals()
                result_mesh_o3d.paint_uniform_color(np.array([160/255, 160/255, 1]))
                o3d.visualization.draw_geometries([result_mesh_o3d])
                o3d.visualization.draw_geometries([result_mesh_o3d, source_mesh_o3d])

        if args.vis_seq:
            # source_mesh.visual.vertex_colors = [0.8, 0.8, 0.8, 1.0]
            # source_mesh_copy = copy.deepcopy(source_mesh)
            # trans_mat = np.array([[1, 0, 0, 1.2],
            #                       [0, 1, 0, 0],
            #                       [0, 0, 1, 0],
            #                       [0, 0, 0, 1]])
            # source_mesh_copy.apply_transform(trans_mat)
            # source_mesh_pyrender = pyrender.Mesh.from_trimesh(source_mesh)
            # source_mesh_copy_pyrender = pyrender.Mesh.from_trimesh(source_mesh_copy)

            result_mesh.visual.vertex_colors = [66 / 255, 149 / 255, 245 / 255, 1.0]  # light blue
            # result_mesh_copy = copy.deepcopy(result_mesh)
            trans_mat = np.array([[1, 0, 0, -1.2],
                                  [0, 1, 0, 0],
                                  [0, 0, 1, 0],
                                  [0, 0, 0, 1]])
            # result_mesh_copy.apply_transform(trans_mat)
            result_mesh_pyrender = pyrender.Mesh.from_trimesh(result_mesh)
            # result_mesh_copy_pyrender = pyrender.Mesh.from_trimesh(result_mesh_copy)

            viewer.render_lock.acquire()
            if t > 0:
                # scene.remove_node(source_node)
                scene.remove_node(result_node)
                # scene.remove_node(source_copy_node)
                # scene.remove_node(result_copy_node)

            # source_node = pyrender.Node(mesh=source_mesh_pyrender, name='scan')
            # scene.add_node(source_node)
            result_node = pyrender.Node(mesh=result_mesh_pyrender, name='scan')
            scene.add_node(result_node)
            # source_copy_node = pyrender.Node(mesh=source_mesh_copy_pyrender, name='scan')
            # scene.add_node(source_copy_node)
            # result_copy_node = pyrender.Node(mesh=result_mesh_copy_pyrender, name='scan')
            # scene.add_node(result_copy_node)

            viewer.render_lock.release()
            # time.sleep(0.5)

        # ######## compute fitting loss
        source_vert_list.append(source_verts)
        result_vert_list.append(result_verts)
        v2v_error_body = np.linalg.norm(source_verts[smplx_body_mask] - result_verts[smplx_body_mask], axis=-1).mean()
        v2v_error_body_list.append(v2v_error_body)
        v2v_error_hand = np.linalg.norm(source_verts[smplx_hand_mask] - result_verts[smplx_hand_mask], axis=-1).mean()
        v2v_error_hand_list.append(v2v_error_hand)
        v2v_error_head = np.linalg.norm(source_verts[smplx_head_mask] - result_verts[smplx_head_mask], axis=-1).mean()
        v2v_error_head_list.append(v2v_error_head)


    v2v_error_body = np.asarray(v2v_error_body_list).mean()
    print('v2v_error body (mm): ', v2v_error_body * 1000)
    v2v_error_hand = np.asarray(v2v_error_hand_list).mean()
    print('v2v_error hand (mm): ', v2v_error_hand * 1000)
    v2v_error_head = np.asarray(v2v_error_head_list).mean()
    print('v2v_error head (mm): ', v2v_error_head * 1000)

    fps = 30
    if args.fitting_stage == 'pose':
        source_vert_list = np.asarray(source_vert_list)
        result_vert_list = np.asarray(result_vert_list)
        source_vel = (source_vert_list[1:] - source_vert_list[0:-1]) * fps
        result_vel = (result_vert_list[1:] - result_vert_list[0:-1]) * fps
        source_acc = (source_vel[1:] - source_vel[0:-1]) * fps
        result_acc = (result_vel[1:] - result_vel[0:-1]) * fps
        source_jitter = (source_acc[1:] - source_acc[0:-1]) * fps
        result_jitter = (result_acc[1:] - result_acc[0:-1]) * fps
        source_jitter = np.linalg.norm(source_jitter, axis=-1).mean()
        result_jitter = np.linalg.norm(result_jitter, axis=-1).mean()
        print('source_jitter (m/s^3):', source_jitter)
        print('result_jitter (m/s^3):', result_jitter)
