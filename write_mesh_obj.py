import open3d as o3d
import numpy as np
import os
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()  # 4.6277
parser.add_argument('--data_root', type=str, default='data')
parser.add_argument('--sub_id', type=str, default='sub_0')
parser.add_argument('--seq_name_list', type=str,
                    default='smplx_N_CON_charades, smplx_N_HAND_free_hand_no_occlusion, smplx_N_HANDS_interlock_fingers, smplx_N_HANDS_touch_squeeze_fingers_palm')
args = parser.parse_args()

SEQ_NUM_CAP = 100
if __name__ == '__main__':
    print('[INDO] data_root: /mnt/ssd/trinity_smplx_seq/')
    print('[INDO] sub_id: sub_0')

    intput_npy_root = os.path.join(args.data_root, 'input_npy', args.sub_id)
    output_obj_root = os.path.join(args.data_root, 'fitting_shape', args.sub_id, 'input_obj')
    seq_name_list = args.seq_name_list.split(',')
    for i, item in enumerate(seq_name_list):
        seq_name_list[i] = item.strip()

    os.makedirs(output_obj_root) if not os.path.exists(output_obj_root) else None
    smplx_faces = np.load('smplx_data/smplx_faces.npy')

    print('[INDO] reading meshes from the first frame of sequences:', seq_name_list)
    if len(seq_name_list) > SEQ_NUM_CAP:
        print('[ERROR] {} sequences are sufficient! You are using more than {} sequences.'.format(SEQ_NUM_CAP, SEQ_NUM_CAP))
    source_npy_path_list = [os.path.join(intput_npy_root, seq_name, seq_name + '.npy') for seq_name in seq_name_list]
    verts_list = [np.load(source_npy_path)[0] for source_npy_path in source_npy_path_list]

    print('[INFO] saving mesh obj...')
    for i, verts in tqdm(enumerate(verts_list)):
        source_mesh_o3d = o3d.geometry.TriangleMesh()
        source_mesh_o3d.vertices = o3d.utility.Vector3dVector(verts)
        source_mesh_o3d.triangles = o3d.utility.Vector3iVector(smplx_faces)
        # source_smpl_mesh_o3d.compute_vertex_normals()
        # o3d.visualization.draw_geometries([source_mesh_o3d])
        o3d.io.write_triangle_mesh(os.path.join(output_obj_root, "{}_frame0.obj".format(seq_name_list[i])), source_mesh_o3d)
    print('[INDO] meshes saved to ', output_obj_root)

