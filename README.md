# Fitting SMPL-X parameters to SMPL-X mesh
Given a SMPL-X mesh sequence, this repo fits the SMPL-X parameters to this sequence. 
This codebase is modified based on the [official SMPLX transfer code](https://github.com/vchoutas/smplx/blob/main/transfer_model/README.md).


## License

## Downloading the model
To download the *SMPL-X* model go to [this project website](https://smpl-x.is.tue.mpg.de) and register to get access to the downloads section. We use the 'SMPL-X with the removed headbun' version. 
After downloading, put SMPL-X model into `smplx_data` folder, which follows the structure below:
```
smplx_data
├── smplx_models_lockedhead
│   ├── smplx
│       ├── SMPLX_FEMALE.npz
│       ├── SMPLX_MALE.npz
│       ├── SMPLX_NEUTRAL.npz
├── SMPL-X__FLAME_vertex_ids.npy
├── smplx_faces.npy
├── trinity2flame_face_ids.npy
```

## Requirements
Run the following command to install all necessary requirements:
```
pip install -r requirements.txt
```

## Data Preparation
Each input sequence is an npy file, which contains the SMPL-X vertex coordinates of the motion sequence.
Please organize your input SMPL-X vertex data as the following structure:
```
data
├── input_npy
│   ├── SUB_ID
│       ├── SEQ_NAME
│           ├── SEQ_NAME.npy
```
For example:
```
data
├── input_npy
│   ├── sub_0
│       ├── smplx_N_CON_charades
│           ├── smplx_N_CON_charades.npy
│       ├── smplx_N_HAND_free_hand_no_occlusion
│           ├── smplx_N_HAND_free_hand_no_occlusion.npy
│       ├── smplx_N_HANDS_interlock_fingers
│           ├── smplx_N_HANDS_interlock_fingers.npy
│       ├── smplx_N_HANDS_touch_squeeze_fingers_palm
│           ├── smplx_N_HANDS_touch_squeeze_fingers_palm.npy
```

## Logistics
### Step 1: shape fitting
Given multiple sequence of the same subject, we first extract the first frame of each sequence of this subject to fit body shape parameter `betas`. 
Since the facial expression does not change across frames, we also fit the face parameters `expression`, `leye_pose`, `reye_pose`, and `jaw_pose` at this stage.

Write the first frame of each sequence of this subject into `obj` mesh files for easier visualization (with MeshLab, it's recommended to pick easy poses):
```
python write_mesh_obj.py --data_root=data --sub_id=SUB_ID --seq_name_list=SEQ_NAME1,SEQ_NAME2,SEQ_NAME3,...
```
It will save the SMPLX mesh obj files into the folder `data/fitting_shape/SUB_ID/input_obj`.

Then fit the shape parameter and face parameters to these meshes: first set `data_folder` and `output_folder` in `onfig_files/mesh2param_shape.yaml`, and run the following script:
```
python main_fitting.py --exp-cfg config_files/mesh2param_shape.yaml
```

By the default setup, the fitting results (all SMPL-X parameters) for each frame here will be saved to `data/fitting_shape/SUB_ID/fitting_params`, and a pickle file `shape_face_fitting.pkl` containing the shape and face parameters of this subject will be saved to `data/fitting_shape/SUB_ID`.

**Note**: this script was tested with 4 frames as input. I set a cap of 100 frames in `write_mesh_obj.py`, as in principle, there is no need to use too many frames for shape fitting step.

### Step 2: pose fitting
Now for each sequence of this subject, we fix the shape and face parameters, and fit the global translation `transl`, global orientation `global_orient`, body pose `body_pose`, and hand poses `left_hand_pose`, `right_hand_pose` to the sequence.
To accelerate the fitting process, for each frame, we initialize the current frame parameters with the previous frame's fitting result. 

The batchwise is implemented within each sequence: given a sequence of length `T` and a batch size `B`, the sequence is evenly divided into `B` clips.
For the _i_-th batch, it contains the _i_-th frame in each clip.

Fit SMPL-X parameters to the sequence: first set `data_folder` and `output_folder` in `onfig_files/mesh2param_pose.yaml`, and run the following script:
```
python main_fitting.py --exp-cfg config_files/mesh2param_pose.yaml
```
By the default setup, the fitting results will be saved to `data/fitting_results/SUB_ID/SEQ_NAME`, with each frame saved as a separate pkl file.

**Note**: this script was tested with `batch_size=100`. Performance with larger batch size is not guaranteed. 

### Evaluation / Visualization of fitting results
Evaluating V2V errors without visualization: 
```
python eval_transfer.py --data_root=data --fitting_stage=STAGE --sub_id=SUB_ID --seq_name=SEQ_NAME --vis_frame=False --vis_seq=False
```
Set `STAGE` to either `shape` or `pose`, depending on the results of which step you want to visualize.

Visualizing with open3d (frame by frame, visualize one frame for each M frames): 
```
python eval_transfer.py --data_root=data --fitting_stage=STAGE --sub_id=SUB_ID --seq_name=SEQ_NAME --vis_frame=True --vis_interval=M --vis_seq=False
```

Visualizing with pyrender (motion animation of the entire sequence): 
```
python eval_transfer.py --data_root=data --fitting_stage=STAGE --sub_id=SUB_ID --seq_name=SEQ_NAME --vis_frame=False --vis_seq=True
```

## Fitting accuracy & speed
TODO
shape fitting:
v2v_error body (mm):  7.255158451196756
v2v_error hand (mm):  3.1269478359873744
v2v_error head (mm):  2.9712582463992407

smplx_N_HANDS_touch_squeeze_fingers_palm:
v2v_error body (mm):  7.395656313747168
v2v_error hand (mm):  3.074789186939597
v2v_error head (mm):  3.0567662324756384
source_jitter: 12.913863
result_jitter: 13.856683


## Acknowlegement
This codebase is adapted from the [SMPLX transfer code](https://github.com/vchoutas/smplx/blob/main/transfer_model/README.md) for faster speed.




