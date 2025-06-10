# Fitting SMPL-X parameters to SMPL-X mesh
Given a SMPL-X mesh sequence, this repo fits the SMPL-X parameters to this sequence. 
This codebase is modified based on the [official SMPLX transfer code](https://github.com/vchoutas/smplx/blob/main/transfer_model/README.md) for customized:

- Faster speed
- Batchwise implementation for initializing from fitted parameters from the previous frame
- Shape / pose / expression prior loss terms, since the input mesh might not strictly reside in the valid SMPL-X shape space (although they are in SMPL-X mesh topology) 



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
### Step 1: shape fitting (once for each subject)
Given multiple sequence of the same subject, we first extract the first frame of each sequence of this subject to fit body shape parameter `betas`. 
Since the facial expression does not change across frames, we also fit the face parameters `expression`, `leye_pose`, `reye_pose`, and `jaw_pose` at this stage.

Write the first frame of each sequence of this subject into `obj` mesh files for easier visualization (with MeshLab, it's recommended to pick easy poses):
```
python write_mesh_obj.py --data_root=data --sub_id=SUB_ID --seq_name_list=SEQ_NAME1,SEQ_NAME2,SEQ_NAME3,...
```
It will save the SMPLX mesh obj files into the folder `data/fitting_shape/SUB_ID/input_obj`.

Then fit the shape parameter and face parameters to these meshes:
```
python main_fitting.py --exp-cfg config_files/mesh2param_shape.yaml --exp-opts data_folder=data/fitting_shape/SUB_ID/input_obj output_folder=data/fitting_shape/SUB_ID/fitting_params
```

By the default setup, the fitting results (all SMPL-X parameters) for each frame here will be saved to `data/fitting_shape/SUB_ID/fitting_params`, and a pickle file `shape_face_fitting.pkl` containing the shape and face parameters of this subject will be saved to `data/fitting_shape/SUB_ID`.

**Note**: this script was tested with 4 frames as input. I set a cap of 100 frames in `write_mesh_obj.py`, as in principle, there is no need to use too many frames for shape fitting step.

### Step 2: pose fitting (once for each sequence)
Now for each sequence of this subject, we fix the shape and face parameters, and fit the global translation `transl`, global orientation `global_orient`, body pose `body_pose`, and hand poses `left_hand_pose`, `right_hand_pose` to the sequence.
To accelerate the fitting process, for each frame, we initialize the current frame parameters with the previous frame's fitting result. 

The batchwise is implemented within each sequence: given a sequence of length `T` and a batch size `B`, the sequence is evenly divided into `B` clips.
For the _i_-th batch, it contains the _i_-th frame in each clip.

Fit SMPL-X parameters to the sequence: 
```
python main_fitting.py --exp-cfg config_files/mesh2param_pose.yaml --exp-opts data_folder=data/input_npy/SUB_ID/SEQ_NAME output_folder=data/fitting_results/SUB_ID/SEQ_NAME shape_dict_path=data/fitting_shape/SUB_ID/shape_face_fitting.pkl
```
By the default setup, the fitting results will be saved to `data/fitting_results/SUB_ID/SEQ_NAME`, with each frame saved as a separate pkl file.

**Note**: this script was tested with `batch_size=100`. Performance with larger batch size is not guaranteed. The batch_size should not be larger than the sequence frame number.

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
Performance running on a single TITAN RTX GPU (24GB memory):

- **Step 1: shape fitting (using 4 frames, 1 frame pre sequence):**

| V2V Body (mm) | V2V Hand (mm) | V2V Head (mm) |
|:-------------:|:-------------:|:-------------:|
|    7.25       |    3.12       |    2.97       |



- **step 2: pose fitting (bs=100):**

| SEQ_NAME                                 | V2V Body (mm) | V2V Hand (mm) | V2V Head (mm) | Jitter_GT (m/s^3) | Jitter_fitting (m/s^3) | Seq_len (frame) | Fitting duration | Speed  |
|------------------------------------------|:-------------:|:-------------:|:-------------:|:-----------------:|:----------------------:|:---------------:|:----------------:|:------:|
| smplx_N_HANDS_touch_squeeze_fingers_palm |     7.39      |     3.07      |     3.05      |       12.91       |         14.64          |       683       |     2min55s      | 3.9fps |
| smplx_N_HAND_free_hand_no_occlusion      |     7.34      |     3.36      |     3.00      |       18.74       |         19.55          |       717       |     3min30s      | 3.4fps |
| smplx_N_HANDS_interlock_fingers          |     7.15      |     2.49      |     3.01      |       9.49        |         10.19          |       642       |     2min20s      | 4.6fps |
| smplx_N_CON_charades                     |     8.45      |     3.71      |     3.21      |       34.08       |         34.13          |      9195       |      37min       | 4.1fps | 


[//]: # (| SEQ_NAME                                 | V2V Body &#40;mm&#41; | V2V Hand &#40;mm&#41; | V2V Head &#40;mm&#41; | Jitter_GT &#40;m/s^3&#41; | Jitter_fitting &#40;m/s^3&#41; | Seq_len &#40;frame&#41; | Fitting duration | Speed  |)

[//]: # (|------------------------------------------|:-------------:|:-------------:|:-------------:|:-----------------:|:----------------------:|:---------------:|:----------------:|:------:|)

[//]: # (| smplx_N_HANDS_touch_squeeze_fingers_palm |     7.39      |     3.07      |     3.05      |       12.91       |         13.85          |       683       |  3.5 minutes     | 3.3fps |)

[//]: # (| smplx_N_HAND_free_hand_no_occlusion      |     7.33      |     3.35      |     3.00      |       18.74       |         19.10          |       717       |    5 minutes     | 2.4fps |)

[//]: # (| smplx_N_HANDS_interlock_fingers          |     7.15      |     2.49      |     3.01      |       9.49        |          9.70          |       642       |    3 minutes     | 3.5fps |)

[//]: # (| smplx_N_CON_charades                     |     8.44      |     3.71      |     3.20      |       34.08       |         34.06          |      9195       |    50 minutes    | 3.1fps | )
[//]: # ()
[//]: # (| SEQ_NAME                                 | V2V Body &#40;mm&#41; | V2V Hand &#40;mm&#41; | V2V Head &#40;mm&#41; | Jitter_GT &#40;m/s^3&#41; | Jitter_fitting &#40;m/s^3&#41; | Seq_len &#40;frame&#41; |    Fitting duration     |      Speed      |)

[//]: # (|------------------------------------------|:-------------:|:-------------:|:-------------:|:-----------------:|:----------------------:|:---------------:|:-----------------------:|:---------------:|)

[//]: # (| smplx_N_HANDS_touch_squeeze_fingers_palm |     7.39      |     3.07      |     3.05      |       12.91       |     13.88 / 14.64      |       683       |  3.6 minutes / 2min55s  | 3.2fps / 3.8fps |)

[//]: # (| smplx_N_HAND_free_hand_no_occlusion      |     7.34      |     3.36      |     3.00      |       18.74       |     18.80 / 19.55      |       717       |  4.5 minutes / 3min30s  | 2.7fps / 3.4fps |)

[//]: # (| smplx_N_HANDS_interlock_fingers          |     7.15      |     2.49      |     3.01      |       9.49        |      9.65 / 10.19      |       642       |   3 minutes / 2min20s   | 3.5fps / 4.5fps |)

[//]: # (| smplx_N_CON_charades                     |     8.45      |     3.71      |     3.21      |       34.08       |     34.06 / 34.13      |      9195       | 50 minutes / 37 minutes | 3.1fps / 4.1fps | )










## Acknowlegement
This codebase is adapted from the [SMPLX transfer code](https://github.com/vchoutas/smplx/blob/main/transfer_model/README.md).


## License
This codebase is licensed under [CC-BY-NC license](#license).


