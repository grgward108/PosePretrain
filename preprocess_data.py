from PoseBridge.data.end2end_dataloader import GRAB_DataLoader, save_preprocessed_data

grab_datasets = ['s7', 's8']
grab_dir = '../../../data/edwarde/dataset/grab/GraspMotion'
smplx_model_path = 'body_utils/body_models'
save_dir = "../../../data/edwarde/dataset/preprocessed_grab"

# Preload and save processed data
dataset = GRAB_DataLoader(clip_seconds=2, clip_fps=30, mode='local_markers_3dv', markers_type='f15_p22')
dataset.read_data(grab_datasets, grab_dir)
dataset.create_body_repr(with_hand=False, smplx_model_path=smplx_model_path)
save_preprocessed_data(dataset, save_dir)