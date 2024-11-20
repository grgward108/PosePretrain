import torch
from WholeGraspPose.trainer import Trainer
from WholeGraspPose.models.models_handweights import MarkerNet

# step 1 load the example data
data = np.load('../dataset/contact_meshes/FullGraspPose/train/s2/cup_pass_1.npz', allow_pickle=True)

print(data.keys())

#step 2 load the markernet model 
# Path to the checkpoint
# model_path = '/home/edwarde/saga_cst/logs/GraspPose/cross_attention_hand_weights/snapshots/TR00_E050_net.pt'

# # Load the checkpoint
# checkpoint = torch.load(model_path)

# # Extract the hand attention bias parameter
# hand_attention_bias = checkpoint['marker_net.hand_attention_bias']

# # Print the hand attention bias values
# print("Hand Attention Bias:", hand_attention_bias)
