import torch
import numpy as np
import json
from smplx.lbs import batch_rodrigues
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import yaml
from matplotlib import cm
# Instantiate the model
from WholeGraspPose.models.models_handweights import FullBodyGraspNet
from utils.train_helper import point2point_signed

data = np.load('dataset/GraspPose/train/s1/airplane_fly_1.npz', allow_pickle=True)

# Extract relevant fields (use only the first frame)
markers = data['markers_143'][0:1]  # Shape: (1, 143, 3) for the first frame
transf_transl = data['transf_transl'][0:1]  # Shape: (1, 3)
verts_object = data['verts_object'][0:1]  # Shape: (1, 2048, 3)
normal_object = data['normal_object'][0:1]  # Shape: (1, 2048, 3)

# Compute the signed distance from markers to the closest point in verts_object
distances = []
for marker in markers:
    min_distance = float('inf')
for vert in data['verts_object'][0]:
    distance = point2point_signed(marker, vert)
    # add exponential function to the distances Iw(d) = exp (-w x d) where w is a weight and d is the distance. for now w = 5
    distance = np.exp(-5 * distance)
    if distance < min_distance:
        min_distance = distance
distances.append(min_distance)
data['hand_to_object_distance'] = torch.tensor(distances)

#visualize the heatmap of markers to object distance (low value means red, high value means blue)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
cmap = cm.get_cmap('coolwarm')
norm = plt.Normalize(vmin=0, vmax=1)
colors = cmap(norm(data['hand_to_object_distance']))
ax.scatter(markers[:, 0], markers[:, 1], markers[:, 2], c=colors)
plt.show()