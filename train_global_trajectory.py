import torch


def get_forward_joint(joint_start):
	""" Joint_start: [B, N, 3] in xyz """
	x_axis = joint_start[:, 2, :] - joint_start[:, 1, :]
	x_axis[:, -1] = 0
	x_axis = x_axis / torch.norm(x_axis, dim=-1).unsqueeze(1)
	z_axis = torch.tensor([0, 0, 1]).float().unsqueeze(0).repeat(len(x_axis), 1).to(device)
	y_axis = torch.cross(z_axis, x_axis)
	y_axis = y_axis / torch.norm(y_axis, dim=-1).unsqueeze(1)
	transf_rotmat = torch.stack([x_axis, y_axis, z_axis], dim=1)
	return y_axis, transf_rotmat


def prepare_traj_input(joint_start, joint_end):
	""" Joints: [B, N, 3] in xyz """
	B, N, _ = joint_start.shape
	T = 62
	joint_sr_input = torch.ones(B, 4, T)  # [B, xyr, T]
	y_axis, transf_rotmat = get_forward_joint(joint_start)
	joint_start_new = joint_start.clone()
	joint_end_new = joint_end.clone()  # to check whether original joints change or not
	joint_start_new = torch.matmul(joint_start - joint_start[:, 0:1], transf_rotmat)
	joint_end_new = torch.matmul(joint_end - joint_start[:, 0:1], transf_rotmat)

	# start_forward, _ = get_forward_joint(joint_start_new)
	start_forward = torch.tensor([0, 1, 0]).unsqueeze(0)
	end_forward, _ = get_forward_joint(joint_end_new)

	joint_sr_input[:, :2, 0] = joint_start_new[:, 0, :2]  # xy
	joint_sr_input[:, :2, -1] = joint_end_new[:, 0, :2]   # xy
	joint_sr_input[:, 2:, 0] = start_forward[:, :2]  # r
	joint_sr_input[:, 2:, -1] = end_forward[:, :2]  # r


	# normalize
	traj_mean = torch.tensor(traj_stats['traj_Xmean']).unsqueeze(0).unsqueeze(2)
	traj_std = torch.tensor(traj_stats['traj_Xstd']).unsqueeze(0).unsqueeze(2)

	joint_sr_input_normed = (joint_sr_input - traj_mean) / traj_std
	for t in range(joint_sr_input_normed.size(-1)):
		joint_sr_input_normed[:, :, t] = joint_sr_input_normed[:, :, 0] + (joint_sr_input_normed[:, :, -1] - joint_sr_input_normed[:, :, 0])*t/(joint_sr_input_normed.size(-1)-1)
		joint_sr_input_normed[:, -2:, t] = joint_sr_input_normed[:, -2:, t] / torch.norm(joint_sr_input_normed[:, -2:, t], dim=1).unsqueeze(1)

	for t in range(joint_sr_input.size(-1)):
		joint_sr_input[:, :, t] = joint_sr_input[:, :, 0] + (joint_sr_input[:, :, -1] - joint_sr_input[:, :, 0])*t/(joint_sr_input.size(-1)-1)
		joint_sr_input[:, -2:, t] = joint_sr_input[:, -2:, t] / torch.norm(joint_sr_input[:, -2:, t], dim=1).unsqueeze(1)

	# linear interpolation

	return joint_sr_input_normed.float().to(device), joint_sr_input.float().to(device), transf_rotmat, joint_start_new, joint_end_new