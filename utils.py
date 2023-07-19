import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from PIL import Image

from box_utils import box_pts

car_name = ['background', 'back bumper', 'bumper', 'car body', 'car_light_right', 'car_light_left','door_back', 'fender','door_front', 'grilles', 'back handle', 'front handle', 'hoods', 'license_plate_front', 'licence_plate_back','logo','mirror','roof','running boards', 'taillight right', 'taillight left','back wheel', 'front wheel','trunks','wheelhub_back','wheelhub_front','spoke_back', 'spoke_front', 'door_window_back', 'back windshield', 'door_window_front', 'windshield']

car_32_palette =[ 
  0,  0,  0,         # background
  238,  229,  102,
  220, 20, 60,
  124,  99 , 34,
  193 , 127,  15,
  106,  177,  21,
  248  ,213 , 42,
  252 , 155,  83,
  220  ,147 , 77,
  99 , 83  , 3,
  116 , 116 , 138,
  63  ,182 , 24,
  200  ,226 , 37,
  225 , 184 , 161,
  233 ,  5  ,219,
  142 , 172  ,248,
  153 , 112 , 146,
  38  ,112 , 254,
  229 , 30  ,141,
  115  ,208 , 131,
  52 , 83  ,84,
  229 , 63 , 110,
  194 , 87 , 125,
  225,  96  ,18,
  73  ,139,  226,
  172 , 143 , 16,
  169 , 101 , 111,
  31 , 102 , 211,
  104 , 131 , 101,
  70  ,168  ,156,
  183 , 242 , 209,
  72  ,184 , 226
  ]


def colorize_mask(mask):
    # mask: numpy array of the mask

    new_mask = Image.fromarray(mask.astype(np.uint8)).convert("P")
    new_mask.putpalette(car_32_palette)
    return np.array(new_mask.convert("RGB"))


def get_keys(d, name):
    if "state_dict" in d:
        d = d["state_dict"]
    d_filt = {k[len(name) + 1 :]: v for k, v in d.items() if k[: len(name)] == name}
    return d_filt


def create_samples(N=256, voxel_origin=[0, 0, 0], cube_length=2.0):
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = np.array(voxel_origin) - cube_length / 2
    voxel_size = cube_length / (N - 1)

    overall_index = torch.arange(0, N**3, 1, out=torch.LongTensor())
    samples = torch.zeros(N**3, 3)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.float() / N) % N
    samples[:, 0] = ((overall_index.float() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N**3

    return samples.unsqueeze(0), voxel_origin, voxel_size


######################### Dataset util functions ###########################
# Get data sampler
def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


# Get data minibatch
def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


############################## Model weights util functions #################
# Turn model gradients on/off
def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


# Exponential moving average for generator weights
def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


################# Camera parameters sampling ####################
def get_campara_blender(resolution, device, batch=1, fov_ang=6, pose_anno=None):

    # generate intrinsic parameters
    dist = torch.ones(batch, 1, device=device) * 1.5 * (13 / 8)
    # dist = torch.ones(batch, 1, device=device)
    fov_angle = (
        fov_ang * torch.ones(batch, 1, device=device) * np.pi / 180
    )  # full fov is 12 degrees
    focal = 0.5 * resolution / torch.tan(fov_angle).unsqueeze(-1)

    # the true fov is 51.98948897809546, half is 25.99

    azim = pose_anno[:, 0] + np.pi / 2
    elev = pose_anno[:, 1]

    #### Generate camera extrinsic matrix ##########

    # convert angles to xyz coordinates
    x = torch.cos(elev) * torch.sin(azim)
    y = torch.sin(elev)
    z = torch.cos(elev) * torch.cos(azim)
    camera_dir = torch.stack([x, y, z], dim=1).view(-1, 3)
    camera_loc = dist * camera_dir

    # get rotation matrices (assume object is at the world coordinates origin)
    up = torch.tensor([[0, -1, 0]]).float().to(device) * torch.ones_like(dist)
    z_axis = F.normalize(
        -camera_dir, eps=1e-5
    )  # the -z direction points into the screen
    x_axis = F.normalize(torch.cross(up, z_axis, dim=1), eps=1e-5)
    y_axis = F.normalize(torch.cross(z_axis, x_axis, dim=1), eps=1e-5)
    is_close = torch.isclose(x_axis, torch.tensor(0.0), atol=5e-3).all(
        dim=1, keepdim=True
    )
    if is_close.any():
        replacement = F.normalize(torch.cross(y_axis, z_axis, dim=1), eps=1e-5)
        x_axis = torch.where(is_close, replacement, x_axis)
    R = torch.cat((x_axis[:, None, :], y_axis[:, None, :], z_axis[:, None, :]), dim=1)
    T = camera_loc[:, :, None]
    extrinsics = torch.cat((R.transpose(1, 2), T), -1)

    return extrinsics, focal


def get_rays_full(focal, c2w, curr_size):

    height, width = curr_size

    # create meshgrid to generate rays
    i, j = torch.meshgrid(
        torch.linspace(0.5, width - 0.5, width),
        torch.linspace(0.5, height - 0.5, height),
    )

    i = i.t().unsqueeze(0).to(focal)
    j = j.t().unsqueeze(0).to(focal)

    dirs = torch.stack(
        [
            (i - width * 0.5) / focal,
            (j - height * 0.5) / focal,
            torch.ones_like(i).expand(focal.shape[0], height, width),
        ],
        -1,
    )

    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(
        dirs[..., None, :] * c2w[:, None, None, :3, :3], -1
    )  # dot product, equals to: [c2w.dot(dir) for dir in dirs]

    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:, None, None, :3, -1].expand(rays_d.shape)

    rays_full_dict = {"rays_o": rays_o, "rays_d": rays_d}

    return rays_full_dict


def get_rays_p2(cam_para, curr_size):

    height, width = curr_size

    # create meshgrid to generate rays
    i, j = torch.meshgrid(
        torch.linspace(0.5, width - 0.5, width),
        torch.linspace(0.5, height - 0.5, height),
    )

    i = i.t().unsqueeze(0).to(cam_para)
    j = j.t().unsqueeze(0).to(cam_para)

    rays_d = torch.stack(
        [
            (i - cam_para[0][2]) / cam_para[0][0],
            (j - cam_para[1][2]) / cam_para[1][1],
            torch.ones_like(i).expand(1, height, width),
        ],
        -1,
    )

    # rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    rays_o = torch.zeros_like(rays_d)
    rays_full_dict = {"rays_o": rays_o, "rays_d": rays_d}

    return rays_full_dict


def resample_rays(rays_full_dict, rgb_gt=None, semantic_gt=None, rays_num=8192):
    rays_o = rays_full_dict["rays_o"]
    rays_d = rays_full_dict["rays_d"]

    rays_d = rays_d.reshape((-1, 3))
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    rays_o = rays_o.reshape((-1, 3))

    if rgb_gt is not None:
        rgb_gt = rgb_gt.permute(0, 2, 3, 1).reshape((-1, 3))
        semantic_gt = semantic_gt.permute(0, 2, 3, 1).reshape((-1, 1))

    if rays_num == -1 or rays_o.shape[0] <= rays_num:
        rays_full_dict = {"rays_o": rays_o, "rays_d": rays_d}
        return rays_full_dict, rgb_gt, semantic_gt

    else:
        sel_index = np.random.choice(rays_o.shape[0], size=(rays_num,))
        rays_o = rays_o[sel_index]
        rays_d = rays_d[sel_index]
        rgb_gt = rgb_gt[sel_index]
        semantic_gt = semantic_gt[sel_index]
        rays_full_dict = {"rays_o": rays_o, "rays_d": rays_d}
        return rays_full_dict, rgb_gt, semantic_gt


def get_rays_box(rays_full_dict):
    rays_o = rays_full_dict["rays_o"]
    rays_d = rays_full_dict["rays_d"]

    lhw_array = rays_full_dict["car_size"][0].tolist()[::-1]
    lhw_array = [num * 2 for num in lhw_array]

    pose = (
        torch.tensor([0.0, lhw_array[1] / 2, 0.0])
        .expand(rays_o.shape)
        .unsqueeze(1)
        .to(rays_o)
    )
    theta_y = torch.tensor([0.0]).repeat(rays_o.shape[0], 1).to(rays_o)
    dim_repeat = torch.tensor(lhw_array).expand(rays_o.shape).unsqueeze(1).to(rays_o)

    world_info, box_info, intersection_map = box_pts(
        [rays_o, rays_d], pose, theta_y, dim_repeat, flag=False
    )
    rays_box_dict = {
        "world_info": world_info,
        "box_info": box_info,
        "intersection_map": intersection_map,
    }

    return rays_box_dict


def get_rays_box_sample(rays_full_dict, curr_box):
    rays_o = rays_full_dict["rays_o"]
    rays_d = rays_full_dict["rays_d"]

    car_bottom = curr_box[:3]
    cat_rota = curr_box[-1].item()
    l = curr_box[5].item()
    w = curr_box[4].item()
    h = curr_box[3].item()

    pose = car_bottom.expand(rays_o.shape).unsqueeze(1).to(rays_o)

    theta_y = torch.tensor([cat_rota]).repeat(rays_o.shape[0], 1).to(rays_o)
    dim_repeat = torch.tensor([l, h, w]).expand(rays_o.shape).unsqueeze(1).to(rays_o)

    world_info, box_info, intersection_map = box_pts(
        [rays_o, rays_d], pose, theta_y, dim_repeat, flag=False
    )
    rays_box_dict = {
        "world_info": world_info,
        "box_info": box_info,
        "intersection_map": intersection_map,
    }

    return rays_box_dict


#################### Mesh generation util functions ########################
# Reshape sampling volume to camera frostum
def align_volume(volume, near=0.88, far=1.12):
    b, h, w, d, c = volume.shape
    yy, xx, zz = torch.meshgrid(
        torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), torch.linspace(-1, 1, d)
    )

    grid = torch.stack([xx, yy, zz], -1).to(volume.device)

    frostum_adjustment_coeffs = (
        torch.linspace(far / near, 1, d).view(1, 1, 1, -1, 1).to(volume.device)
    )
    frostum_grid = grid.unsqueeze(0)
    frostum_grid[..., :2] = frostum_grid[..., :2] * frostum_adjustment_coeffs
    out_of_boundary = torch.any(
        (frostum_grid.lt(-1).logical_or(frostum_grid.gt(1))), -1, keepdim=True
    )
    frostum_grid = frostum_grid.permute(0, 3, 1, 2, 4).contiguous()
    permuted_volume = volume.permute(0, 4, 3, 1, 2).contiguous()
    final_volume = F.grid_sample(
        permuted_volume, frostum_grid, padding_mode="border", align_corners=True
    )
    final_volume = final_volume.permute(0, 3, 4, 2, 1).contiguous()
    # set a non-zero value to grid locations outside of the frostum to avoid marching cubes distortions.
    # It happens because pytorch grid_sample uses zeros padding.
    final_volume[out_of_boundary] = 1

    return final_volume
