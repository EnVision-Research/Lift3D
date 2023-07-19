import torch
import tqdm
import numpy as np

from networks import Lift3D as Model
from options import BaseOptions
from utils import (
    get_campara_blender,
    get_rays_full,
    resample_rays,
    get_rays_box,
    colorize_mask,
    resample_rays
)

from torchvision import utils


opt = BaseOptions().parse()
opt.rendering.N_samples = 64


generator = Model(opt.rendering, opt.model.style_dim)


lc_list = []
lc_loca = "./ckp/obj_latent.pth"
lc_template = torch.load(lc_loca, map_location="cpu")


generator.whether_train = False
generator.cutpaste = False

ckpt_path = "./ckp/lift3d_ckp.pt"
ckpt = torch.load(ckpt_path, map_location="cpu")
generator.load_state_dict(ckpt)


generator.require_grad = False
device = "cuda" if torch.cuda.is_available() else "cpu"
generator = generator.to(device)


for curr_pose_id in tqdm.tqdm(range(0, 100)):

    sample_cam = 40
    sample_cam_elev = 5
    all_sample = sample_cam * sample_cam_elev
    azim_range = np.pi
    elev_range = np.pi / 6
    pose_template_azim = 2 * azim_range / sample_cam * torch.arange(sample_cam)
    pose_template_elev = (
        torch.linspace(0, all_sample, all_sample) / all_sample * elev_range
    )

    curr_pose_id_azim = curr_pose_id % sample_cam

    curr_pose_azim = pose_template_azim[curr_pose_id_azim]
    curr_pose_elev = pose_template_elev[curr_pose_id]
    curr_pose = torch.tensor([curr_pose_azim, curr_pose_elev]).unsqueeze(0).to(device)

    curr_size = 500
    gt_size = (500, 500)
    add_dict = {}
    c2w, focal = get_campara_blender(
        curr_size, device, batch=1, fov_ang=opt.camera.fov, pose_anno=curr_pose
    )
    rays_full_dict = get_rays_full(focal, c2w, gt_size)
    rays_full_dict, rgb_gt, semantic_gt = resample_rays(
        rays_full_dict, None, None, rays_num=-1
    )
    rays_full_dict["car_size"] = np.array([[0.39, 0.35, 0.9]])
    rays_box_dict = get_rays_box(rays_full_dict)
    # rays_box_dict['car_size'] = add_dict['car_size']

    rays_box_dict["lc_shape"] = [x.to("cuda") for x in lc_template["lc_shape"]]
    rays_box_dict["lc_color"] = [x.to("cuda") for x in lc_template["lc_color"]]

    out = generator(None, add_opts=rays_box_dict)
    intersection_map = rays_box_dict["intersection_map"]
    height, width = gt_size

    rgb_out = torch.zeros((1, height, width, 3)).to(device) - 1
    semantic_out = torch.zeros((1, height, width, 32)).to(device)
    weight_out = torch.zeros((1, height, width, 1)).to(device)

    uv_grid = torch.meshgrid(
        torch.linspace(0, height - 1, height), torch.linspace(0, width - 1, width)
    )
    uv_grid = torch.cat([uv.unsqueeze(-1) for uv in uv_grid], -1)

    uv_grid = uv_grid.reshape((-1, 2))
    uv_box = uv_grid[intersection_map].to(device).long()[None, ...]
    weight = torch.sum(out["weight"], -2)
    out["rgb_map"][weight[..., 0] < 0.5, :] = -1

    # splat rays
    rgb_out[:, uv_box[:, :, 0], uv_box[:, :, 1], :] = out["rgb_map"]
    semantic_out[:, uv_box[:, :, 0], uv_box[:, :, 1], :] = out["semantic_map"]
    rgb_out = rgb_out.permute(0, 3, 1, 2)
    semantic_out = semantic_out.permute(0, 3, 1, 2)
    rgb_out = rgb_out / 2 + 0.5

    # color one hot mask
    semantic_bin = torch.argmax(semantic_out, 1)
    semantic_np = colorize_mask(semantic_bin[0].cpu().numpy())
    semantic_out = torch.tensor(semantic_np).permute(2, 0, 1).to(device)
    # semantic_out[:, semantic_bin[0]==0] = 0
    semantic_out = semantic_out[None, ...] / 255

    weight = torch.sum(out["weight"], -2)
    weight_out[:, uv_box[:, :, 0], uv_box[:, :, 1], :] = weight
    weight_out = weight_out.permute(0, 3, 1, 2)

    pred_mask = weight_out > 0.5
    pred_mask = pred_mask[0, 0] * 1

    rgb_out[:, :, pred_mask < 0.5] = 1
    semantic_out[semantic_out == 0] = 1

    utils.save_image(semantic_out, "test_out/sem_%04d.jpg" % curr_pose_id)
    utils.save_image(rgb_out, "test_out/rgb_%04d.jpg" % curr_pose_id)
