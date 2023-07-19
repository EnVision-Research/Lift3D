import numpy as np
import torch
import time
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_objects_from_label(label_file):
    with open(label_file, "r") as f:
        lines = f.readlines()
    objects = [extract_3dbox(line) for line in lines]
    return torch.cat(objects, 0), lines


def extract_3dbox(line):
    label = line.strip().split(" ")
    box_3d = [
        float(label[4]),
        float(label[5]),
        float(label[6]),
        float(label[7]),
        float(label[11]),
        float(label[12]),
        float(label[13]),
        float(label[8]),
        float(label[9]),
        float(label[10]),
        float(label[14]),
    ]

    return torch.tensor(box_3d)[None, ...]


class Object3d(object):
    def __init__(self, line):
        label = line.strip().split(" ")
        self.box_3d = [
            float(label[11]),
            float(label[12]),
            float(label[13]),
            float(label[8]),
            float(label[9]),
            float(label[10]),
            float(label[14]),
        ]

    def get_obj_level(self):
        height = float(self.box2d[3]) - float(self.box2d[1]) + 1

        if self.trucation == -1:
            self.level_str = "DontCare"
            return 0

        if height >= 40 and self.trucation <= 0.15 and self.occlusion <= 0:
            self.level_str = "Easy"
            return 1  # Easy
        elif height >= 25 and self.trucation <= 0.3 and self.occlusion <= 1:
            self.level_str = "Moderate"
            return 2  # Moderate
        elif height >= 25 and self.trucation <= 0.5 and self.occlusion <= 2:
            self.level_str = "Hard"
            return 3  # Hard
        else:
            self.level_str = "UnKnown"
            return 4

    def generate_corners3d(self):
        """
        generate corners3d representation for this object
        :return corners_3d: (8, 3) corners of box3d in camera coord
        """
        l, h, w = self.l, self.h, self.w
        x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
        z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

        R = np.array(
            [
                [np.cos(self.ry), 0, np.sin(self.ry)],
                [0, 1, 0],
                [-np.sin(self.ry), 0, np.cos(self.ry)],
            ]
        )
        corners3d = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)
        corners3d = np.dot(R, corners3d).T
        corners3d = corners3d + self.pos
        return corners3d

    def to_bev_box2d(self, oblique=True, voxel_size=0.1):
        """
        :param bev_shape: (2) for bev shape (h, w), => (y_max, x_max) in image
        :param voxel_size: float, 0.1m
        :param oblique:
        :return: box2d (4, 2)/ (4) in image coordinate
        """
        if oblique:
            corners3d = self.generate_corners3d()
            xz_corners = corners3d[0:4, [0, 2]]
            box2d = np.zeros((4, 2), dtype=np.int32)
            box2d[:, 0] = ((xz_corners[:, 0] - Object3d.MIN_XZ[0]) / voxel_size).astype(
                np.int32
            )
            box2d[:, 1] = (
                Object3d.BEV_SHAPE[0]
                - 1
                - ((xz_corners[:, 1] - Object3d.MIN_XZ[1]) / voxel_size).astype(
                    np.int32
                )
            )
            box2d[:, 0] = np.clip(box2d[:, 0], 0, Object3d.BEV_SHAPE[1])
            box2d[:, 1] = np.clip(box2d[:, 1], 0, Object3d.BEV_SHAPE[0])
        else:
            box2d = np.zeros(4, dtype=np.int32)
            # discrete_center = np.floor((self.pos / voxel_size)).astype(np.int32)
            cu = np.floor((self.pos[0] - Object3d.MIN_XZ[0]) / voxel_size).astype(
                np.int32
            )
            cv = (
                Object3d.BEV_SHAPE[0]
                - 1
                - ((self.pos[2] - Object3d.MIN_XZ[1]) / voxel_size).astype(np.int32)
            )
            half_l, half_w = int(self.l / voxel_size / 2), int(self.w / voxel_size / 2)
            box2d[0], box2d[1] = cu - half_l, cv - half_w
            box2d[2], box2d[3] = cu + half_l, cv + half_w

        return box2d

    def to_str(self):
        print_str = (
            "%s %.3f %.3f %.3f box2d: %s hwl: [%.3f %.3f %.3f] pos: %s ry: %.3f"
            % (
                self.cls_type,
                self.trucation,
                self.occlusion,
                self.alpha,
                self.box2d,
                self.h,
                self.w,
                self.l,
                self.pos,
                self.ry,
            )
        )
        return print_str

    def to_kitti_format(self):
        kitti_str = (
            "%s %.2f %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f"
            % (
                self.cls_type,
                self.trucation,
                int(self.occlusion),
                self.alpha,
                self.box2d[0],
                self.box2d[1],
                self.box2d[2],
                self.box2d[3],
                self.h,
                self.w,
                self.l,
                self.pos[0],
                self.pos[1],
                self.pos[2],
                self.ry,
            )
        )
        return kitti_str


def get_calib_from_file(calib_file):
    with open(calib_file) as f:
        lines = f.readlines()

    obj = lines[2].strip().split(" ")[1:]
    P2 = np.array(obj, dtype=np.float32)
    obj = lines[3].strip().split(" ")[1:]
    P3 = np.array(obj, dtype=np.float32)
    obj = lines[4].strip().split(" ")[1:]
    R0 = np.array(obj, dtype=np.float32)
    obj = lines[5].strip().split(" ")[1:]
    Tr_velo_to_cam = np.array(obj, dtype=np.float32)

    return {
        "P2": P2.reshape(3, 4),
        "P3": P3.reshape(3, 4),
        "R0": R0.reshape(3, 3),
        "Tr_velo2cam": Tr_velo_to_cam.reshape(3, 4),
    }


def extract_bboxes(mask):

    """Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.
    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """

    boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)

    for i in range(mask.shape[-1]):

        m = mask[:, :, i]

        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]

        if horizontal_indicies.shape[0]:

            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.

            x2 += 1
            y2 += 1

        else:

            # No mask for this instance. Might happen due to

            # resizing or cropping. Set bbox to zeros

            x1, x2, y1, y2 = 0, 0, 0, 0

        boxes[i] = np.array([y1, x1, y2, x2])

    return boxes.astype(np.int32)


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


def gen_3dbox_rays(box_info):
    add_dict = {}
    device = box_info["curr_box"].device
    cam_para = box_info["P2"].to(device)
    height, width, _ = box_info["imgsize"]

    # create meshgrid to generate rays
    i, j = torch.meshgrid(
        torch.linspace(0.5, width - 0.5, width).to(device),
        torch.linspace(0.5, height - 0.5, height).to(device),
    )

    i = i.t().unsqueeze(0)
    j = j.t().unsqueeze(0)

    viewdirs = torch.stack(
        [
            (i - cam_para[0][2]) / cam_para[0][0],
            (j - cam_para[1][2]) / cam_para[1][1],
            torch.ones_like(i).expand(1, height, width),
        ],
        -1,
    )

    viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)

    ray_d = viewdirs.reshape((-1, 3))
    ray_o = torch.zeros_like(ray_d)

    curr_box = box_info["curr_box"]

    car_bottom = curr_box[:3]
    cat_rota = curr_box[-1].item()
    l = curr_box[5].item()
    w = curr_box[4].item()
    h = curr_box[3].item()

    add_dict["car_bottom"] = car_bottom
    add_dict["cat_rota"] = cat_rota
    add_dict["lwh"] = [l, w, h]

    pose = car_bottom.expand(ray_o.shape).unsqueeze(1).to(ray_o)
    theta_y = torch.tensor([cat_rota]).repeat(ray_o.shape[0], 1).to(ray_o)

    dim_repeat = torch.tensor([l, h, w]).expand(ray_o.shape).unsqueeze(1).to(ray_o)

    world_info, box_info, intersection_map = box_pts(
        [ray_o, ray_d], pose, theta_y, dim_repeat, flag=False
    )

    if world_info is not None:

        (
            rays_o_hit_w,
            rays_d_hit_w,
            viewdirs_box_w,
            z_vals_in_w,
            z_vals_out_w,
            pts_box_in_w,
            pts_box_out_w,
        ) = world_info
        (
            rays_o_hit_o,
            rays_d_hit_o,
            viewdirs_box_o,
            z_ray_in_o,
            z_ray_out_o,
            pts_box_in_o,
            pts_box_out_o,
        ) = box_info

        rgb_patch = torch.zeros((height, width, 3)).to(device)
        # rgb_patch = torch.tensor(curr_img).to(device)
        uv_grid = torch.meshgrid(
            torch.linspace(0, height - 1, height), torch.linspace(0, width - 1, width)
        )
        uv_grid = torch.cat([uv.unsqueeze(-1) for uv in uv_grid], -1)

        rgb_out = torch.zeros_like(rgb_patch)

        rgb_patch = rgb_patch.reshape((-1, 3))
        uv_grid = uv_grid.reshape((-1, 2))

        rgb_box = rgb_patch[intersection_map].to(device)
        uv_box = uv_grid[intersection_map].to(device)

        uv_box = uv_box.long()

        rgb_out[uv_box[:, 0], uv_box[:, 1], :] = rgb_box

        add_dict["pts_box_in_o"] = pts_box_in_o[None, ...]
        add_dict["pts_box_out_o"] = pts_box_out_o[None, ...]
        add_dict["z_vals_in_w"] = z_vals_in_w[None, ...]
        add_dict["z_vals_out_w"] = z_vals_out_w[None, ...]
        add_dict["z_ray_in_o"] = z_ray_in_o[None, ...].unsqueeze(-1)
        add_dict["z_ray_out_o"] = z_ray_out_o[None, ...].unsqueeze(-1)
        add_dict["uv_box"] = uv_box[None, ...]

        add_dict["rgb_patch_raw"] = rgb_out[None, ...] / 255.0
        add_dict["cam_para"] = cam_para

        return add_dict

    else:
        return None


def get_rays_(box_info):
    add_dict = {}
    device = box_info["curr_box"].device

    cam_para = box_info["P2"].to(device)

    height, width, _ = box_info["imgsize"]

    # create meshgrid to generate rays
    i, j = torch.meshgrid(
        torch.linspace(0.5, width - 0.5, width).to(device),
        torch.linspace(0.5, height - 0.5, height).to(device),
    )

    i = i.t().unsqueeze(0)
    j = j.t().unsqueeze(0)

    viewdirs = torch.stack(
        [
            (i - cam_para[0][2]) / cam_para[0][0],
            (j - cam_para[1][2]) / cam_para[1][1],
            torch.ones_like(i).expand(1, height, width),
        ],
        -1,
    )

    viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)

    ray_d = viewdirs.reshape((-1, 3))
    ray_o = torch.zeros_like(ray_d)

    curr_box = box_info["curr_box"]

    car_bottom = curr_box[:3]
    cat_rota = curr_box[-1].item()
    l = curr_box[5].item()
    w = curr_box[4].item()
    h = curr_box[3].item()

    add_dict["car_bottom"] = car_bottom
    add_dict["cat_rota"] = cat_rota
    add_dict["lwh"] = [l, w, h]

    pose = car_bottom.expand(ray_o.shape).unsqueeze(1).to(ray_o)
    theta_y = torch.tensor([cat_rota]).repeat(ray_o.shape[0], 1).to(ray_o)
    dim_repeat = torch.tensor([l, h, w]).expand(ray_o.shape).unsqueeze(1).to(ray_o)

    world_info, box_info, intersection_map = box_pts(
        [ray_o, ray_d], pose, theta_y, dim_repeat, flag=False
    )

    if world_info is not None:

        (
            rays_o_hit_w,
            rays_d_hit_w,
            viewdirs_box_w,
            z_vals_in_w,
            z_vals_out_w,
            pts_box_in_w,
            pts_box_out_w,
        ) = world_info
        (
            rays_o_hit_o,
            rays_d_hit_o,
            viewdirs_box_o,
            z_ray_in_o,
            z_ray_out_o,
            pts_box_in_o,
            pts_box_out_o,
        ) = box_info

        rgb_patch = torch.zeros((height, width, 3)).to(device)
        uv_grid = torch.meshgrid(
            torch.linspace(0, height - 1, height), torch.linspace(0, width - 1, width)
        )
        uv_grid = torch.cat([uv.unsqueeze(-1) for uv in uv_grid], -1)

        rgb_out = torch.zeros_like(rgb_patch)

        rgb_patch = rgb_patch.reshape((-1, 3))
        uv_grid = uv_grid.reshape((-1, 2))

        rgb_box = rgb_patch[intersection_map].to(device)
        uv_box = uv_grid[intersection_map].to(device)

        uv_box = uv_box.long()

        rgb_out[uv_box[:, 0], uv_box[:, 1], :] = rgb_box

        add_dict["pts_box_in_o"] = pts_box_in_o[None, ...]
        add_dict["pts_box_out_o"] = pts_box_out_o[None, ...]
        add_dict["z_vals_in_w"] = z_vals_in_w[None, ...]
        add_dict["z_vals_out_w"] = z_vals_out_w[None, ...]
        add_dict["z_ray_in_o"] = z_ray_in_o[None, ...].unsqueeze(-1)
        add_dict["z_ray_out_o"] = z_ray_out_o[None, ...].unsqueeze(-1)
        add_dict["uv_box"] = uv_box[None, ...]

        add_dict["rgb_patch_raw"] = rgb_out[None, ...] / 255.0
        add_dict["cam_para"] = cam_para

        return add_dict

    else:
        return None


def gather_nd(params, indices):
    # this function has a limit that MAX_ADVINDEX_CALC_DIMS=5
    ndim = indices.shape[-1]
    output_shape = list(indices.shape[:-1]) + list(params.shape[indices.shape[-1] :])
    flatted_indices = indices.view(-1, ndim)
    slices = [flatted_indices[:, i] for i in range(ndim)]
    slices += [Ellipsis]

    return params[slices].view(*output_shape)


def world2object(pts, dirs, pose, theta_y, dim=None, inverse=False):
    """Transform points given in world frame into N_obj object frames

    Object frames are scaled to [[-1.,1], [-1.,1], [-1.,1]] inside the 3D bounding box given by dim

    Args:
        pts: N_pts times 3D points given in world frame, [N_pts, 3]
        dirs: Corresponding 3D directions given in world frame, [N_pts, 3]
        pose: object position given in world frame, [N_pts, N_obj, 3]/if inverse: [N_pts, 3]
        theta_y: Yaw of objects around world y axis, [N_pts, N_obj]/if inverse: [N_pts]
        dim: Object bounding box dimensions, [N_pts, N_obj, 3]/if inverse: [N_pts, 3]
        inverse: if true pts and dirs should be given in object frame and are transofmed back into world frame, bool
            For inverse: pts, [N_pts, N_obj, 3]; dirs, [N_pts, N_obj, 3]

    Returns:
        pts_w: 3d points transformed into object frame (world frame for inverse task)
        dir_w: unit - 3d directions transformed into object frame (world frame for inverse task)
    """

    #  Prepare args if just one sample per ray-object or world frame only
    device = pts.device
    if len(pts.shape) == 3:
        # [batch_rays, n_obj, samples, xyz]
        n_sample_per_ray = pts.shape[1]

        pose = torch.repeat(pose, n_sample_per_ray, axis=0)
        theta_y = torch.repeat(theta_y, n_sample_per_ray, axis=0)
        if dim is not None:
            dim = torch.repeat(dim, n_sample_per_ray, axis=0)
        if len(dirs.shape) == 2:
            dirs = torch.repeat(dirs, n_sample_per_ray, axis=0)

        pts = torch.reshape(pts, [-1, 3])

    # Shift the object reference point to the middle of the bbox (vkitti2 specific)
    y_shift = (
        torch.Tensor([0.0, -1.0, 0.0]).to(device)[None, :]
        if inverse
        else torch.Tensor([0.0, -1.0, 0.0]).to(device)[None, None, :]
    ) * (dim[..., 1] / 2)[..., None]
    pose_w = pose + y_shift

    # Describes the origin of the world system w in the object system o
    t_w_o = rotate_yaw(-pose_w, theta_y)

    if not inverse:
        N_obj = theta_y.shape[1]

        pts_w = pts[:, None, ...].repeat(1, N_obj, 1)
        dirs_w = dirs[:, None, ...].repeat(1, N_obj, 1)

        # Rotate coordinate axis
        # TODO: Generalize for 3d roaations
        pts_o = rotate_yaw(pts_w, theta_y) + t_w_o
        dirs_o = rotate_yaw(dirs_w, theta_y)

        # Scale rays_o_v and rays_d_v for box [[-1.,1], [-1.,1], [-1.,1]]
        if dim is not None:
            pts_o = scale_frames(pts_o, dim)
            dirs_o = scale_frames(dirs_o, dim)

        # Normalize direction
        dirs_o = dirs_o / torch.norm(dirs_o, dim=3)[..., None, :]
        return [pts_o, dirs_o]

    else:
        pts_o = pts[None, :, None, :]
        dirs_o = dirs
        if dim is not None:
            pts_o = scale_frames(pts_o, dim[None, ...], inverse=True)
            if dirs is not None:
                dirs_o = scale_frames(dirs_o, dim, inverse=True)

        pts_o = pts_o - t_w_o
        pts_w = rotate_yaw(pts_o, -theta_y)[0, :]

        if dirs is not None:
            dirs_w = rotate_yaw(dirs_o, -theta_y)
            # Normalize direction
            dirs_w = dirs_w / torch.norm(dirs_w, dim=-1)[..., None, :]
        else:
            dirs_w = None

        return [pts_w, dirs_w]


def object2world(pts, dirs, pose, theta_y, dim=None, inverse=True):
    """Transform points given in world frame into N_obj object frames

    Object frames are scaled to [[-1.,1], [-1.,1], [-1.,1]] inside the 3D bounding box given by dim

    Args:
        pts: N_pts times 3D points given in N_obj object frames, [N_pts, N_obj, 3]
        dirs: Corresponding 3D directions given in N_obj object frames, [N_pts, N_obj, 3]
        pose: object position given in world frame, [N_pts, N_obj, 3]/if inverse: [N_pts, 3]
        theta_y: Yaw of objects around world y axis, [N_pts, N_obj]/if inverse: [N_pts]
        dim: Object bounding box dimensions, [N_pts, N_obj, 3]/if inverse: [N_pts, 3]

    Returns:
        pts_w: 3d points transformed into world frame
        dir_w: unit - 3d directions transformed into world frame
    """
    device = pts.device

    #  Prepare args if just one sample per ray-object
    if len(pts.shape) == 3:
        # [N_rays, N_obj, N_obj_samples, xyz]
        n_sample_per_ray = pts.shape[1]

        pose = torch.repeat(pose, n_sample_per_ray, axis=0)
        theta_y = torch.repeat(theta_y, n_sample_per_ray, axis=0)
        if dim is not None:
            dim = torch.repeat(dim, n_sample_per_ray, axis=0)
        if len(dirs.shape) == 2:
            dirs = torch.repeat(dirs, n_sample_per_ray, axis=0)

        pts = torch.reshape(pts, [-1, 3])

    # Shift the object reference point to the middle of the bbox (vkitti2 specific)
    y_shift = (
        torch.Tensor([0.0, -1.0, 0.0]).to(device)[None, :]
        * (dim[..., 1] / 2)[..., None]
    )
    pose_w = pose + y_shift

    # Describes the origin of the world system w in the object system o
    t_w_o = rotate_yaw(-pose_w, theta_y)

    pts_o = pts[None, :, None, :]
    dirs_o = dirs
    if dim is not None:
        pts_o = scale_frames(pts_o, dim[None, ...], inverse=True)
        if dirs is not None:
            dirs_o = scale_frames(dirs_o, dim, inverse=True)

    pts_o = pts_o - t_w_o
    pts_w = rotate_yaw(pts_o, -theta_y)[0, :]

    if dirs is not None:
        dirs_w = rotate_yaw(dirs_o, -theta_y)
        # Normalize direction
        dirs_w = dirs_w / torch.norm(dirs_w, axis=-1)[..., None, :]
    else:
        dirs_w = None

    return [pts_w, dirs_w]


def ray_box_intersection(ray_o, ray_d, aabb_min=None, aabb_max=None):
    """Returns 1-D intersection point along each ray if a ray-box intersection is detected

    If box frames are scaled to vertices between [-1., -1., -1.] and [1., 1., 1.] aabbb is not necessary

    Args:
        ray_o: Origin of the ray in each box frame, [rays, boxes, 3]
        ray_d: Unit direction of each ray in each box frame, [rays, boxes, 3]
        (aabb_min): Vertex of a 3D bounding box, [-1., -1., -1.] if not specified
        (aabb_max): Vertex of a 3D bounding box, [1., 1., 1.] if not specified

    Returns:
        z_ray_in:
        z_ray_out:
        intersection_map: Maps intersection values in z to their ray-box intersection
    """
    # Source: https://medium.com/@bromanz/another-view-on-the-classic-ray-aabb-intersection-algorithm-for-bvh-traversal-41125138b525
    # https://gamedev.stackexchange.com/questions/18436/most-efficient-aabb-vs-ray-collision-algorithms
    if aabb_min is None:
        aabb_min = torch.ones_like(ray_o) * -1.0  # torch.constant([-1., -1., -1.])
    if aabb_max is None:
        aabb_max = torch.ones_like(ray_o)  # torch.constant([1., 1., 1.])

    inv_d = torch.reciprocal(ray_d)  # inverse

    t_min = (aabb_min - ray_o) * inv_d
    t_max = (aabb_max - ray_o) * inv_d

    t0 = torch.minimum(t_min, t_max)
    t1 = torch.maximum(t_min, t_max)

    t_near = torch.maximum(torch.maximum(t0[..., 0], t0[..., 1]), t0[..., 2])
    t_far = torch.minimum(torch.minimum(t1[..., 0], t1[..., 1]), t1[..., 2])

    # Check if rays are inside boxes
    intersection_map = torch.where(t_far > t_near)
    # Check that boxes are in front of the ray origin
    intersection_map = torch.cat([x.unsqueeze(-1) for x in intersection_map], -1)
    positive_far = torch.where(gather_nd(t_far, intersection_map) > 0)
    positive_far = torch.cat([x.unsqueeze(-1) for x in positive_far], -1)
    intersection_map = gather_nd(intersection_map, positive_far)

    # # Check if rays are inside boxes
    # intersection_map = torch.where(t_far > t_near)[0].unsqueeze(-1)
    # # Check that boxes are in front of the ray origin
    # positive_far = torch.where(gather_nd(t_far, intersection_map) > 0)[0].unsqueeze(-1)
    # intersection_map = gather_nd(intersection_map, positive_far)

    if not intersection_map.shape[0] == 0:
        z_ray_in = gather_nd(t_near, intersection_map)
        z_ray_out = gather_nd(t_far, intersection_map)
    else:
        return None, None, None

    return z_ray_in, z_ray_out, intersection_map


def scale_frames(p, sc_factor, inverse=False):
    """Scales points given in N_frames in each dimension [xyz] for each frame or rescales for inverse==True

    Args:
        p: Points given in N_frames frames [N_points, N_frames, N_samples, 3]
        sc_factor: Scaling factor for new frame [N_points, N_frames, 3]
        inverse: Inverse scaling if true, bool

    Returns:
        p_scaled: Points given in N_frames rescaled frames [N_points, N_frames, N_samples, 3]
    """
    # Take 150% of bbox to include shadows etc.
    dim = torch.Tensor([1.0, 1.0, 1.0]).to(p) * sc_factor

    half_dim = dim / 2
    scaling_factor = (1 / (half_dim + 1e-9))[:, :, None, :]

    if not inverse:
        p_scaled = scaling_factor * p
    else:
        p_scaled = (1 / scaling_factor) * p

    return p_scaled


def ray_box_intersection_nogather(ray_o, ray_d, aabb_min=None, aabb_max=None):
    """Returns 1-D intersection point along each ray if a ray-box intersection is detected

    If box frames are scaled to vertices between [-1., -1., -1.] and [1., 1., 1.] aabbb is not necessary

    Args:
        ray_o: Origin of the ray in each box frame, [rays, boxes, 3]
        ray_d: Unit direction of each ray in each box frame, [rays, boxes, 3]
        (aabb_min): Vertex of a 3D bounding box, [-1., -1., -1.] if not specified
        (aabb_max): Vertex of a 3D bounding box, [1., 1., 1.] if not specified

    Returns:
        z_ray_in:
        z_ray_out:
        intersection_map: Maps intersection values in z to their ray-box intersection
    """
    # Source: https://medium.com/@bromanz/another-view-on-the-classic-ray-aabb-intersection-algorithm-for-bvh-traversal-41125138b525
    # https://gamedev.stackexchange.com/questions/18436/most-efficient-aabb-vs-ray-collision-algorithms
    if aabb_min is None:
        aabb_min = torch.ones_like(ray_o) * -1.0  # torch.constant([-1., -1., -1.])
    if aabb_max is None:
        aabb_max = torch.ones_like(ray_o)  # torch.constant([1., 1., 1.])

    inv_d = torch.reciprocal(ray_d)  # inverse

    t_min = (aabb_min - ray_o) * inv_d
    t_max = (aabb_max - ray_o) * inv_d

    t0 = torch.minimum(t_min, t_max)
    t1 = torch.maximum(t_min, t_max)

    t_near = torch.maximum(torch.maximum(t0[..., 0], t0[..., 1]), t0[..., 2])
    t_far = torch.minimum(torch.minimum(t1[..., 0], t1[..., 1]), t1[..., 2])

    # Check if rays are inside boxes
    intersection_map, _ = torch.where(t_far > t_near)
    positive_far, _ = torch.where(t_far[intersection_map] > 0)
    intersection_map = intersection_map[positive_far]

    if not intersection_map.shape[0] == 0:
        z_ray_in = t_near[intersection_map].squeeze(1)
        z_ray_out = t_far[intersection_map].squeeze(1)
    else:
        return None, None, None

    return z_ray_in, z_ray_out, intersection_map


def rotate_yaw(p, yaw):
    """Rotates p with yaw in the given coord frame with y being the relevant axis and pointing downwards

    Args:
        p: 3D points in a given frame [N_pts, N_frames, 3]/[N_pts, N_frames, N_samples, 3]
        yaw: Rotation angle

    Returns:
        p: Rotated points [N_pts, N_frames, N_samples, 3]
    """
    # p of size [batch_rays, n_obj, samples, xyz]
    if len(p.shape) < 4:
        p = p[..., None, :]

    c_y = torch.cos(yaw)[..., None]
    s_y = torch.sin(yaw)[..., None]

    p_x = c_y * p[..., 0] - s_y * p[..., 2]
    p_y = p[..., 1]
    p_z = s_y * p[..., 0] + c_y * p[..., 2]

    return torch.cat([p_x[..., None], p_y[..., None], p_z[..., None]], axis=-1)


def box_pts(rays, pose, theta_y, dim=None, one_intersec_per_ray=False, flag=True):
    """gets ray-box intersection points in world and object frames in a sparse notation

    Args:
        rays: ray origins and directions, [[N_rays, 3], [N_rays, 3]]
        pose: object positions in world frame for each ray, [N_rays, N_obj, 3]
        theta_y: rotation of objects around world y axis, [N_rays, N_obj]
        dim: object bounding box dimensions [N_rays, N_obj, 3]
        one_intersec_per_ray: If True only the first interesection along a ray will lead to an
        intersection point output

    Returns:
        pts_box_w: box-ray intersection points given in the world frame
        viewdirs_box_w: view directions of each intersection point in the world frame
        pts_box_o: box-ray intersection points given in the respective object frame
        viewdirs_box_o: view directions of each intersection point in the respective object frame
        z_vals_w: integration step in the world frame
        z_vals_o: integration step for scaled rays in the object frame
        intersection_map: mapping of points, viewdirs and z_vals to the specific rays and objects at the intersection

    """
    rays_o, rays_d = rays
    rays_o_o, rays_d_o = world2object(rays_o, rays_d, pose, theta_y, dim)
    rays_o_o = torch.squeeze(rays_o_o, -2)
    rays_d_o = torch.squeeze(rays_d_o, -2)
    num_of_objs = pose.shape[1]
    num_of_rays = rays_o.shape[0]

    if flag:
        z_ray_in_o, z_ray_out_o, intersection_map = ray_box_intersection(
            rays_o_o, rays_d_o
        )
    else:
        z_ray_in_o, z_ray_out_o, intersection_map = ray_box_intersection_nogather(
            rays_o_o, rays_d_o
        )

    if z_ray_in_o is not None:

        if flag:
            rays_o_hit_o = gather_nd(rays_o_o, intersection_map)  # n 3
            rays_d_hit_o = gather_nd(rays_d_o, intersection_map)
            pose_hit = gather_nd(pose, intersection_map)
            theta_hit = gather_nd(theta_y, intersection_map)
            dim_hit = gather_nd(dim, intersection_map)

            rays_o_hit_w = gather_nd(
                rays_o[:, None, :].repeat(1, num_of_objs, 1), intersection_map
            )
            rays_d_hit_w = gather_nd(
                rays_d[:, None, :].repeat(1, num_of_objs, 1), intersection_map
            )

        else:
            rays_o_hit_o = rays_o_o[intersection_map].squeeze(1)
            rays_d_hit_o = rays_d_o[intersection_map].squeeze(1)
            pose_hit = pose[intersection_map].squeeze(1)
            theta_hit = theta_y[intersection_map].squeeze(1)
            dim_hit = dim[intersection_map].squeeze(1)

            rays_o_hit_w = rays_o[intersection_map]
            rays_d_hit_w = rays_d[intersection_map]

        pts_box_in_o = rays_o_hit_o + z_ray_in_o[..., None] * rays_d_hit_o
        pts_box_in_w, _ = world2object(
            pts_box_in_o, None, pose_hit, theta_hit, dim_hit, inverse=True
        )
        pts_box_in_w = torch.squeeze(pts_box_in_w, -2)
        # Account for non-unit length rays direction
        z_vals_in_w = torch.norm(
            pts_box_in_w - rays_o_hit_w, dim=1, keepdim=True
        ) / torch.norm(rays_d_hit_w, dim=-1, keepdim=True)

        pts_box_out_o = rays_o_hit_o + z_ray_out_o[..., None] * rays_d_hit_o
        pts_box_out_w, _ = world2object(
            pts_box_out_o, None, pose_hit, theta_hit, dim_hit, inverse=True
        )
        pts_box_out_w = torch.squeeze(pts_box_out_w, -2)
        z_vals_out_w = torch.norm(
            pts_box_out_w - rays_o_hit_w, dim=1, keepdim=True
        ) / torch.norm(rays_d_hit_w, dim=-1, keepdim=True)

        # Get viewing directions for each ray-box intersection
        viewdirs_box_o = rays_d_hit_o
        viewdirs_box_w = 1 / torch.norm(rays_d_hit_w, dim=1)[:, None] * rays_d_hit_w

    else:
        # In case no ray intersects with any object return empty lists
        z_vals_in_w = z_vals_out_w = []
        pts_box_in_w = pts_box_in_o = []
        viewdirs_box_w = viewdirs_box_o = []
        z_ray_out_o = z_ray_in_o = []
        return None, None, None

    world_info = [
        rays_o_hit_w,
        rays_d_hit_w,
        viewdirs_box_w,
        z_vals_in_w,
        z_vals_out_w,
        pts_box_in_w,
        pts_box_out_w,
    ]
    box_info = [
        rays_o_hit_o,
        rays_d_hit_o,
        viewdirs_box_o,
        z_ray_in_o,
        z_ray_out_o,
        pts_box_in_o,
        pts_box_out_o,
    ]

    return world_info, box_info, intersection_map


if __name__ == "__main__":

    cam_para = np.array(
        [
            [7.215377e02, 0.000000e00, 6.095593e02, 4.485728e01],
            [0.000000e00, 7.215377e02, 1.728540e02, 2.163791e-01],
            [0.000000e00, 0.000000e00, 1.000000e00, 2.745884e-03],
        ]
    )

    img_path = "/home/leheng.li/my_nerf/obj_prior/scene-nerf/kitti_tracking/training/image_02/0006/000004.png"
    curr_img = cv2.imread(img_path)
    width, height = curr_img.shape[1], curr_img.shape[0]

    # create meshgrid to generate rays
    i, j = torch.meshgrid(
        torch.linspace(0.5, width - 0.5, width),
        torch.linspace(0.5, height - 0.5, height),
    )

    i = i.t().unsqueeze(0)
    j = j.t().unsqueeze(0)

    viewdirs = torch.stack(
        [
            (i - cam_para[0][2]) / cam_para[0][0],
            (j - cam_para[1][2]) / cam_para[1][1],
            torch.ones_like(i).expand(1, height, width),
        ],
        -1,
    )

    viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)

    ray_d = viewdirs.reshape((-1, 3)).to(device)
    ray_o = torch.zeros_like(ray_d)

    pose = (
        torch.tensor([-5.6372, 1.6417, 8.6842])
        .expand(ray_o.shape)
        .unsqueeze(1)
        .to(ray_o)
    )
    theta_y = torch.tensor([2.1658]).repeat(ray_o.shape[0], 1).to(ray_o)
    dim_repeat = (
        torch.tensor([3.5201, 1.4165, 1.4750])
        .expand(ray_o.shape)
        .unsqueeze(1)
        .to(ray_o)
    )

    batch_num = 1

    i = 0
    torch.cuda.synchronize()
    t1 = time.time()
    while i < 50:
        i += 1

        world_info, box_info, intersection_map = box_pts(
            [ray_o[::batch_num], ray_d[::batch_num]],
            pose[::batch_num],
            theta_y[::batch_num],
            dim_repeat[::batch_num],
            flag=False,
        )

        if world_info is not None:

            (
                rays_o_hit_w,
                rays_d_hit_w,
                viewdirs_box_w,
                z_vals_in_w,
                z_vals_out_w,
                pts_box_in_w,
                pts_box_out_w,
            ) = world_info
            (
                rays_o_hit_o,
                rays_d_hit_o,
                viewdirs_box_o,
                z_ray_in_o,
                z_ray_out_o,
                pts_box_in_o,
                pts_box_out_o,
            ) = box_info

            rgb_patch = torch.tensor(curr_img).to(device)
            uv_grid = torch.meshgrid(
                torch.linspace(0, height - 1, height),
                torch.linspace(0, width - 1, width),
            )
            uv_grid = torch.cat([uv.unsqueeze(-1) for uv in uv_grid], -1)

            rgb_out = torch.zeros_like(rgb_patch)

            # rgb_patch_raw = rgb_patch.clone()
            rgb_patch = rgb_patch.reshape((-1, 3))
            uv_grid = uv_grid.reshape((-1, 2))

            rgb_box = rgb_patch[intersection_map].to(device)
            uv_box = uv_grid[intersection_map].to(device)

            uv_box = uv_box.long()

            rgb_out[uv_box[:, 0], uv_box[:, 1], :] = rgb_box

    torch.cuda.synchronize()
    t2 = time.time()
    print("cuda", t2 - t1)
