import cv2
import torch

import numpy as np
from torch import nn
import torch.autograd as autograd
from torch.nn import functional as F
from PIL import Image, ImageDraw

from training.networks_stylegan2 import Generator_triplan as StyleGAN2Backbone
# from torch_utils.ops import grid_sample_gradfix_stylegan3 as grid_sample_gradfix


def rectangle(output_path=None):
    image = Image.new("RGB", (500, 500), "black")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(
        (0, 0, 500, 500), fill="white", outline="white", width=3, radius=150
    )
    # image.save(output_path)
    return np.array(image)


def generate_planes():
    """
    Defines planes by the three vectors that form the "axes" of the
    plane. Should work with arbitrary number of planes and planes of
    arbitrary orientation.
    """
    return torch.tensor(
        [
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[1, 0, 0], [0, 0, 1], [0, 1, 0]],
            [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
        ],
        dtype=torch.float32,
    )


# Basic SIREN fully connected layer
class LinearLayer(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        bias=True,
        bias_init=0,
        std_init=1,
        freq_init=False,
        is_first=False,
    ):
        super().__init__()
        if is_first:
            self.weight = nn.Parameter(
                torch.empty(out_dim, in_dim).uniform_(-1 / in_dim, 1 / in_dim)
            )
        elif freq_init:
            self.weight = nn.Parameter(
                torch.empty(out_dim, in_dim).uniform_(
                    -np.sqrt(6 / in_dim) / 25, np.sqrt(6 / in_dim) / 25
                )
            )
        else:
            self.weight = nn.Parameter(
                0.25
                * nn.init.kaiming_normal_(
                    torch.randn(out_dim, in_dim),
                    a=0.2,
                    mode="fan_in",
                    nonlinearity="leaky_relu",
                )
            )

        self.bias = nn.Parameter(
            nn.init.uniform_(
                torch.empty(out_dim), a=-np.sqrt(1 / in_dim), b=np.sqrt(1 / in_dim)
            )
        )

        self.bias_init = bias_init
        self.std_init = std_init

    def forward(self, input):
        out = (
            self.std_init * F.linear(input, self.weight, bias=self.bias)
            + self.bias_init
        )

        return out


# Siren layer with frequency modulation and offset
class FiLMSiren(nn.Module):
    def __init__(self, in_channel, out_channel, style_dim, is_first=False):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel

        if is_first:
            self.weight = nn.Parameter(
                torch.empty(out_channel, in_channel).uniform_(-1 / 3, 1 / 3)
            )
        else:
            self.weight = nn.Parameter(
                torch.empty(out_channel, in_channel).uniform_(
                    -np.sqrt(6 / in_channel) / 25, np.sqrt(6 / in_channel) / 25
                )
            )

        self.bias = nn.Parameter(
            nn.Parameter(
                nn.init.uniform_(
                    torch.empty(out_channel),
                    a=-np.sqrt(1 / in_channel),
                    b=np.sqrt(1 / in_channel),
                )
            )
        )
        self.activation = torch.sin

        self.gamma = LinearLayer(style_dim, out_channel, bias_init=30, std_init=15)
        self.beta = LinearLayer(style_dim, out_channel, bias_init=0, std_init=0.25)

    def forward(self, input, style):
        batch, features = style.shape
        out = F.linear(input, self.weight, bias=self.bias)
        gamma = self.gamma(style).view(batch, 1, 1, 1, self.out_channel)
        beta = self.beta(style).view(batch, 1, 1, 1, self.out_channel)

        out = self.activation(gamma * out + beta)

        return out


class OSGDecoderFilm(torch.nn.Module):
    def __init__(self, n_features, hidden_dim, cat, other_fea=0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.other_fea = other_fea

        self.cat = cat
        if self.cat:
            n_features = n_features * 3
        else:
            pass

        self.net = FiLMSiren(n_features, self.hidden_dim, style_dim=512)
        self.rgb_linear = LinearLayer(self.hidden_dim, 3, freq_init=True)
        self.sigma_linear = LinearLayer(self.hidden_dim, 1, freq_init=True)

        if self.other_fea > 0:
            self.other_linear = LinearLayer(self.hidden_dim, other_fea, freq_init=True)

    def forward(self, x, style):
        # Aggregate features
        if self.cat:
            x = x.permute(0, 2, 1, 3).reshape(x.shape[0], x.shape[2], -1)
        else:
            x = x.mean(1)

        N, M, C = x.shape
        x = x.view(N * M, C)

        x = self.net(x, style)

        rgb = self.rgb_linear(x)
        sigma = self.sigma_linear(x)

        if self.other_fea > 0:
            others = self.other_linear(x)
            x = torch.cat([rgb, sigma, others], -1)
        else:
            x = torch.cat([rgb, sigma], -1)

        x = x.view(N, M, -1)

        return x


# Full volume renderer
class Lift3D(nn.Module):
    def __init__(self, opt, style_dim=512, mode="train"):
        super().__init__()
        self.whether_train = mode == "train"
        self.perturb = opt.perturb
        self.offset_sampling = (
            not opt.no_offset_sampling
        )  # Stratified sampling used otherwise
        self.N_samples = opt.N_samples
        self.raw_noise_std = opt.raw_noise_std
        self.with_sdf = not opt.no_sdf

        if self.with_sdf:
            self.sigmoid_beta = nn.Parameter(0.1 * torch.ones(1))

        self.use_eikonal = opt.use_eikonal
        self.use_semantic = opt.use_semantic

        if self.use_semantic:
            self.semantic_channel = 32
            self.sem_index = None
        else:
            self.semantic_channel = 0

        self.channel_dim = -1
        self.samples_dim = 3

        # create integration values
        if self.offset_sampling:
            t_vals = torch.linspace(
                0.0, 1.0 - 1 / self.N_samples, steps=self.N_samples
            ).view(1, 1, 1, -1)
        else:  # Original NeRF Stratified sampling
            t_vals = torch.linspace(0.0, 1.0, steps=self.N_samples).view(1, 1, 1, -1)

        self.register_buffer("t_vals", t_vals, persistent=False)
        self.register_buffer("inf", torch.Tensor([1e10]), persistent=False)
        self.register_buffer("zero_idx", torch.LongTensor([0]), persistent=False)

        if not self.whether_train:
            self.perturb = False
            self.raw_noise_std = 0.0

        self.plane_reso = opt.plane_reso
        self.plane_chann = opt.plane_chann

        self.backbone = StyleGAN2Backbone(
            z_dim=style_dim,
            c_dim=0,
            w_dim=style_dim,
            img_resolution=self.plane_reso,
            img_channels=self.plane_chann * 3,
        )  # z_dim c_dim w_dim
        self.final_fc = OSGDecoderFilm(
            self.plane_chann,
            self.plane_chann,
            cat=False,
            other_fea=self.semantic_channel,
        )
        self.plane_axes = generate_planes()

        self.sample_points_num = opt.N_samples

        self.cutpaste = False

        self.mask = rectangle()
        self.mask = cv2.GaussianBlur(self.mask, (5, 5), 0)
        # self.shadowmap = cv2.imread(
        #     "/home/abc/adv3d/lift3d/models/shadowmap.png", cv2.IMREAD_GRAYSCALE
        # )
        # self.shadowmap = cv2.resize(self.shadowmap, (500, 500))[..., None]
        self.shadowmap = None

    def get_eikonal_term(self, pts, sdf):
        eikonal_term = autograd.grad(
            outputs=sdf,
            inputs=pts,
            grad_outputs=torch.ones_like(sdf),
            create_graph=True,
            #  allow_unused=True,
        )[0]
        return eikonal_term

    def sdf_activation(self, input):
        sigma = torch.sigmoid(input / self.sigmoid_beta) / self.sigmoid_beta

        return sigma

    def run_network(self, inputs, viewdirs, styles=None, add_opts=None):

        if add_opts.get("lc_shape", None) != None:  # optimize planes
            planes_shape, noise_shape = add_opts["lc_shape"]
            planes_color, noise_color = add_opts["lc_color"]
            N, n_planes, C, H, W = planes_shape.shape
            planes_shape = planes_shape.view(N * n_planes, C, H, W)
            planes_color = planes_color.view(N * n_planes, C, H, W)

            noise_shape = (
                noise_shape[:, 4:, :]
                if self.plane_reso < 256
                else noise_shape[:, 2:, :]
            )
            noise_color = (
                noise_color[:, 4:, :]
                if self.plane_reso < 256
                else noise_color[:, 2:, :]
            )

            s1, s3, s4, s5 = inputs.shape
            inputs = inputs.reshape((s1, -1, s5))
            _, M, _ = inputs.shape
            projected_coordinates = self.project_onto_planes(
                self.plane_axes.to(inputs), inputs
            ).unsqueeze(1)

            shape_color_seperate = 1
            if shape_color_seperate:
                with torch.no_grad():
                    out_fea = (
                        F.grid_sample(planes_shape, projected_coordinates.float())
                        .permute(0, 3, 2, 1)
                        .reshape(N, n_planes, M, C)
                    )
                    # out_fea = grid_sample_gradfix.grid_sample(planes_shape, projected_coordinates.float()).permute(0, 3, 2, 1).reshape(N, n_planes, M, C)
                    raw_shape = self.final_fc(out_fea, torch.mean(noise_shape, 1))
                    raw_shape = raw_shape.reshape((s1, s3, s4, raw_shape.shape[-1]))
                out_fea = (
                    F.grid_sample(planes_color, projected_coordinates.float())
                    .permute(0, 3, 2, 1)
                    .reshape(N, n_planes, M, C)
                )
                # out_fea = grid_sample_gradfix.grid_sample(planes_color, projected_coordinates.float()).permute(0, 3, 2, 1).reshape(N, n_planes, M, C)
                raw = self.final_fc(out_fea, torch.mean(noise_color, 1))
                raw = raw.reshape((s1, s3, s4, raw.shape[-1]))

                if self.sem_index is not None:
                    with torch.no_grad():
                        semantic_index = torch.argmax(raw_shape[..., 4:], -1)
                        sem_select = semantic_index != self.sem_index[0]

                        for i in self.sem_index[1:]:
                            sem_select_tmp = semantic_index != i
                            sem_select = sem_select & sem_select_tmp

                        raw[..., :3][sem_select] = raw_shape[..., :3][sem_select]

                raw = torch.cat([raw[..., :3], raw_shape[..., 3:]], -1)
                return raw
            else:
                out_fea = (
                    F.grid_sample(planes_color, projected_coordinates.float())
                    .permute(0, 3, 2, 1)
                    .reshape(N, n_planes, M, C)
                )
                # out_fea = grid_sample_gradfix.grid_sample(planes_color, projected_coordinates.float()).permute(0, 3, 2, 1).reshape(N, n_planes, M, C)
                raw = self.final_fc(out_fea, torch.mean(noise_color, 1))
                raw = raw.reshape((s1, s3, s4, raw.shape[-1]))
                return raw

        else:  # inference planes (training stage)
            styles = styles[:, 4:, :] if self.plane_reso < 256 else styles[:, 2:, :]

            planes = self.backbone(styles, None)
            planes = planes.view(
                len(planes), 3, self.plane_chann, planes.shape[-2], planes.shape[-1]
            )
            N, n_planes, C, H, W = planes.shape
            planes = planes.view(N * n_planes, C, H, W)

            s1, s3, s4, s5 = inputs.shape
            inputs = inputs.reshape((s1, -1, s5))  # * 0.9
            _, M, _ = inputs.shape
            projected_coordinates = self.project_onto_planes(
                self.plane_axes.to(inputs), inputs
            ).unsqueeze(1)

            out_fea = (
                # grid_sample_gradfix.grid_sample(planes, projected_coordinates.float())
                F.grid_sample(planes, projected_coordinates.float())
                .permute(0, 3, 2, 1)
                .reshape(N, n_planes, M, C)
            )
            raw = self.final_fc(out_fea, torch.mean(styles, 1))
            raw = raw.reshape((s1, s3, s4, raw.shape[-1]))

            return raw

    def project_onto_planes(self, planes, coordinates):
        """
        Does a projection of a 3D point onto a batch of 2D planes,
        returning 2D plane coordinates.

        Takes plane axes of shape n_planes, 3, 3
        # Takes coordinates of shape N, M, 3
        # returns projections of shape N*n_planes, M, 2
        """
        N, M, C = coordinates.shape
        n_planes, _, _ = planes.shape
        coordinates = (
            coordinates.unsqueeze(1)
            .expand(-1, n_planes, -1, -1)
            .reshape(N * n_planes, M, 3)
        )
        inv_planes = (
            torch.linalg.inv(planes)
            .unsqueeze(0)
            .expand(N, -1, -1, -1)
            .reshape(N * n_planes, 3, 3)
        )
        projections = torch.bmm(coordinates, inv_planes)
        return projections[..., :2]

    def forward(self, styles=None, add_opts=None):
        outputs_dict = self.render_rays(styles=styles, add_dict=add_opts)
        return outputs_dict

    def render_rays(self, styles=None, add_dict=None):

        _, _, _, z_vals_in_w, z_vals_out_w, _, _ = add_dict["world_info"]
        _, _, _, _, _, pts_box_in_o, pts_box_out_o = add_dict["box_info"]

        z_vals_in_w = z_vals_in_w[None, ...]
        z_vals_out_w = z_vals_out_w[None, ...]
        pts_box_in_o = pts_box_in_o[None, ...].unsqueeze(-1)
        pts_box_out_o = pts_box_out_o[None, ...].unsqueeze(-1)

        ww = torch.linspace(0, 1, self.sample_points_num).unsqueeze(0)

        if self.perturb > 0.0 and self.whether_train:
            if self.offset_sampling:
                # random offset samples
                upper = torch.cat([ww[..., 1:], torch.ones_like(ww[..., -1:])], -1)
                lower = ww.detach()
                t_rand = torch.rand_like(ww)
                ww = lower + (upper - lower) * t_rand

        sample_point = torch.lerp(pts_box_in_o, pts_box_out_o, ww.to(pts_box_in_o))
        sample_z = torch.lerp(z_vals_in_w, z_vals_out_w, ww.to(pts_box_in_o))

        sample_point = sample_point.permute(0, 1, 3, 2)
        sample_z = sample_z.unsqueeze(1)

        if self.cutpaste:
            sample_point[..., 1] = -sample_point[..., 1]
            x_min, x_max, y_min, y_max, z_min, z_max = -1, 1, -1, 1, -1, 1
            shadow = 1
            # shadow = 0
            if shadow:
                x_bound_m = x_min
                y_bound_m = y_min - 0.1
                z_bound_m = z_min
                x_bound = x_max
                y_bound = y_min + 0.06
                z_bound = z_max

                point_flag = (
                    (sample_point[..., 0] > x_bound_m)
                    & (sample_point[..., 0] < x_bound)
                    & (sample_point[..., 1] > y_bound_m)
                    & (sample_point[..., 1] < y_bound)
                    & (sample_point[..., 2] > z_bound_m)
                    & (sample_point[..., 2] < z_bound)
                )
            else:
                point_flag = sample_point[..., 0] > 1000000

        if self.use_eikonal and self.whether_train:
            sample_point.requires_grad = True

        raw = self.run_network(sample_point, None, styles=styles, add_opts=add_dict)

        raw[..., :3] = torch.sigmoid(raw[..., :3])

        if self.cutpaste:  # add shadow
            shadowmap = torch.tensor(self.shadowmap).to(raw).repeat(1, 1, 3)
            shadowmap = shadowmap.permute(2, 1, 0)[None, ...]
            clip_map = 1 - torch.clip((1 - shadowmap / 255) * 3, min=0, max=1)
            mask = clip_map

            xz_coord = sample_point[point_flag][:, [0, 2]]
            xz_coord[:, 0] = (
                (xz_coord[:, 0] - (x_min + x_max) / 2) / ((x_max - x_min) / 2) * 0.95
            )  # l
            xz_coord[:, 1] = (
                (xz_coord[:, 1] - (z_min + z_max) / 2) / ((z_max - z_min) / 2) * 0.95
            )
            xz_coord = xz_coord[None, None, ...]

            bottom_rgb = F.grid_sample(
                mask, grid=xz_coord, mode="nearest", align_corners=True
            )
            bottom_rgb = bottom_rgb.squeeze(2).squeeze(0).permute(1, 0).to(raw)
            # bottom_rgb = bottom_rgb.squeeze().permute(1,0).to(raw)
            round_flag = bottom_rgb[:, 0] < 0.4
            # bottom_rgb[bottom_rgb < 0.9] = -1000
            raw = raw.unsqueeze(1)
            raw_rgb = raw[..., :3]
            raw_sigma = raw[..., 3:4]
            raw_others = raw[..., 4:]
            new_sel = point_flag[None, ...]
            weight_pts = torch.norm(xz_coord, dim=-1)
            weight_pts = torch.clip((1 - weight_pts), min=0, max=1)
            original_rgb = raw_rgb[new_sel]
            original_rgb[round_flag] = bottom_rgb[round_flag]
            # original_rgb[round_flag] = bottom_rgb[round_flag] * 2 - 1
            # original_rgb[round_flag] = -1000
            original_sigma = raw_sigma[new_sel]
            original_sigma[round_flag] = 0
            raw_rgb[new_sel] = original_rgb
            raw_sigma[new_sel] = original_sigma
            raw = torch.cat([raw_rgb, raw_sigma, raw_others], -1)

        outputs_dict = self.volume_rendering(raw, sample_z, sample_point)

        return outputs_dict

    def volume_rendering(self, raw, z_vals, pts=None, rays_d=None):
        dists = torch.norm(pts[..., 1:, :] - pts[..., :-1, :], p=2, dim=3)
        dists = torch.abs(dists).unsqueeze(0)
        dists = torch.cat([dists, dists[..., -1:]], self.channel_dim)

        rgb, sdf, semantic = torch.split(
            raw, [3, 1, self.semantic_channel], dim=self.channel_dim
        )

        noise = 0.0
        if self.raw_noise_std > 0.0:
            noise = torch.randn_like(sdf) * self.raw_noise_std

        if self.with_sdf:
            sigma = self.sdf_activation(-sdf)

            if self.use_eikonal and self.whether_train:
                eikonal_term = self.get_eikonal_term(pts, sdf)
            else:
                eikonal_term = None

            sigma = 1 - torch.exp(-sigma * dists.unsqueeze(self.channel_dim))
        else:
            sigma = sdf
            eikonal_term = None

            sigma = 1 - torch.exp(
                -F.softplus(sigma + noise) * dists.unsqueeze(self.channel_dim)
            )

        visibility = torch.cumprod(
            torch.cat(
                [
                    torch.ones_like(
                        torch.index_select(sigma, self.samples_dim, self.zero_idx)
                    ),
                    1.0 - sigma + 1e-10,
                ],
                self.samples_dim,
            ),
            self.samples_dim,
        )
        visibility = visibility[..., :-1, :]
        weights = sigma * visibility

        rgb_map = -1 + 2 * torch.sum(
            weights * rgb, self.samples_dim
        )  # switch to [-1,1] value range
        semantic_map = torch.sum(weights * semantic, self.samples_dim)

        outputs_dict = {}
        outputs_dict["rgb_map"] = rgb_map
        outputs_dict["semantic_map"] = semantic_map
        outputs_dict["weight"] = weights

        if self.use_eikonal and self.whether_train:
            outputs_dict["sdf_out"] = sdf
            outputs_dict["eikonal_term"] = eikonal_term

        return outputs_dict
