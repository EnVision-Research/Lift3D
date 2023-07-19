import configargparse
from munch import *

class BaseOptions():
    def __init__(self):
        self.parser = configargparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # Dataset options
        dataset = self.parser.add_argument_group('dataset')
        dataset.add_argument("--dataset_path", type=str, default='./datasets/FFHQ', help="path to the lmdb dataset")

        # Experiment Options
        experiment = self.parser.add_argument_group('experiment')
        experiment.add_argument('--config', is_config_file=True, help='config file path')
        experiment.add_argument("--ckpt", type=str, default='300000', help="path to the checkpoints to resume training")
        experiment.add_argument("--continue_training", action="store_true", help="continue training the model")

        # Training loop options
        training = self.parser.add_argument_group('training')
        training.add_argument("--expname", type=str, default='debug', help='experiment name')
        training.add_argument("--checkpoints_dir", type=str, default='./checkpoint', help='checkpoints directory name')
        training.add_argument("--iter", type=int, default=300000, help="total number of training iterations")
        training.add_argument("--batch", type=int, default=4, help="batch sizes for each GPU. A single RTX2080 can fit batch=4, chunck=1 into memory.")
        training.add_argument("--chunk", type=int, default=4, help='number of samples within a batch to processed in parallel, decrease if running out of memory')
        training.add_argument("--local_rank", type=int, default=0, help="local rank for distributed training")
        training.add_argument("--lr", type=float, default=0.002, help="learning rate")
        training.add_argument("--pixel_lambda", type=float, default=1, help="weight of the pixel loss")
        training.add_argument("--semantic_lambda", type=float, default=0.2, help="weight of the semantic loss")
        training.add_argument("--occup_lambda", type=float, default=0.1, help="weight of the occupancy loss")
        training.add_argument("--eikonal_lambda", type=float, default=0.1, help="weight of the eikonal regularization")
        training.add_argument("--min_surf_lambda", type=float, default=0.0, help="weight of the minimal surface regularization")
        training.add_argument("--min_surf_beta", type=float, default=100.0, help="weight of the minimal surface regularization")
        training.add_argument("--wandb", action="store_true", help="use weights and biases logging")
        training.add_argument("--vec_class", type=str, default=None, help="path to the lmdb dataset")
        training.add_argument("--optim_point", type=int, default=None, help="path to the lmdb dataset")
        training.add_argument("--rays_num", type=int, default=8192*4, help="path to the lmdb dataset")

        # Generator options
        model = self.parser.add_argument_group('model')
        model.add_argument("--style_dim", type=int, default=512, help="number of style input dimensions")
        model.add_argument("--render_resolution", type=int, default=512, help='spatial resolution of the StyleGAN decoder inputs')

        # Camera options
        camera = self.parser.add_argument_group('camera')
        camera.add_argument("--fov", type=float, default=25.99, help="camera field of view half angle in Degrees")

        # Volume Renderer options
        rendering = self.parser.add_argument_group('rendering')
        rendering.add_argument("--no_sdf", action='store_true', help='By default, the raw MLP outputs represent an underline signed distance field (SDF). When true, the MLP outputs represent the traditional NeRF density field.')
        rendering.add_argument("--N_samples", type=int, default=24, help='number of samples per ray')
        rendering.add_argument("--no_offset_sampling", action='store_true', help='when true, use random stratified sampling when rendering the volume, otherwise offset sampling is used. (See Equation (3) in Sec. 3.2 of the paper)')
        rendering.add_argument("--perturb", type=float, default=1., help='set to 0. for no jitter, 1. for jitter')
        rendering.add_argument("--raw_noise_std", type=float, default=0., help='std dev of noise added to regularize sigma_a output, 1e0 recommended')
        rendering.add_argument("--use_eikonal", type=bool, default=False, help="image sizes for the model")
        rendering.add_argument("--use_semantic", type=bool, default=True, help="image sizes for the model")
        rendering.add_argument("--plane_reso", type=int, default=128, help="number of style input dimensions")
        rendering.add_argument("--plane_chann", type=int, default=64, help="number of style input dimensions")

        self.initialized = True

    def parse(self):
        self.opt = Munch()
        if not self.initialized:
            self.initialize()
        try:
            args = self.parser.parse_args()
        except: # solves argparse error in google colab
            args = self.parser.parse_args(args=[])

        for group in self.parser._action_groups[2:]:
            title = group.title
            self.opt[title] = Munch()
            for action in group._group_actions:
                dest = action.dest
                self.opt[title][dest] = args.__getattribute__(dest)

        return self.opt
