# Lift3D: Synthesize 3D Training Data by Lifting 2D GAN to 3D Generative Radiance Field

[[Paper](https://len-li.github.io/assets/pdf/lift3d_final.pdf)]
[[Project Page](https://len-li.github.io/lift3d-web/)]

> **Lift3D: Synthesize 3D Training Data by Lifting 2D GAN to 3D Generative Radiance Field** <br>
> Leheng Li, Qing Lian, Luozhou Wang, Ningning Ma, Ying-Cong Chen <br>
> Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2023 <br>

<!-- This repo is under construction. Please stay tuned. -->

## Installation

### 1. Clone repository
```
git clone https://github.com/Len-Li/Lift3D.git
cd Lift3D
```
### 2. Set up conda environment or use your existing one
```
conda create --name Lift3D python=3.8
conda activate Lift3D
```
### 3. Install the key requirement 
```
pip install torch torchvision torchaudio
pip install configargparse munch pillow
```
### 3. Download the checkpoint and object latents
- [Onedrive link](https://hkustgz-my.sharepoint.com/:f:/g/personal/lli181_connect_hkust-gz_edu_cn/EpEL6SOfZ85Mv90lB_3JUQUBSt9f_cf3gWJIXpRe5nl9bQ?e=IlJSIc)

Please download `lift3d_ckp.pt` and `obj_latent.pth`, then put them in the folder `ckp`


## Inference
```
python infer.py
```


## Acknowledgment
Additionally, we express our gratitude to the authors of the following opensource projects:

- [EG3D](https://github.com/NVlabs/eg3d) (tri-plane inplementation)
- [StyleSDF](https://github.com/royorel/StyleSDF) (NeRF training framework)





## BibTeX

```bibtex
@InProceedings{lift3D2023CVPR, 
	author = {Leheng Li and Qing Lian and Luozhou Wang and Ningning Ma and Ying-Cong Chen}, 
	title = {Lift3D: Synthesize 3D Training Data by Lifting 2D GAN to 3D Generative Radiance Field}, 
	booktitle = {Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)}, 
	year = {2023}, 
}
```
