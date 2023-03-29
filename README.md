# Lift3D: Synthesize 3D Training Data by Lifting 2D GAN to 3D Generative Radiance Field

<img src="./docs/contents/framework.jpg"/>

**Figure:** *Framework of DiscoScene.*

> **Lift3D: Synthesize 3D Training Data by Lifting 2D GAN to 3D Generative Radiance Field** <br>
> Leheng Li, Qing Lian, Luozhou Wang, Ningning Ma, Ying-Cong Chen <br>

[[Paper](https://arxiv.org/abs/2212.11984)]
[[Project Page](https://len-li.github.io/lift3d-web/)]

This work presents DisCoScene: a 3D-aware generative model for high-quality and controllable scene synthesis.
The key ingredient of our approach is a very abstract object-level representation (3D bounding boxes without semantic annotation) as the scene layout prior, which is simple to obtain, general to describe various scene contents, and yet informative to disentangle objects and background. Moreover, it serves as an intuitive user control for scene editing.
Based on such a prior, our model spatially disentangles the whole scene into object-centric generative radiance fields by learning on only 2D images with the global-local discrimination. Our model obtains the generation fidelity and editing flexibility of individual objects while being able to efficiently compose objects and the background into a complete scene. We demonstrate state-of-the-art performance on many scene datasets, including the challenging Waymo outdoor dataset.

## Results

Qualitative comparison with EG-3D and Giraffe.
<img src="./docs/contents/fig_comparison.png"/>

Controllable scene synthesis.
<img src="./docs/contents/object-editing-v3-a.jpg"/>
<img src="./docs/contents/object-editing-v3-b.jpg"/>

Real image inversion and editing.
<img src="./docs/contents/inversion.jpg"/>



## BibTeX

```bibtex
@InProceedings{lift3D2023CVPR, 
	author = {Leheng Li and Qing Lian and Luozhou Wang and Ningning Ma and Ying-Cong Chen}, 
	title = {Lift3D: Synthesize 3D Training Data by Lifting 2D GAN to 3D Generative Radiance Field}, 
	booktitle = {Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)}, 
	year = {2023}, 
}
```
