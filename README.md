# Lift3D: Synthesize 3D Training Data by Lifting 2D GAN to 3D Generative Radiance Field

[[Paper](https://len-li.github.io/assets/pdf/lift3d_final.pdf)]
[[Project Page](https://len-li.github.io/lift3d-web/)]


<img src="./assets/lift3d_method.png"/>


> **Lift3D: Synthesize 3D Training Data by Lifting 2D GAN to 3D Generative Radiance Field** <br>
> Leheng Li, Qing Lian, Luozhou Wang, Ningning Ma, Ying-Cong Chen <br>
> CVPR2023 <br>


This work explores the use of 3D generative models to synthesize training data for 3D vision tasks. The key requirements of the generative models are that the generated data should be photorealistic to match the real-world scenarios, and the corresponding 3D attributes should be aligned with given sampling labels. However, we find that the recent NeRF-based 3D GANs hardly meet the above requirements due to their designed generation pipeline and the lack of explicit 3D supervision.

In this work, we propose Lift3D, an inverted 2D-to-3D generation framework to achieve the data generation objectives. Lift3D has several merits compared to prior methods: (1) Unlike previous 3D GANs that the output resolution is fixed after training, Lift3D can generalize to any camera intrinsic with higher resolution and photorealistic output. (2) By lifting well-disentangled 2D GAN to 3D object NeRF, Lift3D provides explicit 3D information of generated objects, thus offering accurate 3D annotations for downstream tasks.

We evaluate the effectiveness of our framework by augmenting autonomous driving datasets. Experimental results demonstrate that our data generation framework can effectively improve the performance of 3D object detectors.

## Results

<img src="./assets/lift3d_compos.png"/>
The images are composited by our generated objects and original background.




## BibTeX

```bibtex
@InProceedings{lift3D2023CVPR, 
	author = {Leheng Li and Qing Lian and Luozhou Wang and Ningning Ma and Ying-Cong Chen}, 
	title = {Lift3D: Synthesize 3D Training Data by Lifting 2D GAN to 3D Generative Radiance Field}, 
	booktitle = {Proc. IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)}, 
	year = {2023}, 
}
```
