# Stereo3DMOT

<b>Stereo3DMOT: Stereo Vision Based 3D Multi-Object Tracking with Multimodal ReID (PRCV 2023), Chen Mao</b>

This repository contains the official python implementation for our paper at PRCV 2023 "[Stereo3DMOT: Stereo Vision Based 3D Multi-Object Tracking with Multimodal ReID](https://link.springer.com/chapter/10.1007/978-981-99-8555-5_39)".
If you find my paper or code useful, please cite my papers:

```
@inproceedings{mao2023stereo3dmot,
  title={Stereo3DMOT: Stereo Vision Based 3D Multi-object Tracking with Multimodal ReID},
  author={Mao, Chen and Tan, Chong and Liu, Hong and Hu, Jingqi and Zheng, Min},
  booktitle={Chinese Conference on Pattern Recognition and Computer Vision (PRCV)},
  pages={495--507},
  year={2023},
  organization={Springer}
}
```

<img align="center" width="100%" src="https://github.com/Chain-Mao/Stereo3DMOT/blob/master/test1.gif">
<img align="center" width="100%" src="https://github.com/Chain-Mao/Stereo3DMOT/blob/master/test2.gif">

## Abstract
<img align="center" width="100%" src="https://github.com/Chain-Mao/Stereo3DMOT/blob/master/network.png">

3D Multi-Object Tracking (MOT) is a key component in numerous applications, such as autonomous driving and intelligent robotics, playing a crucial role in the perception and decision-making processes of intelligent systems. In this paper, we propose a 3D MOT system based on a cost-effective stereo camera pair, which includes a 3D multimodal re-identification (ReID) model capable of multi-task learning. The ReID model obtains the multimodal features of objects, including RGB and point cloud information. We design data association and trajectory management algorithms. The data association computes an affinity matrix for the object feature embeddings and motion information, while the trajectory management controls the lifecycle of the trajectories. In addition, we create a ReID dataset based on the KITTI Tracking dataset, used for training and validating ReID models. Results demonstrate that our method can achieve accurate object tracking solely with a stereo camera pair, maintaining high reliability even in cases of occlusion and missed detections. Experimental evaluation shows that our approach outperforms competitive results on the KITTI MOT leaderboard.

<img align="center" width="100%" src="https://github.com/Chain-Mao/Stereo3DMOT/blob/master/architecture.png">

## Benchmarking

We provide instructions (inference, evaluation and visualization) for reproducing our method's performance on various supported datasets ([KITTI](docs/KITTI.md)) for benchmarking purposes. 

## Issues
If you have any questions regarding our code or paper, feel free to open an issue or send an email to maochen981203@gmail.com
    
## Acknowledgements
Parts of this repo are inspired by the following repositories:
- [AB3MOT in PyTorch](https://github.com/xinshuoweng/AB3DMOT)
- [Fast-ACVNet in PyTorch](https://github.com/gangweiX/Fast-ACVNet)
- [PointNet in PyTorch](https://github.com/charlesq34/pointnet)
