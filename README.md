# RealPoint3D: Generating 3D Point Clouds from a Single Image of Complex Scenarios

#### [RealPoint3D: Generating 3D Point Clouds from a Single Image of Complex Scenarios](https://arxiv.org/abs/1809.02743) 

#### Introduction

------

We proposed to integrate the prior 3D shape knowledge into the network to guide the 3D generation. By taking additional 3D information, the proposed network can handle the 3D object generation from a single real image captured from any viewpoint and complex background. 

The work is based on [PointNet++](https://github.com/charlesq34/pointnet2) and [PSGN](https://github.com/fanhqme/PointSetGeneration). Some codes may be missing since the working place is changed to cause data loss.

#### Pre-requisites

------

You need to compile the TF operators. And other requisites are referred  to [PointNet++](https://github.com/charlesq34/pointnet2) and [PSGN](https://github.com/fanhqme/PointSetGeneration).

#### Training

------

To train our network, run the following command:

```
python train.py
```

#### Testing

```
python test.py
```

#### Citation

------

```
@article{xia2019realpoint3d,
  title={RealPoint3D: Generating 3D point clouds from a single image of complex scenarios},
  author={Xia, Yan and Wang, Cheng and Xu, Yusheng and Zang, Yu and Liu, Weiquan and Li, Jonathan and Stilla, Uwe},
  journal={Remote Sensing},
  volume={11},
  number={22},
  pages={2644},
  year={2019},
  publisher={Multidisciplinary Digital Publishing Institute}
}
```

