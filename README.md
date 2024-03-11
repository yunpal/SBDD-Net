# SBDD-Net: Domain-Invariant 3D Structural Block and Double Descriptor for Place Recognition

## Introduction


### Pre-Requisites
* PyTorch 0.4.0
* tensorboardX

## Dataset preparation
Download the Used, Oxford, and TUM dataset, and then run the following scripts to prepare the data, for example,

```
cd generating_queries/


# Generate training tuples for the USyd Dataset
python generate_training_tuples_usyd.py


# Generate evaluation tuples for the USyd Dataset
python generate_test_sets_usyd.py
```

### Generate pickle files
Download the Oxford dataset, and then run the following scripts
```
cd generating_queries/

# For training tuples in our network
python generate_training_tuples_baseline.py


# For network evaluation
python generate_test_sets.py
```

### Train

```
python train.py 
```

### Evaluate
```
python evaluate.py 
```

## References:
The code is in built on [PointnetVlad](https://github.com/mikacuy/pointnetvlad) and [3D_GCN](https://github.com/zhihao-lin/3dgcn).

1. Uy, M.A. and Lee, G.H., 2018. Pointnetvlad: Deep point cloud based retrieval for large-scale place recognition. In *Proceedings of the IEEE conference on computer vision and pattern recognition* (pp. 4470-4479).
2. Lin, Z.-H., Huang, S.-Y., & Wang, Y.-C. F., 2020. Convolution in the cloud: Learning deformable kernels in 3D graph convolution networks for point cloud analysis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 1800-1809).

