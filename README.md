# SBDD-Net: Domain-Invariant 3D Structural Block and Double Descriptor for Place Recognition

This repository is an official implementation for SBDD-Net

### Setting up Environment
```
conda create -n SBDD python=3.8 
conda activate SBDD 
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 -c pytorch
pip install scikit-learn tensorboardx==2.6.2.2
```

### Dataset preparation

Download benchmark_datasets.zip from [here](https://drive.google.com/drive/folders/1Wn1Lvvk0oAkwOUwR0R6apbrekdXAUg7D), and then run the following scripts.


### For training tuples in our network
```
cd generating_queries/
python generate_training_tuples_baseline.py
```
### For test tuples in our network
```
cd generating_queries/
python generate_test_sets.py
```

### Train

```
python train.py 
```

### Evaluate
The pre-trained model can be downloaded from [here](https://drive.google.com/drive/folders/1Wn1Lvvk0oAkwOUwR0R6apbrekdXAUg7D). After downloading, it should be placed in the `log` directory.
```
python evaluate.py 
```

## References:
The code is in built on [PointnetVlad](https://github.com/mikacuy/pointnetvlad) and [3D_GCN](https://github.com/zhihao-lin/3dgcn).

1. Uy, M.A. and Lee, G.H., 2018. Pointnetvlad: Deep point cloud based retrieval for large-scale place recognition. In *Proceedings of the IEEE conference on computer vision and pattern recognition* (pp. 4470-4479).
2. Lin, Z.-H., Huang, S.-Y., & Wang, Y.-C. F., 2020. Convolution in the cloud: Learning deformable kernels in 3D graph convolution networks for point cloud analysis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 1800-1809).

