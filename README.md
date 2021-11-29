# Person re-identification based on nonlinear optimization
## Prerequisites
* Python 3.8.8

* GPU Memory >= 6G

* Pytorch 1.8.1

* CUDA 11.1

* numpy

* sklearn 

* infomap

* faiss-gpu
## baseline
训练一个baseline做特征提取器
### Dataset
Download [Market1501 Dataset](http://www.liangzheng.com.cn/Project/project_reid.html) [[Google]](https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view) [[Baidu]](https://pan.baidu.com/s/1ntIi2Op)
data preparation
```bash
python prepare.py
```
### Train
Train a model by
```bash
python train.py --gpu_ids 0 --name ft_ResNet50 --train_all --batchsize 32  --data_dir your_data_path
```
`--gpu_ids` which gpu to run.

`--name` the name of model.

`--data_dir` the path of the training data.

`--train_all` using all images to train. 

`--batchsize` batch size.

`--erasing_p` random erasing probability.
#### Test
Use trained model to extract feature by
```bash
python test.py --gpu_ids 0 --name ft_ResNet50 --test_dir your_data_path  --batchsize 32 --which_epoch 59
```
`--gpu_ids` which gpu to run.

`--batchsize` batch size.

`--name` the dir name of trained model.

`--which_epoch` select the i-th model.

`--data_dir` the path of the testing data.
## infomap
将特征放在data下，对特征做聚类
### Cluster
```bash
python face-cluster-by-infomap
```
