# MatGGNN

Material Geometric Generate Nerual Network

## 一. 环境配置 

1. python

​		conda create -n MatGGNN python=3.10

​		conda activate MatGGNN

2. 安装cuda pytorch
   检查电脑支持的最高的cuda版本

   查看驱动和版本的对应关系：https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html

   下载cuda：https://developer.nvidia.com/cuda-toolkit-archive

   ```bash
   conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia
   ```
 
3. 安装torch_geometric

   安装顺序：1.torch-scatter 2.torch-sparse 3.torch-cluster 4.torch-spline-conv 5.torch-geometric

   1. 官网安装不成功，可以使用离线安装：https://data.pyg.org/whl/
   2. 在线安装：

   ```
   pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
   pip install torch_geometric
   ```



## 二 启动训练

### cmd命令

```bash
python .\deeplearn_main.py --data_path .\data\oqmd_data\oqmd_data\ --job_name train_job_500 --run_mode Training --model CGCNN_demo --save_model True --model_path my_trained_model.pth
```

