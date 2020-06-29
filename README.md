HetETA: Heterogeneous Information Network Embedding for Estimating Time of Arrival
---------------

This is basic implementation of our KDD'20 Applied Data Science Track (Oral) paper:

Huiting Hong, Yucheng Lin, Xiaoqing Yang, Zang Li, Kung Fu, Zheng Wang, Xiaohu Qie, Jieping Ye. 2020. HetETA: Heterogeneous Information Network Embedding for Estimating Time of Arrival.

The source code is based on [STGCN](https://github.com/VeritasYin/STGCN_IJCAI-18)

HetETA framework            
:-------------------------:
![](https://github.com/didi/heteta/raw/master/figs/framework.png)


Dependencies
------------
The script has been tested running under Python 2.7.5, with the following packages installed (along with their dependencies):

- `argparse==1.1`
- `numpy==1.16.5`
- `scipy==1.2.2`
- `networkx==2.2`
- `tensorflow-gpu==1.13.1`
- `yaml==5.1.2`



Overview
--------------
Here we provide the implementation of HetETA and a toy dataset.

The folder is organised as follows:
- `dataset/` contains:
    - `make_sample.py` randomly generates the `toy_sample` dataset to help readers to figure out the input format;
    - `toy_sample/` contains:
        * `adj_gap_top5.mat` is the vehicle-trajectories based network;
        * `adj.mat` is the multi-relational road network;
        * `link_info.npz` is the static attributes of each road segment;
        * `dynamic_fes.npz` is the dynamic feature (speed) of each road segment over time periods;
        * `eta_label.npz` contains the time it takes for a vehicle to travel through a path starting form period t.
- `codes/` contains:
    - `data/`:
        * `model/` is used to save the trained model;
        * `config_*.yaml` configures the path and paramenter settings.
    - `model/` contains the implementation of the HetETA network;
    - `utils/` contains some tools for loading dataset;
    - `train.py` is used to execute a full training run on the dataset.


How to run
---------------

```shell
cd codes
python -u train.py --config data/config_HetETA_toy.yaml --model_dir data/model/HetETA_toy --dataset_dir ../dataset/toy_sample >> multi-HetETA_toy.log
```

Please note that the `toy_sample` dataset is not a real dataset, which is only used to provide examples of data formats, not to train models.

License
----------
Didi Chuxing, Beijing, China.
