#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project    : MatGGNN 
@File       : main.py
@IDE        : PyCharm 
@Author     : zychen@cnic.cn
@Date       : 2024/3/28 16:25 
@Description: 
"""
import torch
import yaml

from matdeeplearn import get_dataset, models


def load_model(config, data_path, model_path):
    dataset = get_dataset(data_path, 0, False)
    model = models.MEGNet(dataset, dim1=config['dim1'], dim2=config['dim2'], dim3=config['dim3'], pool=config['pool'],
                          post_fc_count=config['post_fc_count'], pre_fc_count=config['pre_fc_count'],
                          gc_count=config['gc_count'], gc_fc_count=config['gc_fc_count'],
                          dropout_rate=config['dropout_rate'])
    save = torch.load(model_path)
    model.load_state_dict(save['state_dict'])
    # TODO 多卡训练模型 在单卡上加载可能出错
    return model


if __name__ == '__main__':
    with open('config_search.yaml', "r") as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)
    print(config)
    model_name = config['Predict']['model']
    model_path = config['Predict']['model_path']
    data_path = config['Predict']['data_path']
    model_config = config['Models'][model_name]
    # 加载模型
    model = load_model(model_config, data_path, model_path)

    # 加载 数据

