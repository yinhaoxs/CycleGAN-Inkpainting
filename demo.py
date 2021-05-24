# -*- coding: utf-8 -*-
"""
# @Date: 2020-10-09 19:04
# @Author: yinhao
# @Email: yinhao_x@163.com
# @Github: http://github.com/yinhaoxs
# @Software: PyCharm
# @File: demo.py
# Copyright @ 2020 yinhao. All rights reserved.
"""
from options.test_options import TestOptions
from models import create_model
from PIL import Image
import torchvision.transforms as transforms
from util import util
from util.util import tensor2im
import os, time

# 设置参数
opt = TestOptions().parse()  # get test options

def predict(A_path):
    # 读取数据
    A_img = Image.open(A_path).convert('RGB')
    # 数据增强
    transform_list = []
    transform_list += [transforms.ToTensor()]
    transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    transform_A = transforms.Compose(transform_list)
    A = transform_A(A_img)
    out = {'A': A.unsqueeze(0), 'A_paths': A_path}

    # 模型推理
    model = create_model(opt)
    # regular setup: load and print networks; create schedulers
    model.setup(opt)
    # test with eval mode.
    if opt.eval:
        model.eval()

    model.set_input(out)  # unpack data from data loader
    model.test()           # run inference
    visuals = model.get_current_visuals()  # get image results
    fake_im = tensor2im(visuals['fake'])

    return fake_im


if __name__ == "__main__":
    img_dir = 'datasets/'
    out_dir = 'results/horse/'
    for img_name in os.listdir(img_dir):
        t = time.time()
        img_path = os.path.join(img_dir+os.sep, img_name)
        output_path = os.path.join(out_dir+os.sep, img_name)
        fake_im = predict(img_path)
        util.save_image(fake_im, output_path)
        print("图片:{}, 转换耗时:{}".format(img_path, time.time()-t)+'\n')