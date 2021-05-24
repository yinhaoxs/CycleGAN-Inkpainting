# -*- coding: utf-8 -*-
"""
# @Date: 2020-10-10 10:01
# @Author: yinhao
# @Email: yinhao_x@163.com
# @Github: http://github.com/yinhaoxs
# @Software: PyCharm
# @File: test.py
# Copyright @ 2020 yinhao. All rights reserved.
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.util import tensor2im
import time
import cv2


def test():
    opt = TestOptions().parse()  # get test options
    ### 准备数据集
    dataset = create_dataset(opt)
    ### 加载模型
    model = create_model(opt)
    # regular setup: load and print networks; create schedulers
    model.setup(opt)
    # test with eval mode.
    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        h, w, _ = data['A_shape']
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        fake_im = tensor2im(visuals['fake'])
        img_path = model.get_image_paths()     # get image paths
        cv2.imwrite(os.path.join("results/", "{}.jpg".format(img_path[0].split("/")[1].split(".")[0])), cv2.resize(fake_im, (w[0], h[0])))


if __name__ == '__main__':
    t = time.time()
    test()
    print("平均耗时为:{}".format((time.time()-t)/10))
