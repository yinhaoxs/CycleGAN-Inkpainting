# 基于CycleGAN模型的中国水墨画实现效果
- 步骤
```python
1:数据准备与模型训练
2:模型风格转换推理流程
```

## 一、数据准备与模型训练
```python
# 模型的训练详情：
下载水墨山水画公开数据集, 训练参考：https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
```

## 二、模型风格转换推理流程
```python
# 容器中创建python 虚拟环境
/usr/local/bin/python3.6 -m venv venv
# 激活虚拟环境
source venv/bin/activate
# 切换到项目主目录执行
pip install -r requirements.txt
# 风格转换过程
python demo.py
```

