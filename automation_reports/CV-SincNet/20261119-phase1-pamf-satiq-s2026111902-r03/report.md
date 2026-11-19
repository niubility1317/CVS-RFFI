# SatIQ英国牛津站训练、瑞士图恩站测试

运行 `20261119-phase1-pamf-satiq-s2026111902-r03` 已在N607的物理GPU 6完成训练、全量预测和独立评分。冻结权重、预测、日志和评分均通过远端读回。两组预测各有30,000条、20列有限分数，评分文件晚于两份预测，读回重新按不透明ID核对了主要指标。[运行核验](launch_readback.json) · [完成核验](completion_readback.json) · [实验登记](experiment.json)

## 数据和调用

本轮从宿舍电脑的 `E:/SatIQ/npy_dataset` 读取真实Iridium下行包头。类别用 `(satellite_id, cell_id)` 表示，`cell_id` 对应波束编号。按英国全量30,000条候选记录的15秒UTC块划分训练和验证，块边界两侧各保留0.5秒隔离带。英国记录的解码置信度至少90。筛选英国训练至少30条、验证至少8条的类别，按训练数量降序取前20类，同数按二元组排序。32类达标，筛选结果为849条训练和269条验证记录。[类别表和数据契约](source_contract.json)

瑞士6个分片的30,000条均为测试。先前审计曾汇总两站重合，本次类别名单完全由英国数据产生。主测试评分覆盖瑞士369条英国已注册类消息，对应19个实际出现的类；(110, 39)无瑞士测试支持。其余29,631条的二元组在名单范围外。原瑞士 `test` 分片的5,000条中没有已注册类，全部369条来自原 `train`、`val` 分片，分别303条和66条。在本次重新划分中，三个瑞士分片角色都只用于测试。解码置信度至少90的已知瑞士子集为99条、14类。

转换后的264,076,962字节共6个文件，上传前后SHA-256均一致。服务器数据根为 `/home/szu2070436088/2510044040/CV-SincNet/data/20261119-pamf-satiq-oxford-thun-v1`。

|文件|字段和用途|
|---|---|
|`source/train.npz`、`source/val.npz`|`iq`、`y`、`ids`，用于英国拟合和验证|
|`source/contract.json`|站点、时间划分、截段规则、连续类别ID及原始(卫星，波束)|
|`target/query/iq.npy`、`target/query/ids.npy`|瑞士测试IQ及不透明ID，允许只读内存映射|
|`target/scorer/truth.npz`|供独立评分连接的瑞士原始ID、解码置信度和原分片角色|

IQ为 `float32[N,4,2,256]`，在25 MS/s下从每条包头的4,500、5,500、6,500、7,500点开始截取。每条包头的四段一起做RMS归一化，逐段提取后对logit取均值。保留原采样率和包级划分。包头全长11,000点的原始波形保存在宿舍电脑上。

```python
from pathlib import Path

import numpy as np
import torch

root = Path("/home/szu2070436088/2510044040/CV-SincNet/data/20261119-pamf-satiq-oxford-thun-v1")
with np.load(root / "source/train.npz", allow_pickle=False) as train:
    x_train = torch.utils.dlpack.from_dlpack(train["iq"])
    y_train = torch.utils.dlpack.from_dlpack(train["y"])
swiss_iq = np.load(root / "target/query/iq.npy", mmap_mode="r", allow_pickle=False)
```

## 对照和结果

两组用同一Sinc、时域、频域、功放分支和联合余弦头，从零开始。两组均移除Wi-Fi包头模板，PAMF-CE只增加三个自然顺序的SIG-TIME视图。模型种子、训练数据、原采样率输入、批次顺序、AdamW和预算相同。完成20轮、540次更新后固定权重。瑞士数据只在英国训练结束后用于推理。本轮为单种子真实下行数据实验，训练不叠加合成信道，不需要跨接收机配对或瑞士无标签适应。[SatIQ论文](https://www.cs.ox.ac.uk/files/14805/main.pdf)

|方法|英国验证准确率|瑞士已知类正确数|瑞士准确率|平衡准确率|Macro-F1|
|---|---:|---:|---:|---:|---:|
|Backbone-CE|8.92%|13/369|3.52%|4.76%|0.74%|
|SatIQ PAMF-CE|7.81%|14/369|3.79%|5.42%|0.62%|
|英国训练多数类|—|13/369|3.52%|5.26%|0.36%|

20类均匀随机的期望准确率为5%。两组对已知瑞士样本各只输出两个类别，Backbone-CE将353/369条预测为英国多数类，PAMF-CE为362/369。PAMF-CE多判对1条，Macro-F1低0.12个百分点。高解码置信度瑞士子集的准确率分别为3.03%、2.02%，Macro-F1分别为2.53%、1.44%。本轮未观察到SIG-TIME在该表示和数据规模下的跨站增益。英国验证准确率也低，现有证据同时包含源域学习困难和跨站差异，两站的不同采集日期、天线与前端也是共同变化因素。[完整评分和逐类支持](scores.json)

## 发布记录

首发 `r01` 在服务器PyTorch与NumPy的旧接口报错退出，`r02` 在Sinc的无query检查中退出。两次退出都发生在训练更新和瑞士预测前。替代版按原计划完成。源域与测试数据、矩阵和预算在替换时保持同一版本。[r01记录](../20261119-phase1-pamf-satiq-s2026111902-r01/report.md) · [r02记录](../20261119-phase1-pamf-satiq-s2026111902-r02/report.md)
