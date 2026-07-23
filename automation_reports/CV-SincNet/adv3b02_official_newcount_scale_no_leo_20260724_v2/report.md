# ADV3B02官方对比方法新类无LEO诊断v2

- 实验ID：`adv3b02_official_newcount_scale_no_leo_20260724_v2`
- 日期：2026-07-24
- 操作方：主代理；N607唯一运行所有者为`no_leo_n607_release`
- 当前状态：`LOCAL_VERIFIED / INDEPENDENT_REVIEW_APPROVED`
- 性质：`DIAGNOSTIC_NEW_CLASS_NO_LEO_NON_FORMAL`

## 目标与边界

与`adv3b02_official_newcount_scale_20260724_v7`的LEO参考组进行matched对照。
复用相同receiver、seed、K、新类集合、physical IDs、support/query划分、ADV3B02 checkpoint、
base26/base31及官方CSIL/MoPC-HR方法，只改变一个因素：

- `target_old`继续使用v7缓存中的unmodified received IQ，字节不变；
- `target_new`按相同physical record从`Dataset_WigSig/ManyTx.pkl`恢复LEO overlay前原始接收IQ；
- 不允许使用`ManySig.pkl`，不允许改变physical IDs或重采样。

该实验不符合“新类必须叠加LEO”的正式对比边界，因此只能用于解释LEO信道影响，
不得作为正式CVS/Stage2性能结果。

## 前次失败闭环

v7附属no-LEO smoke误把base训练数据`ManySig.pkl`传给要求ManyTx物理记录的诊断入口，
CSIL和MoPC-HR两个不同cell均在prediction前产生同一确定性异常
`ValueError: '1-16' is not in list`。0 prediction，full未启动，原路径封存且不重试。
本v2使用新的不可覆盖run ID和输出路径，只修正冻结命令参数为
`Dataset_WigSig/ManyTx.pkl`。

## 冻结矩阵

| 方法 | 新类数 | cells | 场景行 |
|---|---|---:|---:|
| CSIL | 1、3、20 | 300 | 900 |
| MoPC-HR | 1、3、5、10、25 | 500 | 1500 |
| 合计 | — | 800 | 2400 |

接收机、seed、K、方法参数及base状态均与v7 LEO参考组完全一致。
先运行双方法new1 smoke；smoke PASS后按8+8 shards执行完整矩阵。

## N607预注册

- CWD：`/home/szu2070436088/2510044040/CV-SincNet`
- Python：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`
- run root：`runs/adv3b02_official_newcount_scale_no_leo_20260724_v2`
- log root：`logs/adv3b02_official_newcount_scale_no_leo_20260724_v2`
- 输入计划：v7授权后的CSIL/MoPC计划，只读
- base26/base31：v7已完成且receipt PASS的官方base，只读
- 数据参数：`--manytx-pkl Dataset_WigSig/ManyTx.pkl`

本地`ssr-gpu`验证：诊断脚本`py_compile`通过；no-LEO诊断、官方ADV3B02、
CSIL和MoPC-HR四个相关测试文件共`31 passed`；`git diff --check`通过。
独立复审：`P0=0,P1=0,P2=0 / APPROVE`。

出现P0或两个不同row在prediction前同一确定性异常指纹时停止精确run-owned树；
绝不按性能停止。fresh-run retry未授权。终态必须核验800 cells、2400同row场景记录、
prediction/receipt覆盖及相同physical-ID replacement audit。

## 结果表占位

| 方法 | 新类数 | old_acc | seen_new_acc | H_old_new | forgetting | coverage | 结论 |
|---|---:|---:|---:|---:|---:|---|---|
| 待运行 | — | — | — | — | — | 0/800 | `NOT_ANALYZED` |
