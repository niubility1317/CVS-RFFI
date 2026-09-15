# POSTER与RadioNet：保留核心的服务器适配

2026-09-15。完成状态：本地兼容链路VERIFIED；未在N607部署或验证，未开展原始数据正式训练。

## 范围与核心边界

本次只适配已选定的POSTER Homegrown和RadioNet DF有标签微调路线。作者仓库下载原件保持不变，生成独立副本；项目Git发布适配生成器和接口，不重新分发作者完整源码。Tweak不变，继续沿用先前修正版。

两个模型文件内**全部函数AST与原版一致**，包括层数、卷积参数、激活、池化、Dropout和分类头；没有重写网络、损失或优化算法。微调保持作者复制`layers[:-3]`并冻结、其余层重新初始化的行为。只在冻结后增加同配置compile，使Keras3实际执行作者设置的冻结状态。

论文核对依据为本地全文`POSTER_WiSec2021.txt:77-89,239-274`和`RadioNet_CNS2022.txt:864-865`：均描述微调最后层。POSTER源码实际微调三个Dense层，RadioNet DF实际可训练参数只在最后分类层。**本次保留源码，不宣称100%论文方法复现，也不自行将POSTER改成只训练最后层。**这是已存在的作者代码/论文差异，不是本次引入的算法变更。

## 必要适配

| 修改 | 边界 |
|---|---|
| 旧`np_utils`改为Keras utils；`tensorflow.keras`改为统一Keras入口 | Keras3使用CVS环境已有PyTorch后端；不改系统TensorFlow或CUDA |
| `np.int/float/complex`改为等价Python类型；`val_acc`改为`val_accuracy`；优化器名称规范化 | 旧API兼容 |
| 添加显式模块搜索路径；去除RadioNet所选路线未使用的缺失模块导入 | 不伪造augmentation/simulation实现；复杂网络和ADA路线不在本接口支持范围 |
| 冻结后以原损失、Adam和原指标重编译 | 验证冻结权重不变，不修改作者冻结范围 |
| POSTER设置全局channels-first并把外部`N,L,2`转置为`N,2,L` | 统一Conv和GAP轴约定，模型函数不改；RadioNet保持channels-last |
| 独立输出目录、`.keras`文件、明确输入路径和阶段CLI | 不依赖作者服务器硬编码目录；输出存在即拒绝覆盖 |

注意：旧脚本的硬编码`__main__`不作为服务器启动入口；统一使用`tools/li_author_server.py`调用其原有CNN方法。原始数据加载器保留，但本接口接收外部已准备的IQ数组，不自行推断原论文切片、随机划分、归一化或CVS数据角色。

## 本机产物与验证

- 原件：`E:/type10-7/external_sources/rffi_labeled_da_20260915/RadioFingerprinting`及`RadioNet`。
- 适配副本：`E:/type10-7/external_sources/rffi_labeled_da_20260915/server_compat_v1/`，含完整副本、逐行`compat.patch`和`manifest.json`。
- 实际测试：`E:/type10-7/external_sources/rffi_labeled_da_20260915/server_test_v1/`，含6个阶段日志、源/目标权重、预测和检查结果。
- 每篇均调用原作者CNN训练1epoch、微调1epoch，再通过独立CLI加载保存模型进行预测，全部退出0。
- 对两篇逐项比较作者冻结前缀：权重完全一致；重载后逐样本和batch预测一致，概率有限且和为1。
- 测试30条合成IQ、5类，不是论文精度实验。测试权重禁止用于正式训练。首次权重检查将无参数InputLayer序列化后的trainable标志也当作冻结失败，现已改为检查有参数层的冻结标志以及全部实际权重；训练代码未因此更改。

## 服务器运行方式

在服务器现有CVS PyTorch环境之上建立隔离venv，不升级主环境：

```text
python -m venv --system-site-packages /absolute/path/li-author-venv
/absolute/path/li-author-venv/bin/python -m pip install -r requirements/li_author_compat.txt
python tools/prepare_li_author_compat.py --sources /absolute/path/author-downloads --output /absolute/path/compat-v1
```

依赖安装和N607实测尚未执行。默认Keras后端为torch，GPU选择沿用服务器`CUDA_VISIBLE_DEVICES`。TensorFlow可通过参数指定，但尚未验证，不声称与原论文旧框架逐位一致。

输入NPZ必须包含`x`（float IQ，`N,L,2`）及`classes`（有序字符串物理类ID）；train/tune另有整数`y`，取值与classes顺序对应。predict文件不得包含`y`。至少5类对应作者原top-5指标。相同类集、输入形状和论文标识必须与checkpoint旁的JSON一致，否则拒绝加载。

```text
python tools/li_author_server.py --compat-root /absolute/path/compat-v1 --paper poster --stage train --data /absolute/path/source.npz --output /absolute/path/source-run
python tools/li_author_server.py --compat-root /absolute/path/compat-v1 --paper poster --stage tune --data /absolute/path/target-support.npz --checkpoint /absolute/path/source-run/final.keras --output /absolute/path/tune-run
python tools/li_author_server.py --compat-root /absolute/path/compat-v1 --paper poster --stage predict --data /absolute/path/query-without-truth.npz --checkpoint /absolute/path/tune-run/final.keras --output /absolute/path/predict-run
```

RadioNet替换为`--paper radionet`。模型选择固定为DF，未接入其无标签ADA方法。默认训练预算保持原CNN类：POSTER源50epoch/微调10epoch/batch256，RadioNet源100epoch/微调30epoch/batch128；可显式传`--epochs`和`--batch-size`，不把源码默认预算冒称论文预算。

作者train方法末尾会在传入数组上打印评估信息；本接口传入源训练数组，该数字只是训练集诊断，**不是独立测试精度**。正式评测必须使用独立predict阶段，输出概率按输入顺序保存，再由外部scorer连接truth。源模型取作者保存的最佳验证checkpoint；微调返回作者最后epoch模型，未改成最佳checkpoint。

## 仍需满足的正式实验条件

当前仅完成运行兼容和通用数组接口；未实现CVS capsule物理ID/完整checkpoint继承审计或正式scorer接线。JSON类映射检查不能替代项目数据契约。接CVS时继续执行现有协议；原论文数据仍须先取得并固定切片、support/query划分和论文/源码差异的实验命名。不得把本接口的合成验证或训练诊断宣称正式复现完成。
