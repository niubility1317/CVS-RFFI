# Phase1 本地 GPU 环境

适用于本机的 F／F＋跨 RX 路线研发。使用 Windows 原生 Python 和单 GPU；真实实验仍沿用实验登记、源域权限、默认分类头及 truth-last 流程。

## 创建与使用

选择已通过 CUDA 检查的 Python 3.10 作为基础解释器。执行 `tools/local_phase1_env.py --profile <本地 profile.json> create --prefix <新环境目录> --checkout <包含 F 路线的 Git 检出>`。新目录使用 `venv --system-site-packages`，复用基础环境的包；创建过程不安装或升级基础环境。环境依赖该基础解释器及其包，不能搬走后作为独立环境使用。

执行 `tools/local_phase1_env.py --profile <profile.json> check --output <新结果.json>`，检查关键依赖导入、NumPy／CUDA 转换，以及实际 F＋跨 RX 网络的前向、反向、原物理单位 Sinc 优化器更新和默认头联合更新。仅使用合成 IQ，不读取数据、checkpoint 或目标域。输出显存和训练步耗时，不能替代真实数据训练或完整 epoch 的速度测试。

执行 `tools/local_phase1_env.py --profile <profile.json> run -- <Python 参数>`，在已选检出及环境下运行脚本或模块。启动入口设置 GPU 0、2 个数值计算线程、UTF-8 和项目导入路径；不修改系统 PATH。它不会自动改写脚本的数据路径、batch size、DataLoader 或数学精度设置。

## 当前资源设置

单 GPU 串行运行；固定对比的 batch size 保持 128，DataLoader 默认 0 个子进程。采用已登记 F 的完整 FP32 策略，关闭 TF32；不默认启用 AMP、torch.compile 或改动训练预算。Windows 上新增 DataLoader 子进程前须验证 spawn 和内存占用。任何 batch／数值设置的变化必须作为新实验配置披露。

本地环境采用 Python 3.10、PyTorch 2.10.0＋cu128、torchvision 0.25.0＋cu128、NumPy 2.2.6、SciPy 1.15.3、h5py 3.15.1、pandas 2.3.3、scikit-learn 1.7.2、matplotlib 3.10.8；实际完整依赖版本以本地检查输出为准。CUDA 运行库由 PyTorch 包提供；本轮没有编译自定义扩展，不需要额外安装 CUDA Toolkit。

重建独立基础环境时，可在 Python 3.10 环境中安装 `configs/local_phase1_windows_requirements.txt`。已经通过检查的环境无需重复安装。

本机的路径、profile、依赖快照、GPU 检查输出放在 Git 忽略的 `local_artifacts/`，环境放在 `local_envs/`。不提交机器状态、数据或权重。

## 真实数据接入

旧实验入口包含 N607 的绝对数据和输出路径，`run` 仅负责解释器和工作目录，不能直接把远端冻结 launcher 当成本地训练命令。真实本地实验应使用新的 run/output root、合法源数据契约及角色索引，明确本地路径；不得把本地临时样本或重新划分冒充历史匹配比较。仅检查环境不会发起目标域访问。

数据目录保存 `raw/ManySig.pkl`、`source/source_contract.json`、`target/capsule/` 和独立的 `scoring/truth.json`。直接复用既有 VALIDATED_ONCE 数据及契约；取回过程记录字节长度和远端文件元数据，不重新建立数据哈希链。目标 capsule 的历史 satellite 数组包含 `practical_high`、`practical_mid`、`practical_low_urban` 三种场景，不能冒充当前正式实验的六场景完整评价。

用 `tools/local_phase1_data.py --dataset-root <数据目录> --output <新缓存目录>` 按原生源域加载器导出 L/U/V 内存映射缓存。它使用原有归一化和裁剪，逐角色核对全部物理 ID，不构造 target loader；U 不导出标签文件或 TX 元数据。真实训练可导入 `tools.local_phase1_data.SourceRoleDataset(cache, role)`；该接口返回 `(IQ, label, domain, metadata)`，U 的 label 固定为 −1。缓存不包含训练增强，训练端仍须执行登记配方中的增强与阶段日程。V 只作验证。

缓存的 domain 保持原生加载器的原始编号；训练时继续使用缓存 manifest 中的 `domain_label_map` 映射到连续域编号。它不改变 RX/day 分组。源训练进程只加载源缓存，不打开 `target/` 或 `scoring/`。

本地与 N607 的 PyTorch 版本可能不同。涉及复数梯度、临界点、数值精度及序列化时，分别验证对应行为；不能仅凭本地检查通过断言远端兼容，也不能预先承诺本地比 N607 更快。
