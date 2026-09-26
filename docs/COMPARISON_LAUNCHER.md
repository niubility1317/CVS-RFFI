# 对比方法统一启动入口

入口为`tools/comparison.py`。本机使用`C:/Users/lh594/.conda/envs/ssr-gpu/python.exe -X utf8`；N607使用实验登记指定的CVS-RFFI解释器。代码在独立Git检出`E:/type10-7/code/snapshots/cvs_practical_baselines_20260927_wt`中维护。

## 默认行为

- 方法：CVCNN、RIEI-FD、DRIFT各自的CE/PL路线，以及POSTER、RadioNet DF，共8条路线。
- 每条路线使用相同的5个模型种子：392005、2026092701、2026092702、2026092703、2026092704。历史优化种子392005与新种子分开报告。
- 训练200轮，batch size128；最终只用`last.pt`，禁止因验证或测试成绩更好而替换为其他轮次。
- 源域L/U/V和物理ID沿用已核实的数据契约。训练期不加载目标域，不进行目标域早停或选模。
- 训练完成后默认执行独立的clean和3种星地目标测试：`practical_high`、`practical_mid`、`practical_low_urban`，统一`residual/post_sync/noeq`。
- clean是无新星地扰动的Phase1目标域对照，不能作为Phase2方法输入。每个目标物理记录仅分配一个固定星地场景和一次固定信道观测。
- 所有行预测完成后再启动独立scorer，报告准确率、Macro-F1、混淆矩阵和逐接收机指标。测试结果不回流修改参数、选择seed或决定重跑。

## 常用命令

查看支持的方法：

```text
python -X utf8 tools/comparison.py list
```

生成两种方法、两个种子的实验配置与源域/最终评估登记：

```text
python -X utf8 tools/comparison.py prepare --run-id 20260928-poster-radio-r01 --methods poster-ce,radionet-ce --seeds 392005,2026092701 --gpus 6,7
```

可选项：`--epochs 200`、`--batch-size 128`、`--name 实验名称`、`--owner 唯一启动责任人`。科学数据契约保持固定，不通过宽松的任意参数覆盖绕过隔离规则。方法原始配置模板在`configs/baselines_practical_20260927/`。

完成现有项目的Git提交、push和启动前核查后打包：

```text
python -X utf8 tools/comparison.py package --spec automation_reports/CV-SincNet/20260928-poster-radio-r01/experiment.json --output release.tar.gz
```

在登记的N607 release目录中启动，一条命令同时启动源域队列及等待源模型的最终评估队列：

```text
python tools/comparison.py launch --spec automation_reports/CV-SincNet/20260928-poster-radio-r01/experiment.json --commit 实际发布提交
```

不重复启动已有run，不原地恢复失败输出。启动器保留输出碰撞检查。GPU容量仍需按现有N607流程核查：每GPU最多两个训练任务；最终评估默认只用CPU两线程。

## 给已运行的源域实验补最终测试

```text
python -X utf8 tools/comparison.py evaluate --source-spec automation_reports/CV-SincNet/源域run_id/experiment.json --run-id 新评估run_id
```

该命令只生成独立评估登记，不重启训练。按现有发布流程落地后运行：

```text
python tools/comparison_final_eval.py launch --spec automation_reports/CV-SincNet/新评估run_id/experiment.json --commit 实际发布提交
```

本次补充评估run为`20260927-phase1-baselines-final-clean-satellite-m5-r01`，依赖现有40行源域训练。已有Phase2队列继续使用其固定received IQ，互不改变数据与模型。

## 日志与复现

新启动的训练日志包含`[STARTUP]`和`[EFFECTIVE_ARGUMENTS]`：完整配置、解析后的全部默认参数、命令行、模型seed、数据契约路径、输出路径、PID、CWD、Python/PyTorch/CUDA版本、GPU映射及发布commit。对应文件为`startup.json`、`effective_arguments.json`和`resolved_config.json`。源数据角色和物理ID证据保存在`source_contract.json`，初始化继承记录在`initialization.json`。

当前已运行的旧release不在线修改，已有参数证据仍为其`resolved_config.json`和固定release源码；新日志格式从下一次使用新入口启动时生效。评估队列保存`launch.json`、`state.json`、`dispatcher.log`、`build.log`、逐模型日志、预测文件及独立`results.json`，不把正在运行写成完成。
