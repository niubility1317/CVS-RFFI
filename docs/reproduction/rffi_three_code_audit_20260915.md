# 三篇复现代码完整性与可运行性检查

日期：2026-09-15。结论：下载完整性VERIFIED；两份作者原始入口在当前隔离环境运行FAILED；选定模型的兼容训练仅通过合成数据验证。Tweak继续使用此前修正后的项目复现，不重新实现。

## 下载与启动检查

| 项目 | 下载完整性 | 当前实际运行边界 |
|---|---|---|
| POSTER / RadioFingerprinting | commit `23d1fd4aef2f1dd254c5596680ae1da66f416916`；112个跟踪文件、53个Python文件，无跟踪文件缺失，Git对象检查通过 | `fine-tuning/main.py`和`fine-tuning/finetune.py`启动均失败：Keras3不再提供`keras.utils.np_utils` |
| RadioNet | commit `64f4b0a3dfb48b4a389e4d177e9ffa35010e17f5`；50个跟踪文件、28个Python文件，无跟踪文件缺失，Git对象检查通过 | `model_training.py`、`finetune.py`、`model_test.py`启动均失败：当前环境未安装TensorFlow |
| Tweak / Model Portability | 复用既有修正版本`a53c43e983cce82bf2a259a0994403c3b7cf4eec`，不是作者源码 | `official_lora_experiment --help`本轮退出0；前轮36项聚焦测试通过，既有V7预测独立复核通过 |

完整性只表示相对于固定作者仓库版本下载完整，不表示作者发布了运行所需的全部依赖、数据和模块。两个下载仓库均无已跟踪源码改动；RadioNet存在运行生成的未跟踪`__pycache__`。

审计脚本：`tools/audit_rffi_author_downloads.py`。原始结果：同目录`rffi_three_code_audit_20260915.json`。启动探针使用`--help`，只能判定能否通过初始化导入，不能证明训练完成。

## 额外源码缺口

- POSTER的`get_simu_data.py`位于`rf/`，微调目录直接导入需要处理模块路径；它不是下载遗漏。`complex_rf/cxnn/complexnn/fft.py:143`仍为Python2的print语法，该分支Python3编译失败，但不据此认定选定Homegrown模型函数不可用。
- RadioNet整个仓库未提供所引用的`augData.py`、`get_simu_data.py`；`rf_models.py`位于`models/`，其无条件导入还涉及`ADA/`的模块。只安装TensorFlow不足以解决全部问题。
- RadioNet存在现代NumPy已移除的`np.complex`和`np.int`，测试入口使用作者机器绝对路径。需要兼容修复和数据路径配置后再做端到端验证。
- 作者微调代码的复制范围、重新初始化范围与论文“只微调最后层”的表述存在差异；不能静默统一。现有兼容脚本保留`paper_last_layer`和`author_last_three`两种模式。

## 已通过的兼容验证不等于原始流程通过

`tools/rffi_three_author_smoke.py`选择POSTER的Homegrown和RadioNet的DF函数，以Keras3/PyTorch后端完成四种组合的合成数据源训练一步、目标微调一步、冻结参数检查及保存重载一致性检查。证据位于：

`E:/type10-7/external_sources/rffi_labeled_da_20260915/compatibility_v1/validation.json`

这不覆盖作者的数据加载、完整训练循环、其他网络或原始数据评测；不能称为“两篇论文已完整运行”。两篇原始数据仍未取得：已检查的RadioNet分享链接过期，POSTER匿名访问401。

## Tweak固定沿用已有实现

执行目录固定为：

`E:/type10-7/github_publish/CVS-RFFI-repo/.worktrees/two-paper-rffi-reproduction-20260827`

本轮核实该工作树为上述commit且工作区干净。使用`D:/conda-envs/ssr-gpu/python.exe -m paper_reproduction.gaskin_tweak_2023.official_lora_experiment`。保持既有`strict_hard`默认方案；V8的margin-violating诊断不能替代严格复现。根目录同名旧代码存在未平方L2实现，不作为运行入口。

此前V7使用官方LoRa配置数据，独立重评分核对20行、781200次判定全部一致；这证明已有产物可核对，不证明达到论文数值或覆盖全部原始场景。本轮未重启训练、未加载新checkpoint，也未接入CVS。

下一步仍按用户确定的顺序：保留已有Tweak实现；修复两份作者源码的运行缺口并取得原始数据后验证完整训练与评测；原论文阶段完成后再接CVS。
