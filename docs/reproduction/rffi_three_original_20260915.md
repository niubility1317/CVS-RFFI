# 三篇有标签域适应论文：原始源码与复现状态

日期：2026-09-15。用户确定顺序：先按原论文数据复现，再接CVS。仅做同组设备的有标签域适应；不开展新设备注册，不将RadioNet的UDA主方法算入本次有标签复现。

当前状态：SOURCE_DOWNLOAD_VERIFIED / COMPATIBILITY_VERIFIED / ORIGINAL_DATA_RUNS_INCOMPLETE。

## 文件与来源

统一下载目录：`E:/type10-7/external_sources/rffi_labeled_da_20260915/`。

| 论文 | 源码与版本 | PDF |
|---|---|---|
| Deep Learning Model Portability for Domain-Agnostic Device Fingerprinting，IEEE Access2023，10.1109/ACCESS.2023.3305257 | 未找到作者代码；归档本项目已有重建源码，commit a53c43e9，`Tweak_project_reconstruction/`；绝不标为作者代码 | `Tweak_Access2023.pdf`，复用本机已存期刊全文 |
| POSTER: Robust Deep-learning-based Radio Fingerprinting with Fine-Tuning，WiSec2021，10.1145/3448300.3468253 | https://github.com/SmartHomePrivacyProject/RadioFingerprinting ，commit 23d1fd4aef2f1dd254c5596680ae1da66f416916 | `POSTER_WiSec2021.pdf`，https://cse.unl.edu/~nghose/pubs/conf/papers/LI_Wisec21.pdf |
| RadioNet: Robust Deep-Learning Based Radio Fingerprinting，CNS2022，10.1109/CNS56114.2022.9947255 | https://github.com/UCdasec/RadioNet ，commit 64f4b0a3dfb48b4a389e4d177e9ffa35010e17f5 | `RadioNet_CNS2022.pdf`，https://cse.unl.edu/~nghose/pubs/conf/papers/LI_CNS22.pdf |

PDF均经pypdf解析，附同名UTF-8提取文本。原作者仓库保留完整Git历史，不修改、不把其代码混入项目发布；项目提交包含兼容加载器及来源记录。

## 作者源码兼容性与方法差异

两篇Li论文依赖旧Keras/TensorFlow。RadioNet的finetune.py导入`rf_models`和`augData`，但前者在models目录，后者不存在；直接运行存在阻塞。

`tools/rffi_three_author_smoke.py`通过AST读取作者模型定义，仅执行选定模型函数，避开无关缺失依赖。POSTER选Homegrown，RadioNet选论文有监督对比使用的DF。隔离环境`external_sources/rffi_labeled_da_20260915/venv`安装Keras3.12.4，使用PyTorch后端；它是兼容执行版本，不是原论文TensorFlow环境的逐位数值复现。

必须保留两个不同模式：

- `paper_last_layer`：沿用源模型全部权重，只训练最后分类层，对应论文文字。
- `author_last_three`：重建模型，仅复制并冻结`layers[:-3]`，其余重新初始化，对应下载的两个finetune.py的复制范围。POSTER中最后三层均为Dense；RadioNet的DF最后三层包含最后池化、Flatten与Dense。

编译移到冻结设置之后，避免当前Keras训练状态不一致。POSTER采用明确channels-last，修正原函数中Conv和GlobalAveragePooling默认布局不一致的问题。两种模式都不应静默互相替代；正式结果必须按模式命名。未验证ResNet、complex网络、ADA或triplet分支。

验证：四种组合均执行源训练一步、目标训练一步，检查loss/prediction有限、冻结权重逐项不变、可训练参数实际更新、逐样本与batch预测一致、保存/重载预测一致。结果见`compatibility_v1/validation.json`。输入为合成数据，不提供论文性能证据，不可加载这些smoke checkpoint进入正式训练。

复跑兼容验证（新out目录，禁止覆盖）：

```text
E:/type10-7/external_sources/rffi_labeled_da_20260915/venv/Scripts/python.exe tools/rffi_three_author_smoke.py --sources E:/type10-7/external_sources/rffi_labeled_da_20260915 --out E:/type10-7/external_sources/rffi_labeled_da_20260915/compatibility_v2
```

## 原始数据的当前阻塞

- RadioNet README的2026年2月SharePoint链接HTTP200，但正文明确为`Sorry, the link has expired.`。不能把200当作下载成功。
- POSTER README的2026年8月SharePoint链接访问返回HTTP401。本次无法匿名下载，不推断用户浏览器登录后也不可访问。
- 两份作者源码及PDF成功取得，原始HackRF/NEU训练数据未取得。未发送邮件或联系作者。
- Tweak本机原始LoRa配置数据已有100个DAT文件，共16,000,000,000字节；日期数据只有部分，不能视为全部接收机/日期/配置矩阵完整。已有下载器位于`paper_reproduction/gaskin_tweak_2023/data/download_official_tweak_lora_iq.py`。

## Tweak已有实验的核对

根目录旧`paper_reproduction/gaskin_tweak_2023`仍有陈旧未平方L2实现，不能直接复用。正确已有工作树为`github_publish/CVS-RFFI-repo/.worktrees/two-paper-rffi-reproduction-20260827`，commit a53c43e9。本次从此版本归档，36项聚焦测试通过。

V7已有官方LoRa Config2训练、100epoch和20行校准/测试结果。本次重新运行独立scorer，20行781,200次判定全部匹配，结果保存`external_sources/rffi_labeled_da_20260915/tweak_v7_rescore.json`。这是旧实验读回验证，不是本次重新训练，也不是全部论文矩阵复现。

V7原报告：`automation_reports/CV-SincNet/tweak_config2_portability_20260910_v7/report.md`。数值复现仍未达到原论文；不以重复启动相同训练代替解决方法不确定性。V8是margin-violating挖掘诊断，明确不是严格论文复现，不拿其较高分替代V7。

## 后续执行条件

1. 取得两篇Li论文可访问的原始数据链接或本地目录，验证是原论文对应日期、设备和信号处理位置。
2. 根据PDF与源码差异固定训练/微调模式、切片和预算，再开展完整原始数据训练、source-only与有标签适应同row对照，保存预测后独立评分。
3. Tweak继续补齐原始数据场景和方法缺失细节；现有结果只支持配置变化子实验且尚未达到原论文数值。
4. 原始数据阶段完成后再接CVS；当前没有CVS新训练、远端发布或自动运行任务。
