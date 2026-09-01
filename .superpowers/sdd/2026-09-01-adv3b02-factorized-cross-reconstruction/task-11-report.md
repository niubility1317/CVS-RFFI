# ADV3B02-FCR Task11报告：checkpoint、诊断与正式入口闭合

## Status

`LOCAL_VERIFIED / P0P1_REVIEW_PENDING`。Task11的checkpoint bundle、训练外诊断、R0-R8正式入口和真实checkpoint无query smoke已完成本地闭合。此状态不表示N607训练、消融结果、clean/三种LEO_WEAK最终数值评测或晋级结论已经完成；FCR-24仍保持`pending`。唯一独立P0/P1审查由root在本任务提交后执行，本任务未新增reviewer、gate、hash、seal或签名。

## 设计追溯

| ID | 来源 | 要求 | 实现与证据 | 状态 |
|---|---|---|---|---|
| T11-01 | checkpoint contract | 仅FCR模型保存完整、可序列化`fcr_bundle`，权重不重复保存 | bundle含feature schema、完整`FCRConfig`、固定物理基、输入归一化、Fisher gate、nuisance schema和Task9 detach路由；checkpoint单测 | verified |
| T11-02 | checkpoint contract | 旧checkpoint无bundle时关闭态严格兼容 | `use_fcr=False`保存不产生bundle；旧关闭态严格往返；可信旧ADV3B02 metadata在PyTorch2.6可热启动 | verified |
| T11-03 | checkpoint contract | 新FCR模型严格恢复并复现单LEO`z_f_id`，不兼容bundle在state使用前拒绝 | unit round-trip、feature schema负测、真实checkpoint smoke中的fresh模型严格加载与精确复现 | verified |
| T11-04 | 设计第十七节 | 输出17个诊断键，probe只读detach artifact | 独立线性probe、距离、drop-f、移植、Gram/Fisher和资源诊断；训练外collector恢复模型原模式 | verified |
| T11-05 | diagnostics contract | 缺失严格pair/capability写`N/A`并附原因，禁止伪造0 | same-TX、drop-f和三项transplant负测均核对`N/A`+非空reason | verified |
| T11-06 | 设计第九节 | R0-R8显式、递进、不可跳级 | parser仅接受R0-R8；每row解析有效lambda和能力；R9负测拒绝 | verified |
| T11-07 | 项目协议4.3 | E200、三段LEO_WEAK、E80卫星CE及四最终评测显式可达 | launcher与Python CLI dry-run核对；真实四评测数值尚未产生 | implementation verified / result pending |
| T11-08 | launcher contract | caller提供唯一run ID/output root，拒绝覆盖 | 无默认run/output；已有正式output root在非dry-run时失败；每row使用独立子目录 | verified |
| T11-09 | focused verification | 三个新测试、FCR全集、baseline、compile、CLI dry-run和diff检查 | 完整显式文件集合79 passed；`py_compile`、R8 dry-run、`git diff --check`通过 | verified |
| T11-10 | real smoke | 真实ADV3B02 checkpoint+合法source完成forward/backward/save/load/单LEO | epoch194真实checkpoint、2条ManySig source IQ、`leo_clear_weak`、29个FCR有限梯度、严格bundle恢复、单LEO`z_f_id`精确一致 | verified |
| T11-11 | publication boundary | 只提交拥有文件并push、远端OID回读 | 以`feat:close-FCR-local-implementation`提交；最终OID只在提交后回执独立给出 | pending final publication readback |

## TDD证据

实现前先建立三组RED：

- checkpoint组首次为4 failed，原因是FCR bundle和严格加载helper尚不存在；补齐后4 passed。真实smoke又暴露PyTorch2.6默认`weights_only=True`无法读取可信旧checkpoint训练元数据，新增单条回归测试先失败，再将受信任的本地`--init_checkpoint`读取显式设为`weights_only=False`，该组最终5 passed。
- diagnostics组首次为2 failed，原因是模块不存在；训练外collector负测随后先以缺失函数失败。实现17键聚合、detach probe、collector和`N/A`原因后3 passed。
- launcher组首次为4 failed，原因是R0-R8配置解析、Python dry-run和launcher尚不存在；实现后4 passed。R9拒绝作为非法row负测保留。

所有功能修改均先有对应RED或由真实smoke给出可复现失败指纹，再做最小GREEN修复。

## Checkpoint闭合

`fcr_bundle`使用`cvs.phase1.adv3b02_fcr.bundle.v1`，候选feature schema为`ADV3B02:FCR:z_f_id:unit_l2:160:v1`。完整配置为输入长度、content stride/dim、TX state和channel/receiver/sync/gain维度及方差上下界；固定物理基标识为`fixed_response_basis:pa_conjugate_memory4:v1`；归一化标识为`adv3b02_input_iq:v1`；Fisher gate明确为确定性、0个可训练参数；nuisance顺序固定为channel/receiver/sync/gain；路由显式记录`content.s_hat.detach()`。

FCR参数只位于常规`model`state dict，bundle不复制tensor。候选恢复先逐项比较bundle，再严格加载完整state；不兼容feature schema在任何模型state使用前拒绝。关闭态checkpoint不产生bundle，旧模型仍可严格加载。正式训练结束的FCR best与primary checkpoint也改为bundle先验证、随后`strict=True`恢复；普通路径保持既有宽松恢复语义。

## 训练外诊断

诊断JSON固定包含：`zf_tx_probe`、`zf_domain_probe`、`zn_domain_probe`、`zn_tx_probe`、`zs_content_probe`、clean/LEO和same-TX距离、drop-f gap、三项transplant、Gram condition/effective rank、Fisher coverage、训练时间、峰值VRAM和latency。probe只消费detach后的独立train/eval mask，不复用训练分类器、optimizer或反向图。

collector只读Phase1 source validation loader，使用允许的LEO场景生成received view，置于`no_grad`并在结束后恢复模型模式。当前数据若不具备严格same-TX不同content、drop-f对照或matched different-TX common-preamble pair，对应值必须为`N/A`并携带具体reason；不得以0伪装能力或结果。

## R0-R8与正式入口

| row | 显式新增能力 |
|---|---|
| R0 | 所有FCR lambda为0 |
| R1 | self、eta |
| R2 | 加swap |
| R3 | 加shared |
| R4 | 加latent cycle |
| R5 | 加factor及basic-need诊断 |
| R6 | 加need及targeted transplant能力 |
| R7 | 加phys及physics-ordered decoder能力 |
| R8 | 完整三轴intervention入口 |

launcher冻结E200、E80卫星辅助CE、三段LEO_WEAK日程和`clean`/`leo_clear_weak`/`leo_low_elev_weak`/`leo_rain_weak`四最终评测入口。caller必须显式提供`RUN_ID`和`OUTPUT_ROOT`；正式运行前若root存在则拒绝覆盖；每个row独立checkpoint、日志、诊断和状态文件。launcher文本不含Phase2、query、truth或scorer路径。

Git Bash通道此前已判定`FAILED`，因此本任务没有本地执行Bash/WSL。launcher通过Python内容测试和`train.py --fcr_config_dry_run`验证；这不是对远端Bash实际执行的虚假声明。

## 真实checkpoint无query smoke

有界发现选定：

- checkpoint：`E:\type10-7\local_artifacts\adv3b02_ecrs_smoke\best_joint_safe_ssdg.pth`，epoch194，`ADV3B02_CORE90_SOFT_E200`，架构为6类、14域、M/`lite_d`/`no_dac`/`no_stats`、输入长度256。
- source：`E:\type10-7\local_artifacts\cvs_publication_inputs_20260713\ManySig.pkl`。只取该checkpoint原训练配置允许的day0/rx0/equalized1中TX0、TX1各1条真实IQ，shape为`[2,2,256]`。
- 输出：`E:\type10-7\local_artifacts\adv3b02_fcr_task11_smoke_20260902_r1\smoke_result.json`及对应round-trip checkpoint；目录创建前检查不存在，未覆盖旧产物。

smoke结果为PASS：195个旧模型tensor全部匹配加载，新增35个missing key全部属于`fcr.*`；真实clean IQ生成合法`leo_clear_weak`received view；forward/backward损失有限，29个FCR参数获得有限梯度；保存完整bundle后fresh同构模型严格恢复；单条LEO输入的`z_f_id`以`rtol=0,atol=0`精确一致。smoke没有读取Phase2、query、truth或scorer，也没有以随机tensor代替source样本。

## 完整验证

用户给定的通配符命令在Windows PowerShell中不会展开，首次得到`file or directory not found: code/tests/test_phase1_fcr_*.py`且收集0项；这属于命令路由失败，不是测试失败。随后用Windows原生显式展开完全相同的文件集合，结果为79 passed、7条既有`torch.cuda.amp.autocast`弃用warning，耗时17.54秒。

此外，六个拥有Python文件的`py_compile`通过；R8 CLI dry-run输出E200、E80、全8项lambda和四最终评测；`git diff --check`无空白错误，仅报告Windows预期的LF→CRLF提示。真实smoke第一次唯一失败指纹为PyTorch2.6旧checkpoint metadata加载错误，回归修复后同一资产、同一路径重跑PASS。

## 证据边界、自查与Task12接口

- Task9的`fingerprint_excitation=content.s_hat.detach()`保持不变，因此不存在`G_f→E_s`梯度；不得在后续报告声称该梯度存在。
- 当前真实smoke只证明本地实现、可信旧checkpoint热启动、FCR反传、bundle严格恢复和单LEO推理，不证明性能收益。
- 缺失严格Fingerprint Pair时，移植诊断为`N/A`+原因是正确结论，不是技术失败；不得用label-derived fallback补造。
- FCR-22、FCR-23、FCR-26依据精确实现测试更新为`verified`。FCR-23只验证R0-R8正式入口和同row artifact绑定，不声明已有消融结果。
- FCR-24保持`pending`：四评测入口已配置，但尚无真实训练完成后的clean/三LEO数值，不得标成实验完成。
- 普通ADV3B02关闭态不新增FCR参数/loss/bundle；launcher无query路径且拒绝覆盖；未触碰旧隐藏TX冲突测试。

## Git发布

本报告、FCR-22/23/26追踪行和Task11拥有代码/测试将以`feat:close-FCR-local-implementation`同一提交发布。提交后的本地HEAD、push结果和远端branch OID在任务完成回执中独立给出；不把未来OID回写同一提交，避免改变提交对象。
