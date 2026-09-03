# Task 6报告：FCR-V2模型与训练循环集成

日期：2026-09-03
工作树：`E:\type10-7\github_publish\CVS-RFFI-repo\.worktrees\adv3b02-fcr-20260901`
状态：`DONE_WITH_CONCERNS`
实现提交：`5d68753190e9fe9f83bdad6369bd7c657e36ee87`

## 结论

Task6已把Task1–5完成的FCR-V2模块接入`DualCVSincNetDisentangle`与Phase1训练循环。`--fcr_version v2`和`--fcr_matrix_row`支持`C1,C2,C3,S0-S4,M1-M6`；冻结的Task7 launcher继续通过`--fcr_ablation_row`兼容别名进入同一V2解析器。C1绕过FCR，C2/S0保留有标签identity CE但所有FCR辅助损失为0，并通过`forward_identity_only`绕过decoder。其余行直接使用Task4登记表和六阶段schedule，不在训练入口重写行语义。

本地正式验证共79项，覆盖Task6集成、Task1–5公共接口、历史FCR回归和Task7契约，结果为79项全部通过。真实checkpoint无query smoke使用仓库现有`best_test_model.pth`完成加载和V2 identity-only前向，但该文件是`seed=7、epoch=1、model_variant=lite_c`，不能证明冻结的`seed=392005、epoch=200、ADV3B02_CORE90_SOFT_E200`初始化身份。后者按主任务裁定留给Task8在N607固定路径核验，因此本任务状态为`DONE_WITH_CONCERNS`而不是把本地smoke冒充正式checkpoint证据。

## 需求追踪

| ID | 需求 | 落地点 | 状态 | 验证 |
|---|---|---|---|---|
| T6-1 | 接受V1/V2版本及14个V2矩阵行，保持V1 R0-R8行为 | `code/train.py`解析器和`resolve_fcr_training_options` | 完成 | 14行参数化测试、V1 R7回归、C1/C2/M6 dry-run |
| T6-2 | 组装V2 factor/physics并提供不运行decoder的`forward_identity_only` | `code/model_dual_cvsincnet.py` | 完成 | decoder替身设为抛错仍通过；真实checkpoint smoke报告`decoder=not_run_identity_only` |
| T6-3 | 成熟ADV3B02初始化、identity head复制及可审计加载报告 | `code/train.py`、`code/model_dual_cvsincnet.py` | 本地接口完成，正式身份延后 | 合成E200完整checkpoint测试通过；本地真实文件仅验证代码路径；N607正式身份由Task8核验 |
| T6-4 | V2互斥且完备的optimizer参数组 | `code/train.py` | 完成 | 早期主干、后层、identity head、identity projection、新FCR和other组覆盖全部可训练参数且无重复 |
| T6-5 | source metadata、三轴pairing及L_s/U_s权限 | `code/train.py` | 完成 | M6真实pair/objective反传通过；U_s只保留`self/shared_f/shared_s/response/eta`，有监督项和越权机制权重为0 |
| T6-6 | 每epoch记录签名/激活原因，defer target前写Task5 diagnostics | `code/train.py` | 完成 | 日志字段测试通过；V2诊断读取`final.pth`而非best checkpoint；Task5写出函数保留 |
| T6-7 | E200 final-only checkpoint bundle及Task7真实路由 | `code/train.py`、`code/cvsrffi/checkpoint.py` | 完成 | V2 bundle版本化和strict round-trip通过；Task7契约3项通过 |
| T6-8 | 相同seed/batch增强、source-only无query、只改拥有文件和必要接口 | 训练入口、测试和本报告 | 完成 | 继续使用现有按seed/epoch/batch生成的`BaselineOriginSatViewAugment`；metadata按物理ID/crop对齐；Git暂存清单未含无关artifact |

## 实现内容

### 模型与identity-only路径

- `ADV3B02FactorizedCrossReconstructionV2`只组合现有`ConservativeCanonicalizer`、`FCRV2FactorEncoder`和`IdentityInitializedPhysicsDecoder`，没有复制Task1–4算法。
- `DualCVSincNetDisentangle.forward_identity_only`返回正式`tx_logits/z_id`，内部只运行canonicalizer和factor encoder；`fcr_decode=None`，decoder不被调用。
- Task3将`z_f_id/z_f_dev`冻结为160维，但成熟ADV3变体的分类特征可为160维或192维。模型新增两个恒等初始化的矩形投影：骨干特征先投影到160维V2因子接口，`z_f_id`再投影回成熟identity head的输入维度。冻结launcher使用的`lite_d`仍是160维，仓库现有`lite_c`真实checkpoint也可完成同一路径smoke。
- V2 identity head由成熟ADV3分类头深复制；checkpoint warm-start后再次从已加载的成熟头同步，测试确认权重一致。

### 训练路由与损失边界

- C1：`use_fcr=False`，使用成熟legacy身份路径，仍执行冻结的E200/Meta-SSL/source-only约束。
- C2/S0：`fcr_identity_only_training=True`，有标签identity CE保留，FCR辅助集合为空，decoder不运行。
- C3、S1-S4、M1-M6：从`FCRV2Schedule.row_losses`取得行机制，按E1–20、E21–60、E61–100、E101–130、E131–160、E161–200阶段缩放。
- L_s使用identity/prototype/tail；U_s隐藏`tx_id`并只允许`self/shared_f/shared_s/response/eta`。`swap/cycle/need/transplant/physical/factor`在U_s下强制为0并记录`ROLE_NOT_PERMITTED:U_s`。
- 现有按`seed+epoch+batch`确定的星地增强入口保持不变；V2 metadata校验`physical_sample_id`与`crop_offset`和增强视图一致，再由Task3 pair builder生成三轴pair。
- 每epoch记录`fcr_version`、matrix row、execution signature、active losses和capability reasons。V2诊断在defer target返回前执行，并从刚保存的`final.pth`回载；训练进程不读取query或truth。

### 优化器与checkpoint

- 参数组及基准学习率为：早期主干`0→5e-6`、后层`1e-5→2e-5`、identity head`2e-5→5e-5`、identity projection`5e-5`、新FCR模块`2e-4`，其余参数沿用基础学习率。构造器以参数对象ID检查完备性和互斥性。
- `load_init_checkpoint_weights`返回实际/期望seed、epoch、candidate、加载/跳过/缺失计数以及`source_only=True/query_access=False`；空路径返回`not_requested`。
- V2最终checkpoint写入`cvs.phase1.adv3b02_fcr.bundle.v2`、V2 feature schema、正交响应基、物理nuisance结构、identity-only跳过decoder语义，并通过严格回载测试。V1 bundle导出分支保持原值。

## 必要接口修复

除Task6拥有文件外，本任务只修改了两个直接接口不一致点：

1. `code/cvsrffi/phase1_fcr_v2_schedule.py`：Task4原S0登记了`self`，与冻结要求“S0为identity-noop且辅助损失全0”冲突；S0登记改为空集。其他行未改。
2. `code/cvsrffi/checkpoint.py`：原保存器对所有FCR模型固定导出V1 bundle，导致V2`final.pth`被错误标为V1；新增V2导出/校验分支，V1分支和历史回载保持不变。

未修改Task7 launcher、prediction脚本或score脚本，未回退Task5 diagnostics。

## TDD与验证

### 红灯

- 初始集成测试：`21 failed,1 passed`，失败覆盖缺少V2参数、14行解析、identity-only、checkpoint报告、optimizer分组、V2 objective和defer诊断顺序。
- 真实checkpoint复现：`lite_c`输出`[B,192]`进入冻结`z_adv=[B,160]`接口时报`ValueError: z_adv must be [B,160]`。新增192维回归测试先失败，再用身份投影桥修复。
- V2 final bundle测试初次失败：实际为`cvs.phase1.adv3b02_fcr.bundle.v1`。新增版本分支后，V2 strict round-trip和10项历史checkpoint测试共同通过。

### 绿灯

| 验证 | 结果 |
|---|---|
| `test_phase1_fcr_v2_training_integration.py` | 27项通过 |
| Task6+Task1–5公共接口+历史FCR+Task7契约联合串行测试 | 79项通过，0项失败 |
| `py_compile`覆盖5个改动Python文件 | 通过 |
| `git diff --check` | 通过，仅提示工作树LF未来可能转换为CRLF |
| C1/C2/M6显式`--fcr_version v2 --fcr_matrix_row`dry-run | 通过；分别为legacy bypass、identity-noop、完整M6登记路由 |

测试环境为`C:\Users\lh594\.conda\envs\ssr-gpu\python.exe`，即项目规定的`ssr-gpu`环境。联合测试只有既存`torch.cuda.amp.autocast`弃用警告。

## 真实checkpoint无query smoke

仓库内只找到`best_test_model.pth`可用于真实文件smoke。独立读取确认其元数据为`seed=7、epoch=1、phase1_method=adv3b02、model_variant=lite_c、num_classes=3`。smoke只构造全0张量`[2,2,128]`，没有打开数据集、target、query或truth。

结果：

- `SMOKE_STATUS=PASS`
- `loaded=246、skipped=0、missing=24、unexpected=0`
- `source_only=True、query_access=False`
- `tx_logits_shape=(2,3)`
- `fcr_decode_is_none=True、decoder_mode=not_run_identity_only`

该结果只证明真实checkpoint加载、192↔160投影和无query identity-only代码路径可运行。固定N607 checkpoint路径未镜像到本地，因此本报告不把`seed=7/epoch=1`解释为正式初始化证据。Task8必须在N607核对以下固定身份：

- checkpoint：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_fcr_r1r8_s392005_equalized_20260903_v4/ADV3B02/ADV3B02_CORE90_SOFT_E200/final_ssdg.pth`
- seed：`392005`
- epoch：`200`
- candidate：`ADV3B02_CORE90_SOFT_E200`

## Git与关注点

- 基线HEAD：`8f023908ae745580c21db7d7d2c03137113d6a79`。
- 实现提交：`5d68753190e9fe9f83bdad6369bd7c657e36ee87`。
- push后独立`ls-remote`回读：远端分支OID与实现提交一致。
- 两个无关未跟踪`local_artifacts/...`目录未暂存、未修改、未删除。
- 旧`test_phase1_adv3b02_fcr_launcher.py::test_launcher_freezes_defaults_four_evaluations_and_no_query_paths`仍要求旧launcher文本含`seed=392002`，而共享HEAD中的既有launcher已不满足该断言。该冲突不由Task6引入；按主任务裁定仅记录，不修改Task7 launcher。Task7当前冻结V2契约3项全部通过。
- 唯一未闭合证据是N607正式E200 checkpoint身份和真实远端smoke；由Task8完成，不阻塞Task6代码交付。
