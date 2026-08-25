# Time-only Rank-8 Class-Floor Scale-8 Meta-Adapter P4 Target5最小预登记报告

- run ID：`stage2_meta_adapter_target5_time_r8_floor8_p4_s392002_20260825_r1`
- 状态：`LANDED`
- 分支：`codex/meta-adapter-tri-r4-v1-20260824`
- 代码／配置冻结提交：`8b551e6275dbe5e022a884395d75887c16f2eaa6`；push后独立回读远端OID一致。
- Phase1：`phase1_adv3b02_meta_adapter_time_r8_floor8_p4_s392002_20260825_r1`，状态`ARTIFACTS_COMPLETE / SOURCE_SELECTION_ELIGIBLE`。
- 冻结基线：`ADV3B02_CORE90_SOFT_E200`。

## 候选与最小矩阵

- time-only rank8、P4 FOMAML+Meta-SGD、正式3步support更新；Phase1和Phase2共同使用`frozen_prototype_class_floor_ce_v1`／scale8，最终仍为冻结原型余弦argmax。
- 正式bundle预算5458／1055125=0.517285%；仅更新10个`id/dom_backbone.meta_adapter_time`张量，不含分类头、协方差、LDA或持久新头。
- 单seed=`392002`、receiver=`20-1`；`K10/new5`、`K10/new10`、`K10/new20`、`K5/new20`、`K1/new20`各三类LEO weak，共15行。
- 配置：`configs/stage2_meta_adapter_target5_time_r8_floor8_p4_s392002_20260825_r1.json`。相对普通scale8 Target5只替换bundle ID、checkpoint和prototype，15行完全相同。

## Phase2权限与判决边界

- 沿用同一`p2_min_v1`、`VALIDATED_ONCE`、capsule和split，不重验数据。
- 仅读取合法target support IQ／标签、正式bundle和冻结原型；不读取source／clean样本、source cache、query真值或query角色。
- query在3步support反向传播结束并冻结状态后才逐样本推理，`query_state_update_count`必须为0。
- 同一冻结判决比较`DA0_REG0`与`DA1_REG0`；REG0新类指标为N/A。

## 本地验证与审查

- Target5配置与普通scale8计划逐字段比较，除三项bundle路径字段外完全一致：5个operating point、15个row、seed、receiver、scenario、K和3步均未变化。
- 复用本候选274项邻近回归、11个生产入口编译、真实checkpoint无query smoke和唯一一次P0/P1审查；审查结论P0无、P1无。Stage2无新增代码变化，不重复审查。

## N607预登记

- 普通`N607`账户；现有`CVS-RFFI`环境；GPU0。
- release CWD：`/home/szu2070436088/2510044040/CV-SincNet/releases/stage2_meta_adapter_target5_time_r8_floor8_p4_s392002_20260825_r1/checkout`
- 工厂输出：`/home/szu2070436088/2510044040/CV-SincNet/stage2_inputs/meta_adapter_target5_time_r8_floor8_p4_s392002_20260825_r1`
- smoke输出：`/home/szu2070436088/2510044040/CV-SincNet/runs/stage2_meta_adapter_target5_time_r8_floor8_p4_s392002_20260825_r1_smoke`
- prediction输出：`/home/szu2070436088/2510044040/CV-SincNet/runs/stage2_meta_adapter_target5_time_r8_floor8_p4_s392002_20260825_r1`
- stdout：`/home/szu2070436088/2510044040/CV-SincNet/logs/stage2_meta_adapter_target5_time_r8_floor8_p4_s392002_20260825_r1.out`
- 工厂命令：`python code/scripts/build_stage2_meta_adapter_target_matrix.py --plan configs/stage2_meta_adapter_target5_time_r8_floor8_p4_s392002_20260825_r1.json --output-root <工厂输出>`。
- prediction命令：`python code/scripts/run_stage2_meta_adapter_matrix.py --config <工厂输出>/matrix_config.json --output-dir <prediction输出> --device cuda:0`。15行全部prediction闭合后才以`score_stage2_meta_adapter.py`连接各row truth并用`summarize_stage2_meta_adapter_matrix.py`生成Target5总表。
- expected artifacts：工厂／smoke receipt、15行DA0／DA1预测与receipt、truth-last `score.json`、矩阵receipt及`target5_summary.json`。
- 技术停止规则：仅在协议越权、query／source泄漏、错误checkout／row／split、输出覆盖、prediction不完整、scorer连接错误或重复确定性pre-prediction异常时停止；不得因低性能停止。

## 晋级门槛

15行prediction闭合后才连接truth。聚合`DA1_REG0-DA0_REG0`旧类均值至少+1.0pp且floor至少+0.5pp才进入Target25，否则记为科学失败并继续下一少层候选。

## Release、工厂与smoke

- release提交：`092cdba5648d3d9aa5b1e76d49ac62e099d918aa`；远端OID独立回读一致。归档本地／远端唯一一次SHA256均为`cbf1b07780539199b0b26b3a369d8de7c68e70085c2ff25b97e46a1ec6b529fe`，16个相关入口远端编译通过。
- 发布前确认release、工厂、smoke、prediction、stdout和同名Python进程均不存在，正式bundle／原型非空，GPU0空闲。
- truth-free工厂一次完成15／15行，状态`TARGET_INPUTS_COMPLETE`，`query_truth_opened=false`、`query_role_opened=false`、`source_opened=false`。无query配置由首row精确删除`query_path`且本地断言其余字段不变。
- 本次正式Phase1 bundle真实smoke一次通过：3次反向传播，`frozen_prototype_class_floor_ce_v1`／8.0，`trainable_fraction=0.005172846819097263`，`query_opened=false`、`source_opened=false`、`query_state_update_count=0`。
