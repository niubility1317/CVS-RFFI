# Phase1 ADV3B02六折配置等价重训练预注册报告

## 1.状态、目的与边界

- 实验ID：`phase1_adv3b02_clic6_20260813_v1`
- 当前状态：`LOCAL_VERIFIED / NOT_RELEASED / NO_PERFORMANCE_RESULT`
- 本地实现操作者：Terra/max；N607唯一runner：待主控完成独立P0/P1复审后指定。
- 方法身份：`ADV3B02_CORE90_SOFT_E200_CLIC_EQ_RHO07_FINAL`。
- 历史机制锚点：`ADV3B02_CORE90_SOFT_E200`。本任务是保持其完整机制和损失配置的配置等价重训练，不是历史checkpoint的字节级复现。
- 目的：为F1—F6各建立一个source-only ADV3B02训练入口，使其`split_mode=tx_rx_day_1_6_3`、`0.07/0.63/0.30`数据角色、seed、epoch与对应CLIC折一致。后续独立入口才会生成目标已注册known的rich reference；本任务不训练、适配、更新、选择或读取任何目标包、truth、query或scorer结果。
- 假设：在逐字段训练/known-test语义配置等价的前提下，六个ADV折可作为每个CLIC折的合法已知类DG比较组件。该假设尚无性能结论。

ADV3B02在本任务中没有独立source-frozen的端点真实unknown决策规则。因此其目标真实unknown指标为`N/A`，不能记为零，不能以历史proxy/vaccept替代，也不能作为70%门的通过证据。

## 2.冻结训练矩阵

|折|候选目录|source训练TX顺序|held-known验证TX|proxy-unknown TX|GPU|seed|
|---|---|---|---|---|---:|---:|
|F1|`F1_ADV3B02_CLIC`|`20-15,20-19,6-15,8-20`|`14-7`|`14-10`|0|392002|
|F2|`F2_ADV3B02_CLIC`|`14-10,20-19,6-15,8-20`|`20-15`|`14-7`|1|392002|
|F3|`F3_ADV3B02_CLIC`|`14-10,14-7,6-15,8-20`|`20-19`|`20-15`|2|392002|
|F4|`F4_ADV3B02_CLIC`|`14-10,14-7,20-15,8-20`|`6-15`|`20-19`|3|392002|
|F5|`F5_ADV3B02_CLIC`|`14-10,14-7,20-15,20-19`|`8-20`|`6-15`|4|392002|
|F6|`F6_ADV3B02_CLIC`|`14-7,20-15,20-19,6-15`|`14-10`|`8-20`|5|392002|

每折训练、held-known和proxy-unknown TX集合两两互斥。启动器在派生命令前检查角色数量和互斥性；`--dry-run`固定按F1→F6输出六条完整命令且不创建run/log根目录。

## 3.冻结配置

|类别|固定值|
|---|---|
|数据与时间表|`split_mode=tx_rx_day_1_6_3`；`labeled_ratio=0.07`；`unlabeled_ratio=0.63`；`source_val_ratio=0.30`；`epochs=200`；`label_epochs=130`；`pseudo_epochs=70`；`from_scratch=true`；`seed=392002`。|
|选择边界|`checkpoint_selection=final_only`；`phase1_source_val_selection_only=true`；`best_metric=joint_safe`仅保留源侧遥测，不参与checkpoint选择。|
|历史机制与守卫|保留historical`set_candidate_defaults`和`ADV3B02_CORE90_SOFT_E200`分支的全部有效与零权重旗标：ground prototype、feature mask、TX/RX geometry、prototype memory、open-world feature、z-id compact、proxy unknown、soft unknown mixup、source episode、prototype export/fusion、satellite consistency与半监督/域/Fishr项。|
|关键分支值|`lambda_open_world_feat=0.0024`；`lambda_zid_compact=0.032`；`lambda_proxy_unknown=0.0045`；`proxy_unknown_core_quantile=0.90`；`proxy_unknown_accept_quantile=0.85`；`proxy_unknown_vaccept_cvar_alpha=0.30`；`lambda_soft_unknown_mixup=0.0045`；`lambda_source_episode=0.0035`；`lambda_sat_cls=0.68`；`lambda_sat_cons=0`。|
|其余历史损失项|`lambda_u=0.16`；`lambda_ent=0.01`；`lambda_domain=1`；`lambda_adv=0.35`；`lambda_group_ce=0.16`；`lambda_fishr=0.04`；`tau_min=0.92`；`tau_max=0.97`；`pseudo_quantile=0.86`；`use_ema_teacher=true`。完整逐旗标receipt由启动器`--print-contract`给出，并可用`--validate-contract-file`逐字节校验。|
|LEO训练压力|训练场景固定为`leo_clear_weak,leo_low_elev_weak,leo_rain_weak`；schedule=`1@0.30:leo_clear_weak;41@0.60:leo_low_elev_weak,leo_rain_weak;91@0.80:leo_clear_weak,leo_low_elev_weak,leo_rain_weak`。这属于source-only训练时的固定弱扰动，不引入目标接收机输入。|

## 4.本地实现与验证

|文件|用途|状态|
|---|---|---|
|`code/scripts/launch_phase1_adv3b02_clic6_v1_20260813.sh`|六折训练入口、冻结合同、根目录防覆盖、PID表与限定技术停止逻辑|已实现|
|`code/tests/test_phase1_adv3b02_clic6_baseline.py`|六折命令/角色/全部关键旗标、真实训练器解析、无目标侧接口、根目录保护和合同漂移拒绝|已实现|
|本报告|预注册、N607交接与结果表模板|已建立|

|验证命令|结果|
|---|---|
|`python -m pytest code/tests/test_phase1_adv3b02_clic6_baseline.py -q`|5/5通过。先前RED为启动器不存在，5项均按预期失败；GREEN后通过。|
|`bash -n code/scripts/launch_phase1_adv3b02_clic6_v1_20260813.sh`|通过。|
|`bash code/scripts/launch_phase1_adv3b02_clic6_v1_20260813.sh --dry-run`|通过；恰六条F1→F6源侧命令。|
|真实`train_ssdg.build_arg_parser().parse_args(...)`|六条命令均接受`--from_scratch true`、`--checkpoint_selection final_only`及全部训练旗标。|
|历史旗标逐项差异审计|历史ADV3B02命令157个旗标与新命令162个旗标比较：除预注册的split/比例/路径/身份变化及5个新增final-only／三角色旗标外，`unexpected_changed=[]`、`missing_historical=[]`、`unexpected_added=[]`。|
|`python -m py_compile code/SSDG/train_ssdg.py`|通过。|
|`git diff --check`|通过。|

计划Task1 Git提交：`PENDING_TASK1_COMMIT`。提交后回填commit、launcher/test/report的SHA256和干净工作树摘要；未提交前不得N607同步或启动。

## 5.N607发布合同

- 远端项目根：`/home/szu2070436088/2510044040/CV-SincNet`。
- 待建只读release：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_adv3b02_clic6_20260813_v1_<Task1Commit>`。
- 环境：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`。
- 工作目录：release内项目根；`CODE_ROOT=<release>/code`。
- 启动器：`<release>/code/scripts/launch_phase1_adv3b02_clic6_v1_20260813.sh`。
- 单次正式命令：`RUN_ID=phase1_adv3b02_clic6_20260813_v1 PROJECT_ROOT=/home/szu2070436088/2510044040/CV-SincNet CODE_ROOT=<release>/code PYTHON=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python RUN_ROOT=/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_clic6_20260813_v1 LOG_ROOT=/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_adv3b02_clic6_20260813_v1 bash <release>/code/scripts/launch_phase1_adv3b02_clic6_v1_20260813.sh`。
- 正式launch次数：1；retry：`NO`；不复用、不恢复或覆盖任一旧run。
- 本地→远端映射：launcher、测试和本报告均来自上述Git commit；正式同步仅发送launcher和经审核的release archive，不修改远端源文件。

## 6.预期工件、健康与停止规则

每折预期工件为`final_ssdg.pth`、训练器默认`metrics_epoch.csv`/`metrics_epoch.jsonl`、`phase2_zid_prototypes.pt`、完整`F#_ADV3B02_CLIC.out`日志、`F#_ADV3B02_CLIC.pid`和`status/F#_ADV3B02_CLIC.status`。外层预期工件为`frozen_contract.txt`、`outer.pid`、`pids.tsv`及必要时的`systemic_stop.status`。训练器当前ADV路径不承诺不存在的CLIC专属terminal/config JSON；完成性以final checkpoint、嵌入checkpoint的args、完整log、PID/GPU绑定和本任务冻结合同共同核验。

N607 runner在启动前必须确认release哈希、训练器可编译、数据可见、run/log根均不存在、GPU0—5资源可用及每卡训练进程数不超过2。启动后立即核验outer PID、六个child PID、CWD/cmdline/run-root、GPU映射和日志增长。停止仅限协议/hash/覆盖错误，或至少两折在有效`final_ssdg.pth`前出现相同确定性异常指纹；停止时仅向已登记的本run child PID发送有限`TERM`，保留所有部分工件。accuracy、loss、DG、proxy或任何性能值绝不构成停止、重试或选择依据。

## 7.风险与后续接口

- 训练入口保留历史方法机制，不等于历史`0.10/0.70/0.20`原件，也不保证性能与历史数值相同。
- 本任务只闭合训练配置。六个checkpoint完成后，独立Task2/Task3接口才能按既定权限生成3120条盲态预测、开启truth-side rich known metrics并严格ingest six-fold ADV reference。
- ADV与CLIC不要求使用同一目标capsule字节；比较前必须重新核验双方训练数据语义配置、known-test语义配置和指标定义逐字段等价。
- 缺任何fold的合法ADV rich reference时，原combined scorer必须维持`FAIL/CANNOT_ESTABLISH`。

## 8.结果表模板

|折|候选|机制类别|训练TX／known／proxy|K-shot|seed|known DG overall|三scene已知指标|macro/min-class/min-RX|min-day|真实unknown拒识|checkpoint／log|结论|
|---|---|---|---|---:|---:|---:|---|---|---|---|---|---|
|F1|`F1_ADV3B02_CLIC`|source-only ADV3B02 DG comparator|见第2节|N/A|392002|N/A|N/A|N/A|N/A|`N/A—no independently source-frozen endpoint rule`|待运行|NOT_RUN|
|F2|`F2_ADV3B02_CLIC`|source-only ADV3B02 DG comparator|见第2节|N/A|392002|N/A|N/A|N/A|N/A|`N/A—no independently source-frozen endpoint rule`|待运行|NOT_RUN|
|F3|`F3_ADV3B02_CLIC`|source-only ADV3B02 DG comparator|见第2节|N/A|392002|N/A|N/A|N/A|N/A|`N/A—no independently source-frozen endpoint rule`|待运行|NOT_RUN|
|F4|`F4_ADV3B02_CLIC`|source-only ADV3B02 DG comparator|见第2节|N/A|392002|N/A|N/A|N/A|N/A|`N/A—no independently source-frozen endpoint rule`|待运行|NOT_RUN|
|F5|`F5_ADV3B02_CLIC`|source-only ADV3B02 DG comparator|见第2节|N/A|392002|N/A|N/A|N/A|N/A|`N/A—no independently source-frozen endpoint rule`|待运行|NOT_RUN|
|F6|`F6_ADV3B02_CLIC`|source-only ADV3B02 DG comparator|见第2节|N/A|392002|N/A|N/A|N/A|N/A|`N/A—no independently source-frozen endpoint rule`|待运行|NOT_RUN|
