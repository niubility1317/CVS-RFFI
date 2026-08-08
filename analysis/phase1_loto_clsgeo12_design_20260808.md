# P1-LOTO-CLSGeo12冻结设计

状态：`DESIGN_FROZEN`
标签：`DEVELOPMENT_CROSS_TX_CV_NON_CONFIRMATORY`

## 目的与边界

本矩阵只检验一个最小干预：在保持GeoSat-C已验证的clean→LEO一致性训练不变时，是否把**已知类**开集几何损失从`z_id`改接至顶层分类特征`id_feat_cls`，能够在未见TX轮转中不牺牲已知类保护指标。它不是未知TX训练、未知拒识阈值选择、K-shot注册或Phase3确认实验。

训练、模型选择、Q95校准和任何损失均不得读取held primary TX、held secondary TX、target/query角色或真值。`phase1_source_proxy_unknown_tx_ids`在本矩阵中仅是held-TX分割字段，绝不作为TX级proxy unknown训练数据，更不得被解释为按batch轮换known label。

## 固定TX轮转

TX顺序固定为`[14-10,14-7,20-15,20-19,6-15,8-20]`。在fold`Fi`中，primary held proxy是`TX_i`，secondary held known validation是`TX_(i+1 mod 6)`，其余四个TX是唯一的训练TX。

| Fold | 训练TX（仅此四个） | secondary known-validation TX | primary held proxy TX |
|---|---|---|---|
| F1 | 20-15,20-19,6-15,8-20 | 14-7 | 14-10 |
| F2 | 14-10,20-19,6-15,8-20 | 20-15 | 14-7 |
| F3 | 14-10,14-7,6-15,8-20 | 20-19 | 20-15 |
| F4 | 14-10,14-7,20-15,8-20 | 6-15 | 20-19 |
| F5 | 14-10,14-7,20-15,20-19 | 8-20 | 6-15 |
| F6 | 14-7,20-15,20-19,6-15 | 14-10 | 8-20 |

六个primary held-TX结果的预注册汇总是主开发读数；每fold的secondary held-TX只作敏感性读数，不能改写训练、模型选择或校准。所有已知类clean、三种`leo_*_weak`、最差类、最差receiver、最差day指标只在该fold四个训练TX的物理validation样本上选择和保护。Q95同样只由这些物理validation样本计算。

## 两臂、唯一差异与损失

两臂均从头训练120epoch，`seed=7281105`、`sat_view_seed=9281105`、`checkpoint_selection=final_only`，并逐字复用GeoSat-C的公共训练参数。两臂均使用clean→`leo_clear_weak,leo_low_elev_weak,leo_rain_weak`一致性KL，`lambda_sat_cons=0.10`；均保持domain、tail和vacuum开关为零。

| Arm | 冻结设置 | 几何项 |
|---|---|---|
| C | GeoSat-C对照：`lambda_open_world_feat=0` | 不调用开集几何损失；默认`z_id`路径不经过新选择器。 |
| G | C加唯一干预：`lambda_open_world_feat=0.0024`、`ow_feat_key=id_feat_cls` | `0.0024 L_OW(id_feat_cls,y,d)`，半径/类间/样本边界固定为12°/55°/5°。 |

记GeoSat-C全部原始目标为`L_C`。G的唯一新增项为：

`L_G=L_C+0.0024L_OW(f,y,d)`，其中`f=out["id_feat_cls"]∈R^(B×160)`。

`L_OW`只接收已选择的`f`。其余所有以`z_id`为输入的分类、域、卫星一致性和既有身份损失继续接收原始`z_id`，不得改接或新增对齐、OE、tail/vacuum、proxy训练、阈值扫描或模型分支。实现必须拒绝缺失、非Tensor、非有限、非二维、batch行数不匹配或维度不等于当前`z_id`（160）的`id_feat_cls`。配置、terminal manifest和completion receipt必须记录`ow_feat_key`。

## 运行矩阵与资源

共12个独立任务，每个进程使用`CUDA_VISIBLE_DEVICES=<物理GPU>`且训练CLI固定`--device cuda:0`。同一GPU最多两个进程，输出根目录和日志根目录已存在时必须拒绝覆盖。

| 物理GPU | 任务 |
|---|---|
| 0 | F1C、F5G |
| 1 | F1G、F5C |
| 2 | F2C、F6G |
| 3 | F2G、F6C |
| 4 | F3C |
| 5 | F3G |
| 6 | F4C |
| 7 | F4G |

启动器写入`pids.tsv`和`completion.tsv`。任一子任务异常结束后，启动器保持全部已有结果并以终态退出码8返回；不重启、不覆盖、不基于中间性能停机。

## 选择与窄晋级门

1. 先完成全部六fold的C/G同fold配对，再读取主汇总；禁止用任何单fold、secondary held或proxy读数调参。
2. 每个fold的known保护项至少包括clean与三种`leo_*_weak`、min-class、min-receiver和min-day。若任一保护项的`G-C<-2pp`，该fold的G拒绝；其他指标不能抵消这一拒绝。
3. Q95与任何部署阈值只可由该fold四训练TX物理validation产生；本矩阵不选择Phase3拒识阈值，也不报告confirmed-unknown性能。
4. 只接受配对、完整、同一fold与同一seed的产物。缺少任一指定场景、receiver/day或类floor字段时，该fold不能进入主汇总。
5. 本轮的可发布结论上限是“source-held cross-TX开发性泛化证据”；不是独立确认、不是K-shot、不是unknown-FAR或Phase3能力声明。

## 最小实现与验证

实现入口为`code/SSDG/train_ssdg.py`：解析`--ow_feat_key={z_id,id_feat_cls}`，默认`z_id`；仅当`lambda_open_world_feat>0`时选择特征并调用既有`open_world_feature_space_loss`。新增focused测试应覆盖parser allowlist、默认对象恒等、G选择顶层`id_feat_cls`、全部非法形状/值分支和反传只到被选特征。`code/scripts/launch_phase1_loto_clsgeo12_20260808.sh`还须通过`bash -n`和`DRY_RUN=1`命令审计。

## 尚未关闭的风险

- **P0：**focused synthetic no-query smoke已把`lite_d`顶层`id_feat_cls=[B,160]`写入测试契约；仍须在实际启动后审计source split receipt确实排除了两类held TX。该运行时分割证据闭合前不得发布结果。
- **P1：**六个已注册TX意味着每fold仅四个训练TX；跨TX轮转不能完全外推到最终四类部署或新TX。`id_feat_cls`几何是否带来稳健收益也尚无性能证据。
- **不可主张：**本设计不训练或评估真实unknown，不使用target/query真值或角色，不产生bundle v2更新，不构成Phase3confirmed-unknown、校准阈值、独立复现或上线结论。
