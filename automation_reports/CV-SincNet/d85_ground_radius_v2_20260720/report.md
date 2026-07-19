# D85地面radius v2原型研发报告

## 实验身份与目标

|字段|值|
|---|---|
|实验ID|`D85_ADV3B02_GROUND_RADIUS_V2`|
|时间|2026-07-20 CST|
|执行者|Codex|
|目标|从ADV3B02 Phase1合法labeled-train流中生成只读int8中心、rank-3跨域残差和p90类内半径，使Phase2能按统一公式区分“可靠地面先验”和“宽散布地面先验”，避免D83/D84只加载中心却无法改善离散预测的问题|
|比较目标|D81地面中心Cauchy先验、D83逐cell精度加载、D84跨类共识中心|
|当前状态|`LOCAL_VALIDATED_REMOTE_GENERATION_PENDING`|

## 机制、假设与停止条件

主要差异不是继续增强地面中心融合强度，而是给每个合法domain×class中心附带量化p90余弦半径和rank-3跨域残差。Phase2后续统一使用标准化偏差

`e(c,d)=(1-cos(p_target(c),p_ground(c,d)))/(r_ground(c,d)+r_target(c)+epsilon)`

估计地面分量可靠度，再将可靠度用于弱正则、尺度校准或support-only闭式更新；所有类使用同一公式，不按TX/class ID分支。地面原型只读，不直接覆盖target-old或target-new原型。

预期可观察结果：D85真实组件应具有非退化的逐cell半径分布和rank-3残差，且持久化状态仅包含严格allowlist内int8数组与FP16尺度。若半径全部饱和/塌缩、active cell不足、双遍流hash不一致、checkpoint/WiSig/class binding任一哈希不匹配，立即失败关闭，不进入Stage2窄筛。若后续D85 Stage2锁定query相对D81无正向离散预测变化，则标记性能中性或负结果，不扩展到seed2/125。

最小验证矩阵：先生成一个固定Phase1组件并审计全部domain×class几何；然后仅在预登记development seed713101、K10/new5的105行合法单LEO弱观测上，与D81同row比较old-before/old-after/new/H/forgetting、全部逐类结果、混淆和资源。该组件未完成外部固定权威联合签名之前只能作为研发诊断输入，不能声称正式Phase2资格。

## 数据与协议边界

|项目|值|
|---|---|
|Phase1 checkpoint|`runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`|
|checkpoint SHA256|`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`|
|WiSig输入|`Dataset_WigSig/ManySig.pkl`|
|WiSig SHA256|`2b0a7a7488dd3650bcae7b1d80efbcffd1598aaa671ae6b0a0df2a24dc0f694f`|
|Phase1比例约束|checkpoint重建split，`rho_label<=0.1`|
|特征|逐样本L2归一化`z_id160`|
|统计|双遍有界流式：第一遍中心sum/count；第二遍4096-bin直方图估计p90余弦距离|
|持久化|仅严格3文件v2组件；不保留样本特征、count、source路径或FP32中心|
|Phase2 target/query访问|无|
|正式性边界|独立组件固定为`PENDING_OUTER_JOINT_SEAL`和`formal_phase2_eligible=false`；只有外部权威签发联合bundle后才可正式使用|

## 本地实现与验证

现有实现已覆盖所需最小链路，无需重写治理：

|文件|作用|
|---|---|
|`code/scripts/export_adv3b02_center_lowrank_radius_component.py`|真实checkpoint/WiSig入口、双遍有序流hash、严格输出|
|`code/cvsrffi/phase1_geometry_streaming.py`|有界内存中心和p90半径统计|
|`code/cvsrffi/phase1_center_lowrank_prototype_bundle.py`|int8中心、rank-3残差、int8半径codec与验证|
|`code/cvsrffi/phase1_adv3b02_deployment_bundle.py`|runtime、组件、binding、lock的联合内容根与外部签名请求|
|`configs/phase1_d85_adv3b02_center_lowrank_radius.json`|本次生成参数和协议边界的哈希绑定配置|

验证命令：

`python -m pytest tests/test_phase1_geometry_streaming.py tests/test_phase1_center_lowrank_prototype_bundle.py tests/test_export_adv3b02_center_lowrank_radius_component.py tests/test_phase1_adv3b02_deployment_bundle.py -q`

结果：38项通过；pytest退出后的Windows临时目录清理出现`PermissionError`告警，但测试进程退出码为0且不影响项目验证。

## N607预检与运行计划

2026-07-20 06:51 CST执行`tools\n607_ssh_preflight.ps1`，直连成功，项目根可见，8张RTX3090均为0%利用率和10MiB显存。随后只读核查显示没有用户训练Python进程和GPU compute app；checkpoint为8,582,116B且哈希匹配，ManySig为2,359,341,461B且哈希匹配。远端尚缺v2导出器、流式几何和codec文件，因此必须先完成本地Git提交和精确SCP同步。

计划远端工作目录：`/home/szu2070436088/2510044040/CV-SincNet`

计划输出目录：`runs/d85_ground_radius_v2_20260720/adv3b02_component`

计划日志：`logs/d85_ground_radius_v2_20260720/export.log`

计划GPU：GPU0；该任务是单次Phase1离线推理和有界统计，不启动训练。

计划环境：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`

计划命令：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=code /home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -u code/scripts/export_adv3b02_center_lowrank_radius_component.py \
  --checkpoint runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth \
  --wisig-pkl Dataset_WigSig/ManySig.pkl \
  --output runs/d85_ground_radius_v2_20260720/adv3b02_component \
  --device cuda:0 \
  --expected-checkpoint-sha256 2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98 \
  --expected-wisig-sha256 2b0a7a7488dd3650bcae7b1d80efbcffd1598aaa671ae6b0a0df2a24dc0f694f \
  --expected-class-handle-binding-sha256 76735ae6d9b2d7e58f683635ca2644e00fbd27a515246aab9d47488c1ab5111f \
  --generation-config configs/phase1_d85_adv3b02_center_lowrank_radius.json \
  --expected-generation-config-sha256 c12d5d23d1fb8908a6f8e02575e915ad6322ca28b6bddf2ec1bc8162662f1cd5 \
  --expected-generation-code-sha256 69bf0820414f43cc2a37dc8b9cdb14cf2d169a013c18ffa744847adf11e53f5e \
  --batch-size 512 --num-workers 0 --min-samples-per-cell 2 --radius-histogram-bins 4096
```

运行前再次核查GPU进程；输出目录必须不存在或为空。启动后记录PID、GPU、精确命令、stdout/stderr、输出成员及SHA256。完成后必须解析组件manifest和NPZ全部形状、active cell、半径/残差分布、量化误差与状态字节，再决定D85 Stage2算法。

## 同步与版本状态

根目录`E:\type10-7`不是Git仓库；本报告及配置先写入Git工作树`E:\type10-7\code\snapshots\d81wt`，提交后再精确镜像到主发布仓库和根目录报告承载面。

本地哈希与计划SCP映射：

|本地文件|SHA256|远端目标|
|---|---|---|
|`code/scripts/export_adv3b02_center_lowrank_radius_component.py`|`69bf0820414f43cc2a37dc8b9cdb14cf2d169a013c18ffa744847adf11e53f5e`|`code/scripts/export_adv3b02_center_lowrank_radius_component.py`|
|`code/cvsrffi/phase1_geometry_streaming.py`|`f7eb4e5950ecaccc5fbecb25dab8d955e747d5384990ecc63100b013d7d28bf0`|`code/cvsrffi/phase1_geometry_streaming.py`|
|`code/cvsrffi/phase1_center_lowrank_prototype_bundle.py`|`7bd410108129bbe8096e2b2c49180877adcc5160f9fc980eb1da404da5d5086c`|`code/cvsrffi/phase1_center_lowrank_prototype_bundle.py`|
|`configs/phase1_d85_adv3b02_center_lowrank_radius.json`|`c12d5d23d1fb8908a6f8e02575e915ad6322ca28b6bddf2ec1bc8162662f1cd5`|`configs/phase1_d85_adv3b02_center_lowrank_radius.json`|

## 结果表

组件生成和Stage2性能尚未完成。本节不得以设计、单测或启动状态代替性能结果；每个完成版本将补齐同row旧类、新类、H、遗忘、逐类、混淆、资源和缺陷。
