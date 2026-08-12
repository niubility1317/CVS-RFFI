# Phase1 CLIC source-L LEO weak第二波预注册报告

## 1. 状态与目标

- 实验ID：`phase1_clic_source_leo_20260812_v1`
- 当前状态：`LOCAL_VERIFIED / NO_PERFORMANCE_RESULT`
- 操作者：主控Codex；N607唯一runner待本地提交及独立审查完成后交接给`Luna/max`。
- 目标：为F1—F6各构建一份不可变source-L单观测LEO weak received-IQ缓存，并让同fold C／G checkpoint复用完全相同的NPZ字节，各导出一份source LEO特征与binding。
- 比较对象：同fold C参数匹配control与G多尺度三点复曲率token；G固定lag=`{1,2,4,8}`，其余训练数据配置、source物理行、信道场景、seed和received-IQ字节相同。
- 本波不读取性能，不执行PAIR、bundle或target测试，不产生晋级结论。

## 2. 输入与冻结数据

- 训练run：`phase1_clic12_20260812_v5`，12／12 final checkpoint与terminal已技术闭合。
- source第一波：`phase1_clic_postfreeze_20260812_v2`，12／12`source_clean_proxy.npz`已完成；报告commit=`c3139aea`。
- N607只读inventory证明：每fold source-L为3920行；4个local TX×7个source RX共28个cell，每cell恰140行；同fold C／G的TX、RX、day、equalized、signal及场景元数据逐元素和字节一致。
- ManySig：`/home/szu2070436088/2510044040/CV-SincNet/Dataset_WigSig/ManySig.pkl`，SHA256=`2b0a7a7488dd3650bcae7b1d80efbcffd1598aaa671ae6b0a0df2a24dc0f694f`。
- 场景固定为`leo_clear_weak`、`leo_low_elev_weak`、`leo_rain_weak`。每个TX×RX cell按稳定物理ID排序后轮转分配；140行分为47／47／46，每个物理ID只进入一个scene。
- base seed沿用冻结训练seed加`991`；不同scene使用固定步长`1000003`派生。信道实现只复用项目既有`simplified_leo_residual`／`leo_residual`配置，不新增信道公式。

## 3. 许可与边界

- 只读取source-L及其checkpoint／terminal／ManySig；U、source-V、proxy、target、query、truth和role均不参与缓存生成、forward、fit、阈值、选择或停止。
- 每fold缓存只构建一次，C／G共用同一`source_l_received_iq.npz`字节；不为不同arm重采样，不跨scene复用物理ID，不融合／择优／TTA。
- 缓存NPZ exact member仅为`received_iq,tx_ids,rx_ids,day_ids,physical_sample_id,sat_scenarios`；不保存clean IQ、feature、logit、模型状态或target信息。
- receipt仅保存计数、配置、路径和SHA；`held_validation_forward_rows=0`、`proxy_forward_rows=0`、`target_access=false`、`query_access=false`、`fit_rows=0`、`threshold_fit_rows=0`。
- 这不是重复目标数据验证，也不是零时适配；它是冻结source训练配置的一次性source-only后冻结输入构建。ADV3B02仍只要求训练／测试配置等价，不要求与CLIC复用同一包或received-IQ字节。

## 4. 冻结矩阵与资源

|阶段|行数|输入|输出|GPU|
|---|---:|---|---|---|
|source-L received-IQ cache|6|每fold C/G checkpoint+terminal、ManySig exact L split|每fold1份共享NPZ+receipt|fold F1—F6→GPU0—5，串行构建|
|source LEO forward/export|12|每fold共享NPZ、对应C/G checkpoint+terminal|每臂`source_leo.npz`+binding|每fold C/G同GPU并行；GPU0—5|

- launcher：`code/scripts/launch_phase1_clic_source_leo12_20260812.sh`。
- 构建器：`code/build_phase1_clic_source_leo_iq.py`。
- consumer：`code/export_phase1_clic_leo_features.py`。
- 环境：`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；工作目录为新release的`code`。
- 输出根：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_clic_source_leo_20260812_v1`。
- 日志根：`/home/szu2070436088/2510044040/CV-SincNet/logs/phase1_clic_source_leo_20260812_v1`。

## 5. 本地验证

- TDD RED：专项测试初次collection精确失败于`ModuleNotFoundError: build_phase1_clic_source_leo_iq`。
- GREEN：allocator与builder专项=`12 passed`；覆盖4×7×140、输入置换不变、三scene每cell≥20、duplicate／缺cell／59行／空ID、C/G配置漂移、非有限信道、ManySig TOCTOU、不可覆盖和consumer重开。
- 联合回归：`test_build_phase1_clic_source_leo_iq.py + test_phase1_clic.py + test_phase1_clic_postfreeze.py`全部通过，共309项；仅既有10条AMP弃用warning。
- `py_compile`通过；launcher`bash -n`通过；dry-run精确18行=`6 cache+12 export`，C6／G6，target／query／truth／role参数为0。
- `git diff --check`通过。独立Terra终裁：`P0=0/P1=0/ALLOW`；审查额外攻击每cell 61／139／141、全局3921及总数3920但两cell分别141／139，全部fail-closed。

## 6. 发布、健康和停止合同

- 正式launch恰1次；retry=`NO`。先串行完成6份cache并逐份经生产consumer重开，再并行执行12份C／G LEO export。
- 预期成功工件：6份共享`source_l_received_iq.npz`、6份cache receipt、12份`source_leo.npz`和12份binding。
- 启动后记录outer／12 export PID、确切CWD／命令、run-root、GPU映射、日志增长、工件计数及每fold C/G binding中的received-IQ SHA相等。
- 若错误checkout／hash、输出覆盖、协议字段漂移，或至少2个独立fold／arm在相应工件前出现同一确定性异常，则停止该run并封存为`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。
- 不得因accuracy、loss、AUROC、`u_gap`、unknown拒识率或其他性能值停止、调参、选scene、换seed、补采样或重试。

## 7. 待回填

- Git commit待提交；生产SHA=`9170A73079E7048A104979DD17833411C323E2396618875C8692CBD670958774`，launcher SHA=`90C71124BE8C113F41AE83F420237397E4C7BDC5B16AAC48EED5D49AC3A541F6`，测试SHA=`43253DD293CF90AB0998E155C78DE6284CCB617C477217FE9ACB47A19063C0E2`；独立审查=`P0=0/P1=0/ALLOW`。
- N607 archive／release／SCP次数、remote hashes、唯一launch、PID／GPU／日志及工件闭合。
- 完整工件返回前保持`NO_PERFORMANCE_RESULT`。
