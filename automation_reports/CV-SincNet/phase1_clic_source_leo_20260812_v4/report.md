# Phase1 CLIC source-L LEO weak导出v4预注册报告

## 状态与目标

- 实验ID：`phase1_clic_source_leo_20260812_v4`。
- 当前状态：`READY_TO_LAUNCH / FORMAL_LAUNCH=0 / NO_PERFORMANCE_RESULT_YET`。
- 目标：复用v3已经完整生成的6份不可变source-L received-IQ cache，只执行F1—F6×C／G共12个source-LEO forward与binding封存。
- 这是两轮发布工程修复后的更小独立one-shot：不重建cache、不改变物理样本、scene、seed、模型、checkpoint、矩阵、GPU映射或停止规则。

## 前序技术证据

- v3唯一正式launch完成6／6 cache与receipt；每份3920行，builder日志无异常。随后12／12 export在输出前同一位置失败：`export_phase1_clic_leo_features.py:330 torch.from_numpy(iq[index])`，均为N607 Torch 2.1／NumPy 2.2旧ndarray桥接TypeError；`source_leo.npz=0/12`、binding=0。
- v3已封存为`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`，报告commit=`cefac51c`；不得重试或改写v3。
- v4生产修复只处理同一数组桥接根因：exporter的NumPy→Torch转换替换为已在真实builder smoke验证的`memoryview→torch.frombuffer→reshape→clone`路径；source-LEO显式启用共享feature extractor的安全Tensor→NumPy开关，使用`contiguous→tolist→np.asarray(float32)`。该开关默认关闭，其他历史export路径不变；两者只改变本次12个forward的数组复制方式，不触碰模型forward、特征公式、标签、geometry或阈值。

## 冻结输入与输出

- checkpoint／terminal：`runs/phase1_clic12_20260812_v5/F{1..6}{C|G}_CLIC12/`。
- cache：只读`runs/phase1_clic_source_leo_20260812_v3/F{1..6}_SHARED/source_l_received_iq.npz`及receipt；同fold C／G读取同一cache路径和相同字节。生产exporter逐次重开receipt，核NPZ路径／SHA、3920行、fold／scene／source-only零访问字段，并把receipt内C／G checkpoint、terminal及source-split SHA与当前arm实际输入交叉绑定。
- 新输出：`runs/phase1_clic_source_leo_20260812_v4/F{1..6}{C|G}_CLIC12/{source_leo.npz,source_leo.binding.json}`；日志写入独立v4 log root。
- 12个导出按fold映射GPU`0,1,2,3,4,5`，同fold C／G可并发；无cache构建阶段。
- source-L-only；U、V、proxy、target、query、truth、role均不进入forward、fit、阈值或选择；每physical row只进行一次既有received-IQ forward。

## 本地门与发布合同

- TDD必须先证明旧exporter缺少安全converter，再使禁用`torch.from_numpy`、禁用`Tensor.numpy()`、非连续float64输入、完整DataLoader提取循环、float32／finite／contiguous／断别名测试转绿。
- 必须通过exporter／builder／postfreeze专项回归、`py_compile`、launcher`bash -n`、dry-run精确12行=C6／G6且全部绑定cache run v3。
- 独立审查要求`P0=0/P1=0`；只审兼容转换、v3 cache只读复用和v4输出不可覆盖，不重新审科学设计。
- 唯一N607运行席从干净Git commit发布；直连preflight、资源与v4 path ABSENT、至多一次SCP、远端SHA／静态门后唯一launch=1，入口显式`bash <launcher>`且不检查执行位。launcher强制项目根、v5训练根、v3 cache根、v4 run／log根的冻结绝对关系，拒绝环境重定向。
- 不再构建或修改v3 cache；发布前允许用一行v3 cache做consumer converter真实smoke，但不得写正式v4输出。至少两个不同row在输出前出现同一异常时停止exact v4 run。
- 不读取性能做停止、调参、选择或重试。

## 预期完成态

- 12／12`source_leo.npz`、12／12binding、12行PID/GPU表、12份日志闭合；所有进程、GPU与SSH连接退出。
- 完成只记`ARTIFACTS_COMPLETE / NO_PERFORMANCE_RESULT`。source common/proxy/PAIR、bundle、F6和target指标继续由后续独立阶段完成。

## 待回填

- Git commit、文件SHA、本地回归、独立审查结论。
- archive／release／SCP／真实converter smoke／唯一launch、PID／GPU／工件闭合。

## v4发布、静态门与smoke裁定（2026-08-12）

- 冻结commit=`d3a886bee2be6fc496489557fda45664d74c1bbc`；Task7/PAIR dirty未进入archive、未stage。干净archive=`E:\type10-7\code\runner_tmp_phase1_clic_source_leo_20260812_v4_d3a886be_git_archive.tar`，SHA256=`2F583DA5DAC68309D8799372977BE8619E2696C30E1F26FF8F55502348CC8E4B`，bytes=`267048960`。
- SCP恰1次；远端SHA／bytes闭合，原子release=`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_clic_source_leo_20260812_v4_d3a886be`。正确v4入口是commit中新文件`code/scripts/launch_phase1_clic_source_leo_export12_20260812.sh`（旧`launch_phase1_clic_source_leo12_20260812.sh`为v3历史文件，未调用）；入口SHA=`25A9E94A63EFACB8786D4BC0C2D3145ADA294019A23B4591F5E450EE2A51D692`，exporter SHA=`B57B3EEAD2DCEB6E398A43A016F7CD9909C8535890BD650DB98860957780DEE4`。
- 远端静态门通过：exporter/shared extractor/phase1_clic `py_compile`、exporter`--help`含sealed cache参数、v4入口`bash -n`、dry-run精确12行（C6/G6、cache_run=v3）通过，输出无target/query/truth/role输入。
- F1真实consumer smoke未执行forward：sealed asset校验强制`postfreeze-output-root`必须为正式v4 run root且输出目录必须位于正式候选目录，与“smoke不得写正式v4输出、临时路径可用”要求冲突；主控裁定不放宽sealed边界、不在正式root做临时smoke、不新commit，按已存在的v2真实F1 smoke、v3真实6 cache与本次动态安全桥接审查证据直接跳过。smoke跳过不是失败；v4 run/log/outer仍ABSENT，launch计数=0。
- 下一步仅调用正确v4入口：检查文件存在后以`bash "$REL/code/scripts/launch_phase1_clic_source_leo_export12_20260812.sh"`启动一次，retry=NO。
