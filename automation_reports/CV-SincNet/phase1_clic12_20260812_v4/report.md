# Phase1 CLIC 12臂训练v4预注册与运行报告

## 1. 状态与目标

- 实验ID：`phase1_clic12_20260812_v4`
- 当前状态：`LANDED / REMOTE_STATIC_VERIFIED / SMOKE_VERIFIED / READY_TO_LAUNCH`
- 操作者：主控Codex；N607唯一runner：`Luna/max`
- 目标：训练P1-CLIC的F1—F6×C/G共12臂；训练完成后所有正式指标均叠加`LEO weak`目标域测试，联合评估未知类拒识与域泛化。
- v1—v3均为训练前系统性技术失败，全部封存为`NO_PERFORMANCE_RESULT`；v4不恢复、覆盖或重试旧run。

## 2. v3根因与v4核心修复

- v3唯一launch后12/12在训练前同一异常退出：`CLICConfigError: P1-CLIC source training forbids proxy-unknown rows`。
- 根因是CLIC配置门把“声明TX互斥proxy角色”错误等同于“训练读取proxy行”。实际`_phase1_tx_partition_view`只把4个source-L训练TX写入训练视图，known-validation和proxy TX只进入不可训练的分区回执。
- v4删除该字符串误判；数据构建后、任何模型forward前新增强制门：必须存在TX分区回执，且`held_tx_loaded_by_training=false`，否则fail-closed。
- 核心修复commit：`2f590771bbb2997f109205a22bd6a48f9ac5c31a`。
- 不改变候选、公式、loss、seed、epoch、fold、GPU映射或launcher；v4仅通过环境变量覆盖新`RUN_ID`。

## 3. 冻结矩阵与配置

- 矩阵：F1—F6×C/G=12臂；C=`raw_phase_control`，G=`complex_local_invariant_curvature`。
- G从同一份`received_i`提取lag=`{1,2,4,8}`的多尺度三点复曲率token；C/G除operator外配置相同。
- seed=`7281164`；40epoch；batch=128；AdamW；lr=`2e-4`；`clean CE+0.10×KL(clean-stopgrad→single-LEO)`；场景固定为`leo_clear_weak,leo_low_elev_weak,leo_rain_weak`；final-only。
- 每折TX角色为4个source-L训练TX、1个known-validation TX、1个proxy-unknown TX，三者互斥且并集为冻结六TX；proxy仅作后冻结诊断，训练forward/fit/update均为0。
- target/query/target truth/role训练访问均为0；正式registered/unknown都使用同规单份预固定`leo_*_weak received_i`。
- GPU映射：0=`F1C,F5G`；1=`F1G,F5C`；2=`F2C,F6G`；3=`F2G,F6C`；4=`F3C`；5=`F3G`；6=`F4C`；7=`F4G`。

## 4. 本地验证证据

- TDD RED：非空proxy角色声明在旧代码精确触发v3同款`CLICConfigError`。
- GREEN：新增两条回归分别证明角色声明可到达source-L数据构建边界，以及任何`held_tx_loaded_by_training=true`都在forward前被拒绝。
- `ssr-gpu`：`py_compile`通过；`test_phase1_tx_partition.py + test_phase1_clic.py`共171项通过（仅既有AMP弃用warning）。
- 本地真实模型单批技术烟测：C/G均forward/backward通过，各11个CLIC参数梯度finite/nonzero；`query_rows_opened=0`、`proxy_rows_loaded=0`。
- 独立审查：`SPEC=PASS / QUALITY=PASS / P0=0 / P1=0`；P2仅建议未来再补完整真实builder枚举测试，不阻塞发布。

## 5. N607发布与启动合同

- 发布源：Git commit `2f590771bbb2997f109205a22bd6a48f9ac5c31a`的干净archive；不得携带未提交Task7工作树。
- launcher：`code/scripts/launch_phase1_clic12_20260811.sh`；实际命令前置`RUN_ID=phase1_clic12_20260812_v4`。
- release：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase1_clic12_20260812_v4_2f590771`
- run/log/outer：`runs/phase1_clic12_20260812_v4`、`logs/phase1_clic12_20260812_v4`、项目根`phase1_clic12_20260812_v4_outer.out`。
- 正式launch前必须做一次N607真实路径技术烟测：真实ManySig、真实F1 final checkpoint、4+1+1分区、C/G单批forward/backward、proxy/query零访问；不读取性能。
- 正式launch唯一一次，retry=`NO`；launch后核验12 PID、CWD/cmdline、GPU映射、日志增长、TX分区与首批技术telemetry。
- 仅按P0/安全/系统性执行故障停止，绝不按任何性能指标停止。

## 6. 预期工件与后续评测

- 每臂：`final_ssdg.pth`、`phase1_clic_terminal_receipt.json`、完整log、PID/GPU绑定。
- 训练技术闭合后执行postfreeze和叠加LEO weak的目标域known/registered与真实unknown盲态测试；C/G使用各自同规配置，unknown拒识与域泛化同时报告。
- 每个目标域指标必须包含三种`LEO weak`场景的fold×scene、三scene等权、全局等权及样本池化口径；unknown的defer不计unknown拒识分子。

## 7. 运行回填

- archive SHA/bytes：`CD550674EFF0C67E694E0384A679AD04E9DE204070449C440294F2E634D1835A`，266864640 bytes；SCP=1，远端SHA/bytes闭合。
- SCP/release/launch：SCP=1，release原子落地，launch待执行，fresh-run retry=NO。
- release静态门：核心文件hash、远端py_compile、`train_ssdg.py --help`、`bash -n`、dry-run12、真实parser12/12、TX4+1+1六TX闭合通过，release无pycache。
- 真实路径烟测：通过；F1C/F1G真实checkpoint重建，真实`_build_ssdg_wisig_data`返回`tx_partition_enabled=true`、`held_tx_loaded_by_training=false`、四类`class_id_to_tx`及一条proxy TX回执；取到128行train batch；C/G各一次clean+`leo_clear_weak` forward，`clean CE+0.10 KL` backward有限且CLIC梯度非零；`proxy_rows_loaded=0`、`query_rows_opened=0`。烟测后GPU/SSH清零。
- PID/GPU/日志：待runner
- checkpoint/terminal计数：待runner
- 最终状态：待runner
