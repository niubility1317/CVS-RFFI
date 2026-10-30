# PAMF protected160 源侧三组反证实验

## 预登记

Run `20261030-phase3-star-protected-open-s2026092701-r01`，实际执行代码 `7dd90671968ddf58aab5a5b302b4d56371c8b9c9`，唯一 owner `codex-phase3-protected-open-20261030`。固定设计为冻结原 Tri-SIG-TIME SSL F0/defaulthead，以四项同次融合贡献生成相对L2幅度≤0.1、零初始化的160维残差；身份始终输出原同P1 logits，拒识由support关系最大分数加固定known风险阈值决定。方法与固定预算见 `configs/pamf_phase3_protected_geometry_20261030.json` 和 `docs/PAMF_PHASE3_PROTECTED_GEOMETRY_DESIGN_20261030.md`，唯一逐行矩阵维护于 experiment.json。

主R1有界残差＋L/OE成员及软邻域；R0相同结构但残差关闭、学习关系；N0有界残差但完全不读OE样本，只用L成员＋L邻域。N0在任何新数据访问前加入，以满足协议的匹配无OE对照。原主R1和指标不变，控制不能按结果替主。三个arm各2220新增adapter更新、总6660；原F0权重仍原11100步、native新增0，不写成F0续训13320。各组相同L和slot随机流、parts初始化，N0较少辅助样本／计算，R0较少可训练参数，分别披露而不声称全部训练成本相同。

## 数据和合法来源

只复用既有source G10MIX01/P1：L6300（六旧TX各1050）、OE6134（21辅助设备，其真实TX/K仅builder保管）、knownV27000（六旧TX各4500）、source novel3150（19-15/19-16/7-1各1050）。源接收机编号1/3/4/6/8，对应1-19/18-2/19-2/2-19/3-19，日期2021-03-08/15/23，每旧L或novel设备每RX/day70条。Source RX已见、novel多次开发曝光，不是新RX/TX独立盲测。

原F0为 `20261027-phase1-pamf-repl-shuffled-views-m16-r02/main_split-repl-shuffled_time-3view-s2026092701`，完整Tri-SIG-TIME SSL E50/11100/232993，source split392005、原L6300/Us56700/V27000。三组各自严格加载该原父，不从已训练Phase3模型续训。全部既有source cache与该F0及DURA v1.0 R69 E20/4940/266951身份一致；DURA只作为缓存祖先，本run不加载或前向G。模型数据契约和来源核对由既有祖先证据及实际严格加载完成，无不明或目标训练祖先、无静默旧权重回退。

L/OE缓存只读各自P1及合法公共元数据，N0训练只打开L特征文件；原Us、P0/clean实值、target U/Q、正式RX24-18和private truth训练读取均0，不重抽实信道。L标签只为旧六类，OE仅outside-six成员监督，不能称整个系统零源标签。source novel不训练、早停、拟合参数或选阈值。

## 方法、自动预测与科学判定

每步L support96/query96互斥，六类每15公共RX/day格各1条＋轮转格1条；OE64只R0/R1按每域4＋四轮转格1，不按未知TX补齐。模拟已注册旧类子集K按2/3/4/5均匀循环、其余可见旧类模拟未注册；六类均已被F0见过，该模拟不宣称编码器未见TX。每query原型排除相同RX或相同day。成员BCE以模拟known/unknown各半，OE最大六类关系logit为负类；软邻域T.1、权重.1，跨RX且跨day、候选格及anchor格等权，教师stopgradient原unitfused160，不生成未知硬TX/K。N0仅L邻域；R0/R1 L/OE各半。新AdamW lr1e−4/wd1e−4、clip5，无EMA、GRL、VICReg、身份头训练、早停或目标状态更新。

每行固定终点冻结后立即自动预测，不等待其余行。sourceL聚合六类×五RX×三day的FP64 sums/counts，不保留逐source行；query依公共RX/day筛选后形成六原型。原knownV只做既定predicted-class/common-tail的六τ校准，必须逐条保持cache中独立原同P1 closed预测；全部阈值和known输出写完才打开novel。原型、模型及聚合不因V/NV更新。

每行known27000＋novel3150完整预测后，独立CUDAempty/BLAS1 scorer先核所有数组、重建known校准，并为novel同次输出的原/新160两表征固定全部8套近邻；之后唯一连接source novel truth。报告宏A_K/C_K/R_U/H、AUROC/OSCR、原F0损伤及计数、42known风险组、逐RX/day/scene/TX和未知几何。缺类或缺场景支持为N/A，不补采样，不把源检索当未来A_N。

唯一R1 known42风险全过且完整总体宏R_U≥90%，才记为可单独预登记跨RX星地开发验证；当前run从不访问target。几何只诊断、不另加晋级门槛；控制不能替主。该source门槛不证明跨新RX目标、无标签适应、发现或匿名注册完成，完整Phase3目标保持ACTIVE，AN/HOW为N/A。

## 执行、成本和保护

- Python `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/20261030-phase3-star-protected-open-s2026092701-r01-release`。
- 命令 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -B -u -m experiments.pamf_phase3_open_world.star_protected_open_launch start --runtime /home/szu2070436088/2510044040/CV-SincNet/releases/20261030-phase3-star-protected-open-s2026092701-r01-release/runtime.json`。
- 独占输出 `/home/szu2070436088/2510044040/CV-SincNet/runs/20261030-phase3-star-protected-open-s2026092701-r01`；先一个featurecache worker，之后至多三个独立行pipeline。实际每次训练／预测开始前free≥4096MiB、每卡≤2新增own，foreign不整卡排除，不停止／修改／重启。
- 计划真实F前向12434共同cache＋各行30150的V/NV，共102884行；各stage合成smoke2，共cache1＋train3＋eval3=14行，G0。训练只有cached adapter前反向：N0每步192、R0/R1每步256，共1562880行；每行L聚合6300另计。原始source cache/G/F0训练成本属于历史，不当作零成本。各模型新增parts66177，active预计299170、原完整容器预计966899／unused667729，实际checkpoint首步确认。R0 trainable10305，N0/R1为66177。
- 聚焦cache/train13项＋24子项PASS9.51s，eval/score9项PASS9.92s（共享metrics弃用提示保留），launcher8项PASS7.65s；原P1缺arm已修复，实际producer→consumer定点回归1项PASS52.90s，原问题一次定点复核PASS。15AST和实际runtime validate通过；先前未改的model/loss/集成测试复用，不重复全套。
- 一次preflight已通过，仅原source契约／角色manifest、父文件存在及实际显存，无IQ/truth值读取；本run没有新capsule，派生特征复用VALIDATED_ONCE句柄。接下来单release archive本地／远端SHA比较及远端compile，launcher实际checkpoint no-query smoke先于真实数据，PASS立即接续，不另建批准或签章。
- 预期launch/pipeline/config/log、feature L/OE NPZ和smoke/complete、逐行parts-only final/2220trace/summary/complete、聚合／阈值／known/novel预测与几何特征、独立score结果及observations。数据、权重、原始日志和机器运行配置不Git。
- Cache技术失败不发任何行；行技术失败保留该行并停止未启动pending，已开始健康行继续，不自动重启／重试或按科学低分停机。UNKNOWN先核原PID/CWD/argv/root，foreign操作0。

此刻为代码、验证、输入预检与预登记完成，尚未宣称已启动或得到性能结果。实际状态、各行完成度及同row结论后续追加，完整目标不因本轮source小矩阵而缩小。
