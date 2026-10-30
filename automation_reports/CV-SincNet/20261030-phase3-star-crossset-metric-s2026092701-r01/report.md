# PAMF 开放表征重设计：已知与辅助未知的合法分离

固定 run `20261030-phase3-star-crossset-metric-s2026092701-r01`。这是前一独立度量残差的机制修正，不替换已经固定的 Tri-SIG-TIME SSL/default6head 或 DURA v1.0。唯一主 arm 为 `S_SEP_TOPO`，匹配控制为 `S_SEP`；唯一主 readout 为 `learned_known_support`，其余分数仅作预登记诊断，不能按结果换主。

## 问题与固定假设

前轮 L 监督度量与 OE 软邻域损失只约束各自集合内部关系，没有显式约束 L 与 OE 的相对位置。共享网络限制了任意独立变换，但并不等价于学习已知与未知的区分。本轮直接把 OE 的合法 outside-six 成员信息加入 L anchor 的对比分母。该信息证明 L 与 OE 不同旧设备；不能证明任意两个 OE 来自不同设备，也不能证明 OE 覆盖所有未来未知。

OE 背景密度可能很低，即使 known 支持也很低，密度比仍可能偏大。因此预先把 absolute known-support density 固定为主 readout，并保留相同特征的 density ratio 作诊断。以上为结构假设，不把前轮负结果写成已证实的因果解释。

## 输入、标签与来源

只使用既有同 G10MIX01/P1、原 F0 的 L6300 与 OE6134 特征训练。L 为旧六设备各1050；OE 为21辅助设备，算法只能看到 outside-six 成员角色与公共 RX/day，OE 真 TX/K 隐藏。两 arm 都读取同一批 L192/OE64，模型初始化、L/OE随机流和2220步完全匹配。

known V27000（旧六设备各4500）仅在模型与聚合统计冻结后用于既定 known42 风险/阈值校准；source novel3150（19-15/19-16/7-1各1050）只在全部阈值冻结后预测，随后独立 truth-last。原 Us56700、本轮 target U/Q、正式 RX24-18 均不读取、不优化，历史 query 不转训练。四 source 角色互斥，共42584条既有物理记录。

源 RX1/3/4/6/8 对应1-19/18-2/19-2/2-19/3-19；日期2021-03-08/15/23均已见。L/novel 每TX×RX×day70，V300。source novel 有历史开发曝光，不是新RX、新TX盲测或未来注册观测。

全部沿固定 G10MIX01/P1：600km/25MHz/2.45GHz、residual/post_sync/noEQ/refSNR30dB/reflect，郊区/城区各0.5及原六面积权重；receiver2027/scene2026103001/channel2026104001，原一次 unitRMS/单 received 不变。P0是叠加星地信道后的 received，不是clean；本轮不生成新IQ/信道、不读P0/clean实值、不运行G。

L/OE 原 fused160 与四贡献来自 protected-cache actualcode `7dd90671968ddf58aab5a5b302b4d56371c8b9c9`；V/novel 的 u160/c640/defaultlogits 与固定原 F0 density reference 来自 contribution-density actualcode `edc28ad00baace5e4b9567fbf3391154bf6c34b4`。原 F0/sourcecontract392005、DURA祖先与现有匹配核验结论复用，不加载前轮任何学习后的开放网络或640密度权重。

## 网络、损失与预算

同前轮102688参数新网络：640→128 Linear→GELU→160 Linear，seed2026092701，最后层权重与bias精确零初始化，z=unit(u+g(c))。u为原F0 fused160单位向量，c为四贡献各自FP64单位化、拼接整体单位化后FP32的640向量。无0.1硬残差界、maxmember关系头、新身份头、EMA或目标状态。原F0/default6head冻结；训练容器不加载原F0，旧身份始终为原默认head argmax。

L每步192：六类×15公共RX/day格各2，再每类两个不同轮转格各1；extra域 `(2*step+c)%15` 与下一格，L RNG2026103017。OE每步64：15域各4，再从 `4*step mod15` 起连续四域各1；OE RNG2026103018 独立。step从0计，格内无放回、单batch不重复，两arm全部L/OE索引匹配。

新 cross-set 监督项仅以L为anchor，T=.1。候选集合为L∪OE，候选须同时不同RX且不同day。L候选与OE候选各占先验质量1/2；每个集合内部先按有效公共域等权，再域内按样本均分。因此candidate logit为 cosine/T + log(.5) − log(有效域数) − log(该域candidate数)。两个集合每个anchor均须有有效候选，否则报错。正样本只为L集合中同旧TX且有效的记录，目标先按有正样本的公共域等权，再域内均分；OE目标质量为0。L anchor loss为目标对全部候选学生概率的KL；anchor先按旧六类×15域格内平均，再格等权。L和OE投影均参与反向，teacher与原输入detach。没有OE-anchor硬负关系或OE–OE异设备假设。

`S_SEP` 只优化上述 cross-set KL；`S_SEP_TOPO` 优化 cross-set KL + OE软邻域KL（权重1）。OE项完全复用前轮 frozen-u teacher、跨RX且跨day、公共域等权、T=.1、teacher stopgrad 的软结构。它是弱结构约束，不是真实未知身份标签；通过匹配控制检验它是否帮助保持未知结构。

两arm fresh AdamW lr1e−4/wd1e−4/clip5、常量LR、固定2220更新，不早停或追加。每arm L426240/OE142080训练特征曝光，全部L/OE索引与真实loss/梯度/漂移记录。原F0仍E50/11100/232993，新native更新0；新4440总更新不写成原F0续训。合成梯度probe不更新优化器，不计入2220。

## 冻结读出、评分与推进

共同prepare第一步实际原F0 synthetic G10两行no-query smoke，PASS立即导出既有L/OE派生特征；新真实F0/G前向0。每arm训练终点严格重载真实新checkpoint，合成两行投影smoke后映射L/OE，拟合同族D160 known90 Gaussian和background32/25Lloyd/seed3019/covshrink.1。known统计与background统计均在读取V前冻结。

固定三个输出列：`fused160_density_ratio`（原F0参考，模型/阈值/预测逐值复用）、`learned_known_support`（主）、`learned_density_ratio`（诊断）。两个learned读出使用同一次冻结投影和同一known聚合统计；known-only状态保存known字段，ratio状态另含OE background字段，不用不同参数或不同拟合预算制造差异。全部列的旧身份一致，不另训或替换分类头。

每个learned读出在相同V上独立执行已固定known42/common-tail六predicted-class阈值规则。所有模型、聚合统计与两套learned阈值均固定后，才消费novel特征。独立CUDAempty CPU scorer验证保存的features/聚合统计能重建全部raw/margin后，唯一连接novel真值。报告宏A_K、R_U、H、C_K、同输入原F0闭集损失、AUROC/OSCR、逐TX/RX/day/scene/42风险组；保存主与固定F0、主与learned-ratio的配对纠错，并在矩阵结束后比较两arm。不能依据NV结果选readout、阈值或arm。

唯一主 `S_SEP_TOPO::learned_known_support` 必须通过known42和source整体宏R_U≥90%才另预登记跨未见RX星地验证；控制或诊断通过不替代主。source门槛不代表最终跨RX成功。本轮无target DA/discovery/REG，A_N/H_OW=N/A；完整目标保持ACTIVE：跨RX星地宏C_K/R_U≥90%、A_K较同输入原闭集下降≤2pp，主报A_K/R_U/H，继续争取更高拒识及未来无标签注册。

N607：prepare后最多两个本任务train/eval worker，每卡新增≤2、启动实际free≥4096MiB；独立CPU scorer一次最多1个。允许与外国owner共卡，不停止/修改/重启其工作。每row冻结即自动预测/独立评分，健康peer继续。非法来源/角色、非有限、输出碰撞、无法产生合法预测或系统技术失败保留产物且不自动重启；科学负结果不停止健康训练。唯一launch owner和不可覆盖root，一次代码commit、聚焦负测、一次直接P0/P1审查、最小登记、一次preflight/archive SHA比较/remote compile、PID/CWD/argv/GPU/log增长与独立终态读回。

## 实际发布预登记

Actual code `fb016284e6c1f4715da4c690e2333efce89103b5`；唯一launch owner `codex-phase3-crossset-metric-20261030`。唯一逐row矩阵维护于experiment.json，本报告以上固定设计与配置共同生效。

环境 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/20261030-phase3-star-crossset-metric-s2026092701-r01-release`；run/output `/home/szu2070436088/2510044040/CV-SincNet/runs/20261030-phase3-star-crossset-metric-s2026092701-r01`；日志同root/logs。命令 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -B -u -m experiments.pamf_phase3_open_world.star_crossset_metric_launch start --runtime /home/szu2070436088/2510044040/CV-SincNet/releases/20261030-phase3-star-crossset-metric-s2026092701-r01-release/runtime.json`。配置和各row独占路径见experiment.json。

本地聚焦验证：{"model_train": {"unique_tests": 8, "subtests": 13, "seconds": 5.92, "log_field_only_targeted_recheck_tests": 2, "targeted_seconds": 5.02, "stdout": "local_artifacts/pamf_phase3_star_crossset_metric_20261030_r01/core_train_tests.log"}, "eval_score": {"unique_tests": 8, "subtests": 14, "pair_test": 1, "pair_seconds": 2.95, "remaining_tests": 7, "remaining_seconds": 12.95, "legacy_trapz_deprecations": 972, "stdout": "local_artifacts/pamf_phase3_crossset_metric_20261030_r01/eval_score_tests.log"}, "launcher": {"tests": 5, "seconds": 3.37, "pytest_cache_write_warning": 1}, "AST_files": 10, "mirror_bytewise_matches": 10, "unique_tests_total": 21, "real_data_weights_truth_read_in_local_tests": 0}；唯一直接P0/P1 review PASS。一次N607preflight PASS，未读IQ/私有truth实值，原source和F0/DURA ancestry未变。尚未发生本轮新训练，参数和数据准备不是性能结果。

预期产物：commonprepare smoke/featuremetadata；每arm完整2220loss/gradient/indices、final.pt与严格重载smoke；known/background聚合状态及25步fit记录、三列预测与三套known-only阈值、独立CPUtruthlast结果；PID/CWD/argv/GPU/log增长和终态独立观察。IQ、权重、source逐行featurebank、原始日志和预测只存私有artifact，不入Git。


## 实际启动与独立观察

VERIFIED／RUNNING（所存snapshot时刻）。code `fb016284e6c1f4715da4c690e2333efce89103b5`；单archive SHA `c9649c3ef81a81f67fb02af9a9b62aaa8c668e4cdd7918fb0086c900272eb8bf` 比较/remotecompile/runtime PASS。独立读取本run进程PID/CWD/完整argv/CUDA/GPU，并跨两次观察确认日志增长：[{"pid": 795666, "argv": "/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -B -u -m experiments.pamf_phase3_open_world.star_crossset_metric_train train --config /home/szu2070436088/2510044040/CV-SincNet/runs/20261030-phase3-star-crossset-metric-s2026092701-r01/configs/S_SEP_train_config.json --device cuda:0 ", "cwd": "/home/szu2070436088/2510044040/CV-SincNet/releases/20261030-phase3-star-crossset-metric-s2026092701-r01-release", "CUDA_VISIBLE_DEVICES": "6"}, {"pid": 795669, "argv": "/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -B -u -m experiments.pamf_phase3_open_world.star_crossset_metric_train train --config /home/szu2070436088/2510044040/CV-SincNet/runs/20261030-phase3-star-crossset-metric-s2026092701-r01/configs/S_SEP_TOPO_train_config.json --device cuda:0 ", "cwd": "/home/szu2070436088/2510044040/CV-SincNet/releases/20261030-phase3-star-crossset-metric-s2026092701-r01-release", "CUDA_VISIBLE_DEVICES": "4"}, {"pid": 795502, "argv": "/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -B -u -m experiments.pamf_phase3_open_world.star_crossset_metric_launch driver --runtime /home/szu2070436088/2510044040/CV-SincNet/releases/20261030-phase3-star-crossset-metric-s2026092701-r01-release/runtime.json ", "cwd": "/home/szu2070436088/2510044040/CV-SincNet/releases/20261030-phase3-star-crossset-metric-s2026092701-r01-release", "CUDA_VISIBLE_DEVICES": ""}]。离散GPU状态：["795669, GPU-95d925a1-7218-ca7b-adef-164241f4f491, 344", "795666, GPU-5463e082-259e-537b-603a-01575d45801c, 344"]，不当峰值；foreign操作0。固定预算健康继续，冻结后逐row自动预测/独立truthlast；本时刻无新性能结果，完整目标ACTIVE。


## 已完成实际结果

# PAMF L–OE分离与已知支持主读出：实际源侧结果

**执行VERIFIED，科学判定NO_PROMOTION。** 两arm各2220新开放网络更新、全部预测与独立truth-last评分完成。唯一主S_SEP_TOPO::learned_known_support的A_K90.496296%、C_K98.685185%、R_U23.015873%、H36.698290%，较同输入原F0闭集损失0.485185pp。完整跨未见RX星地、无标签适应/发现/匿名注册目标继续ACTIVE。

Run `20261030-phase3-star-crossset-metric-s2026092701-r01`，actualcode `fb016284e6c1f4715da4c690e2333efce89103b5`。固定设计 `docs/PAMF_PHASE3_CROSSSET_METRIC_PLAN_20261030.md`，唯一row矩阵维护于experiment.json；不按结果换主、调阈值或追加训练。

## 数据、星地条件与权限

| 角色 | 物理记录 | 训练/校准/评分用途 |
| --- | --- | --- |
| source L | 6300，旧六类各1050 | 两arm真旧TX监督的Lanchor；冻结后known聚合统计 |
| source OE | 6134，21辅助设备 | 两arm合法outside-six负候选；主另软邻域KL；两arm背景统计。真OE TX/K隐藏 |
| known V | 27000，旧六类各4500 | 模型及统计冻结后，两个learned分数独立known42风险/六阈值校准 |
| source novel | 3150，19-15/19-16/7-1各1050 | 所有τ冻结后预测，本row全部预测固定后独立truth-last |
| 原Us | 56700祖先记录 | 本轮新读0、真TX不解封 |
| target U/Q、正式RX24-18 | 本轮0 | 无适应、发现、注册或正式跨未见RX测试 |

四source角色物理互斥，共42584条既有记录，没有新增物理样本。source RX1/3/4/6/8对应1-19/18-2/19-2/2-19/3-19；日期2021-03-08/15/23。L/novel每TX×RX×day70、V300；OE不按隐藏设备补齐。源RX和日期已见、novel历史开发曝光，不是新RX/TX盲测，也不是未来注册观测。

全部postPhase1输入沿固定G10MIX01/P1：600km/25MHz/2.45GHz、residual/post_sync/noEQ/refSNR30dB/reflect，郊区/城区各0.5及原六面积权重；receiver2027/scene2026103001/channel2026104001，原一次unitRMS/单received不变。P0是星地received，不是clean。本轮只用既有P1特征，没有新IQ/信道、clean/P0实值或G运行。

L/OE原fused160/四贡献来自protected-cache actualcode7dd90671968ddf58aab5a5b302b4d56371c8b9c9；V/novel的u160/c640/defaultlogits及固定F0 density参考来自contribution-density actualcodeedc28ad00baace5e4b9567fbf3391154bf6c34b4。原Tri-SIG-TIME SSL F0 E50/11100/default6head/232993、sourcecontract392005和DURA v1.0祖先匹配复用；没有前轮学习后的NN或640density权重进入初始化。

## 实际方法与完整训练

新网络102688参数，640→128 Linear/GELU→160 Linear，seed2701、最后层精确零初始化；z=unit(u+g(c))。u是原fused160，c为四贡献各自单位化后拼接整体单位化的640向量。原F0/default6head冻结，不进入训练容器；旧身份逐条复制原默认head。没有0.1硬残差界、maxmember头、新identityhead或EMA。

两arm每步同L192/OE64，独立RNG3017/3018；L六类×15域各2＋每类两轮转格，OE15域各4＋四轮转格，同batch无重复。Lanchor仅跨RX且跨day，L/OE候选集合各0.5先验、各自有效公共域等权/域内均分；positive仅L同旧TX，target按正域等权。crossset KL对两侧student反向，不把OE–OE当硬负；主另加frozen-u OE软邻域KL权1/T.1，控制无该项。

两arm freshAdamWlr1e−4/wd1e−4/clip5/constantLR，固定2220步。原F0 native更新0；新4440更新不是原F0续训。合成probe不更新参数，实际首两步额外VJP只验证crossset对L/OE梯度，不改变optimizer次数。完整trace/采样与公开metadata重放分析保存在私有analysis.json；没有用V/NV早停、选模或追加预算。

| arm | trace量 | 全2220步均值 | 首222步 | 末222步 |
| --- | --- | --- | --- | --- |
| S_SEP | loss | 0.707833523 | 1.034598254 | 0.585405391 |
| S_SEP | L_metric | 0.707833523 | 1.034598254 | 0.585405391 |
| S_SEP | OE_KL | 0.000000000 | 0.000000000 | 0.000000000 |
| S_SEP | L_feature_drift | 0.616195478 | 0.177091468 | 0.778691405 |
| S_SEP | OE_feature_drift | 0.390020094 | 0.085000496 | 0.614760595 |
| S_SEP | gradient_norm_before_clip | 0.479367535 | 0.673631044 | 0.584591250 |
| S_SEP_TOPO | loss | 0.843039826 | 1.088044095 | 0.770058293 |
| S_SEP_TOPO | L_metric | 0.793467234 | 1.050789165 | 0.718694539 |
| S_SEP_TOPO | OE_KL | 0.049572591 | 0.037254934 | 0.051363754 |
| S_SEP_TOPO | L_feature_drift | 0.264158466 | 0.112407462 | 0.343322330 |
| S_SEP_TOPO | OE_feature_drift | 0.103951305 | 0.052310925 | 0.147656912 |
| S_SEP_TOPO | gradient_norm_before_clip | 0.344356466 | 0.583266383 | 0.343818151 |

| arm | 实际step | hidden梯度 | last梯度 | L最小行VJP | OE最小行VJP |
| --- | --- | --- | --- | --- | --- |
| S_SEP | 1 | 0.000000000 | 0.975875684 | 0.001967327 | 0.003062592 |
| S_SEP | 2 | 0.002718267 | 0.743477497 | 0.003783545 | 0.003109326 |
| S_SEP_TOPO | 1 | 0.000000000 | 0.975875676 | 0.001967327 | 0.003062592 |
| S_SEP_TOPO | 2 | 0.002714631 | 0.742477253 | 0.003783545 | 0.003109326 |

两集合最小行VJP为正，分离目标确实作用于L和OE。全部4440步连续有限，L与OE索引在两arm全部2220步逐值相同；公开metadata逐步重放、192/64域配额和batch唯一性全部通过，各arm实际unique L6300/OE6134。两arm hidden首步为0、之后2219步非零，last2220步非零，clip0；首步hidden为0符合zero-last初始化。loss降低不能替代拒识效果；OE软teacher不是未知真实身份，不凭KL或漂移宣布未知身份保持。

## 冻结读出、主指标与配对

严格终点checkpoint重载/合成feature smoke后，各arm映射L/OE并拟合相同D160 Gaussian：known90、background32/25Lloydseed3019/covshrink.1。known-support8字段与ratio11字段的known统计逐值共享，V/NV两分数使用同一次NN投影。全部统计先冻结，V只校准两个learned读出的已固定known42/common-tail规则；三套阈值都在novel前冻结。独立CPU scorer从保存z/聚合统计重建三列raw/margin后才唯一连接novel truth，不重载NN、重fit或扫τ。

同P1原F0闭集90.981481%=24565/27000，所有候选默认分类身份逐条不变。A_K为误拒计错的旧六类宏平均正确接收，C_K为旧六类宏平均接收，R_U为三未知设备宏平均拒识；H=2A_KR_U/(A_K+R_U)，AUROC/OSCR只作固定分数曲线诊断。

| arm | readout | A_K% | C_K% | R_U% | H% | 原F0损失pp | AUROC% | OSCR% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S_SEP | fused160_density_ratio | 90.374074 | 98.325926 | 28.095238 | 42.864791 | 0.607407 | 91.705905 | 84.743671 |
| S_SEP | learned_known_support | 90.459259 | 98.562963 | 15.841270 | 26.961099 | 0.522222 | 90.380103 | 83.967991 |
| S_SEP | learned_density_ratio | 90.477778 | 98.525926 | 17.746032 | 29.672242 | 0.503704 | 90.848891 | 84.529431 |
| S_SEP_TOPO | fused160_density_ratio | 90.374074 | 98.325926 | 28.095238 | 42.864791 | 0.607407 | 91.705905 | 84.743671 |
| S_SEP_TOPO | learned_known_support | 90.496296 | 98.685185 | 23.015873 | 36.698290 | 0.485185 | 89.584163 | 83.137141 |
| S_SEP_TOPO | learned_density_ratio | 90.418519 | 98.325926 | 32.952381 | 48.301593 | 0.562963 | 91.787741 | 84.893828 |

| arm主−参考 | 参考 | 正确gain | 正确loss | 正确net | 接收net | 未知拒对gain | 未知拒对loss | 未知拒对net |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S_SEP | fused160_density_ratio | 56 | 33 | 23 | 64 | 183 | 569 | -386 |
| S_SEP | learned_density_ratio | 9 | 14 | -5 | 10 | 59 | 119 | -60 |
| S_SEP_TOPO | fused160_density_ratio | 76 | 43 | 33 | 97 | 233 | 393 | -160 |
| S_SEP_TOPO | learned_density_ratio | 60 | 39 | 21 | 97 | 114 | 427 | -313 |

主已知正确接收24434、接收26645、误拒355；未知拒对725/3150。A_K变化只来自拒绝集合，原分类器没有改善。source整体R_U90早筛与known42必须共同通过；本轮pipeline eligibility=False。密度比诊断的收益不能把固定主改为密度比，也不构成跨RX达标。

## 主读出的完整设备、日期和42分组

| arm | 未知TX | 分母 | 拒对 | R_U% |
| --- | --- | --- | --- | --- |
| S_SEP | 19-15 | 1050 | 116 | 11.047619 |
| S_SEP | 19-16 | 1050 | 153 | 14.571429 |
| S_SEP | 7-1 | 1050 | 230 | 21.904762 |
| S_SEP_TOPO | 19-15 | 1050 | 314 | 29.904762 |
| S_SEP_TOPO | 19-16 | 1050 | 213 | 20.285714 |
| S_SEP_TOPO | 7-1 | 1050 | 198 | 18.857143 |

| arm | 旧TX | 分母 | 闭集% | A_K% | C_K% | 额外失去正确数 |
| --- | --- | --- | --- | --- | --- | --- |
| S_SEP | 14-10 | 4500 | 89.044444 | 88.511111 | 98.200000 | 24 |
| S_SEP | 14-7 | 4500 | 80.333333 | 79.822222 | 98.688889 | 23 |
| S_SEP | 20-15 | 4500 | 95.911111 | 95.533333 | 99.111111 | 17 |
| S_SEP | 20-19 | 4500 | 85.466667 | 84.533333 | 97.933333 | 42 |
| S_SEP | 6-15 | 4500 | 96.333333 | 95.755556 | 98.066667 | 26 |
| S_SEP | 8-20 | 4500 | 98.800000 | 98.600000 | 99.377778 | 9 |
| S_SEP_TOPO | 14-10 | 4500 | 89.044444 | 88.466667 | 98.177778 | 26 |
| S_SEP_TOPO | 14-7 | 4500 | 80.333333 | 79.844444 | 98.755556 | 22 |
| S_SEP_TOPO | 20-15 | 4500 | 95.911111 | 95.466667 | 98.955556 | 20 |
| S_SEP_TOPO | 20-19 | 4500 | 85.466667 | 84.644444 | 98.266667 | 37 |
| S_SEP_TOPO | 6-15 | 4500 | 96.333333 | 95.977778 | 98.511111 | 16 |
| S_SEP_TOPO | 8-20 | 4500 | 98.800000 | 98.577778 | 99.444444 | 10 |

RX和日期已见，scene是同一G10MIX固定分配的切片，不是六次新物理观测。下表所有required TX覆盖完整，未剔除困难设备或用缺类0填充。

| arm | 组 | known n | novel n | A_K% | C_K% | R_U% | H% | 原F0损失pp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S_SEP | overall | 27000 | 3150 | 90.459259 | 98.562963 | 15.841270 | 26.961099 | 0.522222 |
| S_SEP | rx/1 | 5400 | 630 | 93.000000 | 99.685185 | 13.492063 | 23.565360 | 0.092593 |
| S_SEP | rx/3 | 5400 | 630 | 84.074074 | 96.314815 | 13.174603 | 22.779591 | 1.074074 |
| S_SEP | rx/4 | 5400 | 630 | 91.166667 | 98.888889 | 26.190476 | 40.691148 | 0.462963 |
| S_SEP | rx/6 | 5400 | 630 | 92.574074 | 99.481481 | 13.015873 | 22.822862 | 0.259259 |
| S_SEP | rx/8 | 5400 | 630 | 91.481481 | 98.444444 | 13.333333 | 23.274441 | 0.722222 |
| S_SEP | scene/practical_high | 1103 | 133 | 97.370002 | 98.339177 | 23.306233 | 37.610188 | 0.542878 |
| S_SEP | scene/practical_mid | 1983 | 256 | 96.948982 | 98.534782 | 21.653429 | 35.400257 | 0.493681 |
| S_SEP | scene/practical_low_suburban | 10304 | 1203 | 94.994222 | 98.531963 | 19.984098 | 33.021422 | 0.610696 |
| S_SEP | scene/practical_high_urban | 1171 | 130 | 96.382529 | 98.565109 | 27.310789 | 42.561441 | 0.569874 |
| S_SEP | scene/practical_mid_urban | 1979 | 236 | 91.099254 | 98.698798 | 12.617521 | 22.165108 | 0.614577 |
| S_SEP | scene/practical_low_urban | 10460 | 1192 | 83.210100 | 98.583564 | 9.065867 | 16.350339 | 0.412459 |
| S_SEP | rxscene/1/practical_high | 202 | 20 | 98.569157 | 99.019608 | 30.952381 | 47.111085 | 0.000000 |
| S_SEP | rxscene/1/practical_mid | 437 | 46 | 98.569227 | 100.000000 | 19.289534 | 32.264966 | 0.000000 |
| S_SEP | rxscene/1/practical_low_suburban | 2042 | 245 | 97.210196 | 99.652464 | 16.200980 | 27.773285 | 0.048876 |
| S_SEP | rxscene/1/practical_high_urban | 236 | 22 | 99.290780 | 100.000000 | 16.190476 | 27.841142 | 0.000000 |
| S_SEP | rxscene/1/practical_mid_urban | 401 | 44 | 93.945919 | 100.000000 | 6.302521 | 11.812575 | 0.000000 |
| S_SEP | rxscene/1/practical_low_urban | 2082 | 253 | 86.376140 | 99.620868 | 8.847491 | 16.050892 | 0.188098 |
| S_SEP | rxscene/3/practical_high | 208 | 25 | 92.621246 | 96.615955 | 8.333333 | 15.290910 | 1.181698 |
| S_SEP | rxscene/3/practical_mid | 407 | 54 | 92.354423 | 94.755300 | 14.692982 | 25.352542 | 1.817016 |
| S_SEP | rxscene/3/practical_low_suburban | 2007 | 239 | 90.666235 | 96.364110 | 17.064407 | 28.722850 | 1.229574 |
| S_SEP | rxscene/3/practical_high_urban | 241 | 29 | 91.175490 | 95.091905 | 28.703704 | 43.661859 | 0.653595 |
| S_SEP | rxscene/3/practical_mid_urban | 388 | 44 | 86.307010 | 96.514268 | 6.944444 | 12.854582 | 1.828704 |
| S_SEP | rxscene/3/practical_low_urban | 2149 | 239 | 74.246306 | 96.658426 | 8.654437 | 15.501911 | 0.632999 |
| S_SEP | rxscene/4/practical_high | 213 | 26 | 97.207456 | 97.207456 | 47.222222 | 63.565219 | 0.641026 |
| S_SEP | rxscene/4/practical_mid | 403 | 48 | 97.880771 | 99.184735 | 27.777778 | 43.274577 | 0.268817 |
| S_SEP | rxscene/4/practical_low_suburban | 2049 | 248 | 95.477339 | 98.735235 | 31.513781 | 47.386809 | 0.577233 |
| S_SEP | rxscene/4/practical_high_urban | 238 | 30 | 96.476988 | 98.196248 | 47.777778 | 63.907159 | 1.803752 |
| S_SEP | rxscene/4/practical_mid_urban | 392 | 40 | 92.208995 | 99.140212 | 25.740741 | 40.246429 | 0.297619 |
| S_SEP | rxscene/4/practical_low_urban | 2105 | 238 | 84.265148 | 99.124331 | 14.865937 | 25.273211 | 0.280750 |
| S_SEP | rxscene/6/practical_high | 240 | 24 | 99.593496 | 99.593496 | 10.000000 | 18.175074 | 0.406504 |
| S_SEP | rxscene/6/practical_mid | 365 | 47 | 98.266005 | 99.722222 | 22.380952 | 36.458222 | 0.277778 |
| S_SEP | rxscene/6/practical_low_suburban | 2104 | 240 | 96.502285 | 99.433880 | 17.437980 | 29.538371 | 0.243231 |
| S_SEP | rxscene/6/practical_high_urban | 233 | 27 | 98.791974 | 100.000000 | 21.481481 | 35.289548 | 0.000000 |
| S_SEP | rxscene/6/practical_mid_urban | 408 | 57 | 92.960665 | 99.712644 | 11.931818 | 21.149080 | 0.000000 |
| S_SEP | rxscene/6/practical_low_urban | 2050 | 235 | 85.762602 | 99.367965 | 6.223728 | 11.605270 | 0.329836 |
| S_SEP | rxscene/8/practical_high | 240 | 38 | 98.016317 | 99.164578 | 19.576720 | 32.635231 | 0.396825 |
| S_SEP | rxscene/8/practical_mid | 371 | 61 | 97.272834 | 98.631709 | 22.100122 | 36.017228 | 0.260417 |
| S_SEP | rxscene/8/practical_low_suburban | 2102 | 231 | 95.007067 | 98.413764 | 16.870080 | 28.652444 | 0.961968 |
| S_SEP | rxscene/8/practical_high_urban | 223 | 22 | 95.911730 | 99.185053 | 12.121212 | 21.522443 | 0.387597 |
| S_SEP | rxscene/8/practical_mid_urban | 390 | 51 | 90.133822 | 97.965208 | 15.479876 | 26.421959 | 1.027668 |
| S_SEP | rxscene/8/practical_low_urban | 2074 | 227 | 85.779487 | 98.304752 | 6.720415 | 12.464310 | 0.610265 |
| S_SEP | day/1 | 9000 | 1050 | 90.555556 | 98.688889 | 16.857143 | 28.423230 | 0.477778 |
| S_SEP | day/2 | 9000 | 1050 | 90.566667 | 99.011111 | 18.285714 | 30.427928 | 0.333333 |
| S_SEP | day/3 | 9000 | 1050 | 90.255556 | 97.988889 | 12.380952 | 21.774898 | 0.755556 |
| S_SEP_TOPO | overall | 27000 | 3150 | 90.496296 | 98.685185 | 23.015873 | 36.698290 | 0.485185 |
| S_SEP_TOPO | rx/1 | 5400 | 630 | 92.962963 | 99.722222 | 26.825397 | 41.636239 | 0.129630 |
| S_SEP_TOPO | rx/3 | 5400 | 630 | 84.037037 | 96.592593 | 16.825397 | 28.037327 | 1.111111 |
| S_SEP_TOPO | rx/4 | 5400 | 630 | 91.296296 | 99.092593 | 26.349206 | 40.895485 | 0.333333 |
| S_SEP_TOPO | rx/6 | 5400 | 630 | 92.796296 | 99.814815 | 29.047619 | 44.245319 | 0.037037 |
| S_SEP_TOPO | rx/8 | 5400 | 630 | 91.388889 | 98.203704 | 16.031746 | 27.278250 | 0.814815 |
| S_SEP_TOPO | scene/practical_high | 1103 | 133 | 97.576310 | 99.131364 | 42.005420 | 58.728802 | 0.336570 |
| S_SEP_TOPO | scene/practical_mid | 1983 | 256 | 96.900083 | 98.727699 | 32.291478 | 48.440423 | 0.542580 |
| S_SEP_TOPO | scene/practical_low_suburban | 10304 | 1203 | 95.150092 | 98.784814 | 28.854808 | 44.281116 | 0.454826 |
| S_SEP_TOPO | scene/practical_high_urban | 1171 | 130 | 96.461145 | 98.744369 | 29.144060 | 44.763582 | 0.491257 |
| S_SEP_TOPO | scene/practical_mid_urban | 1979 | 236 | 90.908112 | 98.385173 | 19.935897 | 32.700636 | 0.805719 |
| S_SEP_TOPO | scene/practical_low_urban | 10460 | 1192 | 83.173082 | 98.585218 | 12.889773 | 22.320431 | 0.449477 |
| S_SEP_TOPO | rxscene/1/practical_high | 202 | 20 | 98.569157 | 99.019608 | 50.793651 | 67.040616 | 0.000000 |
| S_SEP_TOPO | rxscene/1/practical_mid | 437 | 46 | 98.334486 | 99.765258 | 45.951108 | 62.633814 | 0.234742 |
| S_SEP_TOPO | rxscene/1/practical_low_suburban | 2042 | 245 | 97.162454 | 99.753828 | 35.220588 | 51.700259 | 0.096618 |
| S_SEP_TOPO | rxscene/1/practical_high_urban | 236 | 22 | 99.290780 | 100.000000 | 32.857143 | 49.375144 | 0.000000 |
| S_SEP_TOPO | rxscene/1/practical_mid_urban | 401 | 44 | 93.700821 | 99.526591 | 21.191554 | 34.565670 | 0.245098 |
| S_SEP_TOPO | rxscene/1/practical_low_urban | 2082 | 253 | 86.417275 | 99.757514 | 15.499007 | 26.283965 | 0.146963 |
| S_SEP_TOPO | rxscene/3/practical_high | 208 | 25 | 92.621246 | 97.542331 | 16.666667 | 28.249921 | 1.181698 |
| S_SEP_TOPO | rxscene/3/practical_mid | 407 | 54 | 92.354423 | 95.841159 | 19.956140 | 32.820383 | 1.817016 |
| S_SEP_TOPO | rxscene/3/practical_low_suburban | 2007 | 239 | 90.916715 | 96.828979 | 21.712443 | 35.053516 | 0.979093 |
| S_SEP_TOPO | rxscene/3/practical_high_urban | 241 | 29 | 91.502287 | 96.568128 | 16.666667 | 28.197335 | 0.326797 |
| S_SEP_TOPO | rxscene/3/practical_mid_urban | 388 | 44 | 86.163127 | 96.498590 | 9.027778 | 16.343191 | 1.972588 |
| S_SEP_TOPO | rxscene/3/practical_low_urban | 2149 | 239 | 73.925743 | 96.568187 | 12.551604 | 21.459646 | 0.953562 |
| S_SEP_TOPO | rxscene/4/practical_high | 213 | 26 | 97.848482 | 100.000000 | 48.148148 | 64.538794 | 0.000000 |
| S_SEP_TOPO | rxscene/4/practical_mid | 403 | 48 | 97.607547 | 98.911511 | 40.370370 | 57.117152 | 0.542041 |
| S_SEP_TOPO | rxscene/4/practical_low_suburban | 2049 | 248 | 95.814544 | 99.170580 | 31.953058 | 47.924006 | 0.240029 |
| S_SEP_TOPO | rxscene/4/practical_high_urban | 238 | 30 | 97.090264 | 98.412698 | 37.777778 | 54.391750 | 1.190476 |
| S_SEP_TOPO | rxscene/4/practical_mid_urban | 392 | 40 | 92.208995 | 99.126984 | 31.851852 | 47.348173 | 0.297619 |
| S_SEP_TOPO | rxscene/4/practical_low_urban | 2105 | 238 | 84.228886 | 99.053407 | 12.691441 | 22.059066 | 0.317013 |
| S_SEP_TOPO | rxscene/6/practical_high | 240 | 24 | 100.000000 | 100.000000 | 66.296296 | 79.732739 | 0.000000 |
| S_SEP_TOPO | rxscene/6/practical_mid | 365 | 47 | 98.543783 | 100.000000 | 49.084249 | 65.528850 | 0.000000 |
| S_SEP_TOPO | rxscene/6/practical_low_suburban | 2104 | 240 | 96.695466 | 99.806129 | 35.538378 | 51.974592 | 0.050050 |
| S_SEP_TOPO | rxscene/6/practical_high_urban | 233 | 27 | 98.791974 | 100.000000 | 41.944444 | 58.887025 | 0.000000 |
| S_SEP_TOPO | rxscene/6/practical_mid_urban | 408 | 57 | 92.722570 | 99.761905 | 26.913876 | 41.718453 | 0.238095 |
| S_SEP_TOPO | rxscene/6/practical_low_urban | 2050 | 235 | 86.092438 | 99.744618 | 12.281197 | 21.495967 | 0.000000 |
| S_SEP_TOPO | rxscene/8/practical_high | 240 | 38 | 98.016317 | 99.164578 | 28.042328 | 43.608365 | 0.396825 |
| S_SEP_TOPO | rxscene/8/practical_mid | 371 | 61 | 97.272834 | 98.892126 | 20.695971 | 34.130306 | 0.260417 |
| S_SEP_TOPO | rxscene/8/practical_low_suburban | 2102 | 231 | 95.056879 | 98.319054 | 17.482118 | 29.532796 | 0.912156 |
| S_SEP_TOPO | rxscene/8/practical_high_urban | 223 | 22 | 95.557120 | 98.830443 | 12.121212 | 21.513486 | 0.742207 |
| S_SEP_TOPO | rxscene/8/practical_mid_urban | 390 | 51 | 89.858663 | 97.076894 | 15.479876 | 26.410106 | 1.302828 |
| S_SEP_TOPO | rxscene/8/practical_low_urban | 2074 | 227 | 85.589044 | 97.972041 | 11.372724 | 20.077616 | 0.800708 |
| S_SEP_TOPO | day/1 | 9000 | 1050 | 90.600000 | 98.855556 | 27.142857 | 41.771415 | 0.433333 |
| S_SEP_TOPO | day/2 | 9000 | 1050 | 90.522222 | 98.988889 | 23.619048 | 37.463201 | 0.377778 |
| S_SEP_TOPO | day/3 | 9000 | 1050 | 90.366667 | 98.211111 | 18.285714 | 30.416619 | 0.644444 |

## 成本、执行和声明边界

| arm | 冻结投影行 | 固定F0 B | 主known B | 诊断ratio B | 三套条件τ B |
| --- | --- | --- | --- | --- | --- |
| S_SEP | 42584 | 364340 | 322868 | 364340 | 144 |
| S_SEP_TOPO | 42584 | 364340 | 322868 | 364340 | 144 |

每arm训练L426240＋OE142080=568320特征行，两arm1136640；评估两arm各42584，共85168；合计新增真实开放网络投影1221808行。原F0真实前向0/common合成G10两行，G0；OPEN合成probe和实际重载smoke共20行，未计作真实IQ或2220 optimizer更新。所有density状态是聚合统计，不部署source逐行bank。两arm25Lloyd全部有限、单调下降，32个mode无空；known8/ratio11共享known8项逐值相同。8份完整stdout共65行，无warning/error/nonfinite。

新NN102688参数，原F0+NN335681；加既有DURA266951的算术总数602632，不能据此推定B1时延。主部署可只保留known-support统计和对应六阈值，报告中的固定F0/ratio是额外诊断状态。原F0 E50/11100及既有G/缓存成本均保留，不能因本轮真实F0/G前向0而称完整系统无成本。

Pipeline实际墙钟196.731775s；没有测端到端B1/峰值显存，离散GPU状态不当峰值。actualcodefb016284e6c1f4715da4c690e2333efce89103b5，523文件/data0 archive SHA c9649c3ef81a81f67fb02af9a9b62aaa8c668e4cdd7918fb0086c900272eb8bf单次比较、remote compile/runtime PASS；21项本地独立测试、27子测试与唯一直接P0/P1审查PASS。已知本地legacy metrics.trapz弃用警告和pytestcache写警告与远端运行分别记录。

独立live核实driver795502，S_SEP train795666/CUDA6/344MiB、S_SEP_TOPO train795669/CUDA4/344MiB及跨快照日志增长；主eval797355/CUDA6/332MiB和控制CPU scorer797270被live观察。其余短stage依据jobs/完整产物，未冒称全部有live GPU观测。终态所有row train/eval/score COMPLETE、无本run live/GPU，foreign停止/修改/重启0。

## 研究判断与下一步

合法L–OE分离和OE软结构都实际执行，但唯一主known-support没有达到未知拒识目标。密度比在主arm优于固定F0，属于已固定诊断的有限收益；当前证据不足以支持R_U90或95，也没有证明跨未见RX、域适应或未来新类注册有效。不能由seen-source结果外推目标性能。当前所有三列/两arm known42都通过；主校准共同尾率再前进一步会在RX3/practical_mid（407条）产生2.030692pp身份正确损失，虽该时接收率仍95.627484%，绑定条件是2pp正确接收保护。此项只分析V风险规则，不用NV寻找新τ。

当前所有新学习仍发生在冻结F0输出之后。本轮没有检验浅/中层信号表征能否保留更多未知设备信息。后续优先转向信号级开放分支设计：先核实合法L/OE的固定G10/P1 IQ复用路径和身份保持视图，再让开放目标进入可训练浅/中层，同时保留冻结旧分类路径及同输入2pp保护。该方向尚未实现或验证，不继续凭本轮未知结果扫描阈值、残差权重或密度超参数。

完整目标继续ACTIVE：默认跨未见RX，所有postPhase1样本叠加星地信道，宏C_K/R_U≥90%、A_K距同输入闭集≤2pp，主报A_K/R_U/H并继续争取更高拒识；无标签DA、未知发现和未来独立观测上的匿名注册仍需完成。A_N/H_OW本轮N/A，不用OE软结构或聚类回测替代注册性能。
