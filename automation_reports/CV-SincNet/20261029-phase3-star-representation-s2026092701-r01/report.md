# PAMF第三阶段：星地开放集表征与防塌缩学习

- run_id：`20261029-phase3-star-representation-s2026092701-r01`
- group_id：`pamf-phase3-star-native-open-representation`；类别：`diagnostic`；阶段：`Phase3-star-source-representation`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED；完整实测与科学判定见下。

## 目的与对照

Matched fixed2220-step original-F0 native increments: labeled anchor, plus source-membership OE, plus rawfused VIC regularization; no target optimization.

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

实际作业已完成，源端结果及权限边界见末尾追加。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。

## 实际发布前固定

代码/config d319e3196f90113045a4f4f0958eeffe497f8785；实际runtime已校验cache及全部3组train/eval严格接口，唯一主oe_vic/P1/energy。41项unique聚焦测试＝data6+cache6+train/loss12+eval/score13+launch4，各subtest另记；data与科学候选各一次独立P0/P1 PASS，prepare四个cap/split接口问题已修正定向核对，未加额外gate。

数据OE6134、novel3150均G10 VALIDATED_ONCE，原L/V合同与原F0/G祖先沿用已核实身份。一次N607 preflight通过，各卡实际余量已记录；archive仅一次SHA比对/remote compile。缓存首步F0/G真实checkpoint合成星地P0/P1 smoke；每组训练首步native F梯度合成星地P0 smoke，不加载G，scope分开，PASS即继续。当前未声称真实训练/性能已经完成。

固定2220updates，所有新IQ星地/P1缓存；evalmode保持梯度，关闭BN/dropout/旧6forwardhook/As/CRIC/U/PL/EMA。全矩阵三组独立且共享合法输入，逐组终点直接源预测与独立truth评分；科学失败不改配方、不停其他健康row、不扩seed、不访问target。源通过仅下一冻结目标小矩阵的eligibility，不是跨RX/注册完成。

## 2026-10-29 G10 源端表征学习结果

`20261029-phase3-star-representation-s2026092701-r01` cache＋三arm train/eval/score均完成；固定主 `oe_vic/P1/energy`：**NO_PROMOTION**，总目标 **ACTIVE**。

| arm/P1 energy | C_K% | A_K% | R_U% | H_OS% | macroOSCR% | F0损pp | arm源gate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| anchor | 97.4778 | 90.4852 | 17.0159 | 28.6450 | 79.3009 | 0.4963 | False |
| oe | 98.2074 | 90.8222 | 22.1905 | 35.6666 | 84.2900 | 0.1593 | False |
| oe_vic | 98.4481 | 90.7259 | 25.3968 | 39.6847 | 83.3727 | 0.2556 | False |

三组从原F0 E50/11100独立2220 IID更新至13320，L两view CE＋同输入原F0 KD；B加OE uniformKL，C再加OE raw-fused VICReg。L6300标签＋OE6134 outside-six额外成员监督，V27000只42组τ，novel3150只冻结后源评分；target/U_s0。
baseline保留cache同输入原F0，而非新encoder自身closed；完整训练6660步、Sinc/head梯度与buffer不变、曝光、七方法/两path/分组/逐TX/源1NN及资源边界见 `docs/PAMF_PHASE3_STAR_REPRESENTATION_RESULTS_20261029.md`。
本轮按事前C_K≥95%/R_U≥90%/原F0总损≤2pp判定，不追改；后续用户新C_K≥90%口径另见targets/metrics，未重算本轮τ。C相对B运行点H升4.0182pp，但A_K降0.0963pp、macroOSCR降0.9173pp、源1NN降13.2063pp，不能据H或接收率单项推进。
源1NN为已有五source RX间heldout TX结构诊断，不是A_N；A_N/H_OW=N/A，无DA/发现/注册。source成功也不意味着完整Phase3达成，控制不能替换C，下一假设未实施。
