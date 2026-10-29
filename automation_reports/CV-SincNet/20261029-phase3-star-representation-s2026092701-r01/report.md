# PAMF第三阶段：星地开放集表征与防塌缩学习

- run_id：`20261029-phase3-star-representation-s2026092701-r01`
- group_id：`pamf-phase3-star-native-open-representation`；类别：`diagnostic`；阶段：`Phase3-star-source-representation`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Matched fixed2220-step original-F0 native increments: labeled anchor, plus source-membership OE, plus rawfused VIC regularization; no target optimization.

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

尚无结果。按预登记artifact逐项记录路径和缺项；保留每row与RX/day/TX/scene/K/seed的对应关系。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。

## 实际发布前固定

代码/config d319e3196f90113045a4f4f0958eeffe497f8785；实际runtime已校验cache及全部3组train/eval严格接口，唯一主oe_vic/P1/energy。41项unique聚焦测试＝data6+cache6+train/loss12+eval/score13+launch4，各subtest另记；data与科学候选各一次独立P0/P1 PASS，prepare四个cap/split接口问题已修正定向核对，未加额外gate。

数据OE6134、novel3150均G10 VALIDATED_ONCE，原L/V合同与原F0/G祖先沿用已核实身份。一次N607 preflight通过，各卡实际余量已记录；archive仅一次SHA比对/remote compile。缓存首步F0/G真实checkpoint合成星地P0/P1 smoke；每组训练首步native F梯度合成星地P0 smoke，不加载G，scope分开，PASS即继续。当前未声称真实训练/性能已经完成。

固定2220updates，所有新IQ星地/P1缓存；evalmode保持梯度，关闭BN/dropout/旧6forwardhook/As/CRIC/U/PL/EMA。全矩阵三组独立且共享合法输入，逐组终点直接源预测与独立truth评分；科学失败不改配方、不停其他健康row、不扩seed、不访问target。源通过仅下一冻结目标小矩阵的eligibility，不是跨RX/注册完成。
