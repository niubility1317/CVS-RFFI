# PAMF第三阶段：RX24-1冻结开集拒识与DURA配对迁移

- run_id：`20261029-phase3-development-transfer-s2026092701-r01`
- group_id：`pamf-phase3-unseen-rx-practical-transfer`；类别：`diagnostic`；阶段：`Phase3-development-unseen-receiver-practical-frozen-transfer`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

Source-frozen E51 OE encoder/MLP/threshold on unseen RX24-1 development V864, paired original received and DURA. No fit, U or formal Q.

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

## 已固定的真实运行

code/config commit `a3b8bb24b6b9e357bf37678209cf0c21d666f7df`；10 tests＋16 subtests PASS，一次P0/P1 PASS。N607只读preflight PASS，源E51/聚合几何/MLP/原风险阈值与G frozen文件存在，新run/release未占用，八GPU各24124MB空闲（启动时再查实际容量），foreignstop0。主方法P1/MLP，P0及另外6分数为诊断，不根据开发成绩换选。

只预测V_dev864，U/formal/sourceIQ/逐条sourcebank0；先加载实际冻结checkpoint并对2条固定合成IQ做no-query smoke，PASS立即继续。新state/阈值更新0。P0为原received，P1为冻结DURA同输入恢复，分别以原F同输入closed作为已知保护基线；不读取clean参考。全部paired predictions.npz和complete固定后才独立score CLI连接private truth。score输出scores.json，不用query真值或未知数调阈值。

目标采用用户最新实用景跨RX总体/每景未知拒识90%，known接收率尽可能高且FRR≤5%、原F同输入open损伤≤2pp。数据scope只是新RX开发、novel3重复source开发设备，不是正式确认/DA/注册成功。技术错误/碰撞/越界/非法输入保留partial，不用低分重跑/改样本。远端普通短SSH在immutable release CWD执行predict/score，独立读回PID/CWD/argv/GPU/log与完成产物。实际命令/环境/输出详见experiment.json。

## 实际预测与评分完成（2026-10-29）

已VERIFIED/ANALYZED_DEVELOPMENT_ONLY，科学NO_PROMOTION。864记录全部固定预测后独立truth-last评分，主方法MLP保持，全部source状态与阈值更新0、U/formal读取0。实际checkpoint的2条合成IQ smoke在V读取前PASS，证据保存在完整predict.log和complete，不存在独立smoke.json，登记已改为真实artifact。

首次独立post已终态，未捕获运行期独立PID/CWD/GPU/log增长；CLI首行实际身份和完整日志/markers可用，但不冒充live证据，不为补观察重跑。当前快照其他GPU进程不当作本run；foreignstop0。实际代码`a3b8bb24b6b9e357bf37678209cf0c21d666f7df`，同row完整报告见`docs/PAMF_PHASE3_DEVELOPMENT_TRANSFER_RESULTS_20261029.md`，评分及post见local_artifacts/pamf_phase3_development_transfer_20261029_r01。目标仍ACTIVE，clean95/star90及known保护未同时达成。
