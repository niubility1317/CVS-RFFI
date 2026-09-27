# CVS：DAOT＋FastTrust-RC4原始residual_noeq五种子实验

- run_id：`20260927-phase1-cvs-daot-rc4-practical-manysig-m5-r01`
- group_id：`phase1-cvs-daot-rc4-original-practical-matched`；类别：`cvs`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：PLANNED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

用户确认的原始DAOT A1＋FastTrust-RC4。与对比共享物理L/U/V和五个模型种子，200轮final-only；关闭历史周期目标评分。后续接D92 E0真实256维v2。

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

## 本次版本与权限

使用原始 `rc4_practical_residual_noeq_20260918.json` 的DAOT A1＋FastTrust-RC4，保留全部算法参数。仅更换模型seed、输出路径，并禁用旧配置的周期目标评分和final weak reference。200轮后使用 `final_ssdg.pth`，独立评估复用对比实验固定clean／星地capsule。

五模型种子固定为392005、2026092701、2026092702、2026092703、2026092704；392005为历史优化参考，其余四种子单独汇总。不按目标结果选种子或重跑。源域实际物理L/U/V角色必须与对比契约完全相同。全部从零初始化，EMA仅来自本次student，无历史checkpoint、teacher、蒸馏或ground bundle继承。

原生CVS使用unlabeled_loader定义epoch，Ubatch256；对比训练每轮步数不同。因此本次为相同数据、种子与轮数，不能称等计算预算。原生信道随机生成器依赖模型seed/epoch/batch；receiver seed2027，均使用residual/post_sync/noeq，但不宣称跨方法训练信道逐样本抽样相同。

已通过CPU真实scratch checkpoint保存／严格重载、三场景信道、concat CE及DAOT反传、source V尾批检查。五配置解析与七项污染／版本负测通过。证据：`E:/type10-7/local_artifacts/cvs_matched_20260927/local_smoke/acceptance.json`。首次远端启动还执行一次真实checkpoint无query GPU smoke，随后直接训练。

D92后续固定 `P2-256-FULL`：D92 E0 identity160＋FFT96真实256维v2，非288维P2-FULL。ground统计只从本次最终模型与本次source L生成；注册使用固定residual_noeq support，query只读，独立truth-last评分。D92接线尚在实现，不将此状态写成已启动。
