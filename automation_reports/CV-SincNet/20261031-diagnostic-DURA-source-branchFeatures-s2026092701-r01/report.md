# 恢复后身份特征损伤：原头四分支源域替换诊断

- run_id：`20261031-diagnostic-DURA-source-branchFeatures-s2026092701-r01`
- group_id：`phase1-dura-source-additive-branch-features`；类别：`diagnostic`；阶段：`Phase1`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

复用defaultDURA/pureG源512×01/20 native IQ，分解实际F160四个加性分量，逐分支换clean分量后接原默认头。PA唯一主候选，其他分支与pureG只描述。0G/训练/TX-V-target。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

完整SOURCE_ONLY结果见analysis.md/results.json；本次没有新G训练或正式预测。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。


## 固定源域四分支诊断

原F50/defaulthead的160维特征实际等于基础identity、加权PA、加权reference response及加权TripleSIG四项之和。用实际模块hooks获取，不复制替换骨干，不拟合分类头；每次forward后清理hooks。逐分支把恢复后分量换成同源clean分量，再接同一原头；all_clean只作恒等正控制，不称真实TX100%或恢复上限。

固定源512（128L/384U）×01/20，默认DURA为主、pureG为描述对照；读取旧quarterOrbit产物中的native IQ[0]，不读失败的orbit平均。PA为唯一主候选，两视图都须与clean F选择一致数至少净增6/512，五个源RX不损失，每非clean场景最多少1，clean场景不少，才支持另登记源IQ可达性。其他分支与pureG不能替选；失败不放宽或搜索。PA路径仍可含接收机/信道信息，不等于真实PA硬件参数。

3项聚焦检查、实际入口和唯一P0P1通过。首步真实2Ls原F分解/native缓存/原默认头smoke通过后立即继续。主F2560、显式head-only12800，smokeF4/head4，0G前向/权重读取/优化/TX/V/target；全部2048条恢复观测、四组件/六案例/原头logits与分组独立复算。这里只是源域潜在特征oracle，不保证对应IQ存在，不启动新G或正式预测。默认DURA与完整目标保持。


## 完整源诊断结果

VERIFIED / ANALYZED，SOURCE_ONLY。固定defaultDURA主对象/pureG描述、source512（128L/384U）×缓存01/20；CPU2主循环2.426344秒自然结束。主PA替换clean分量后与clean F选择一致数净增10/3，第二视图不足事前双视图各+6且RX3净−1，主科学判定FAIL，不放宽、不改主。描述性的时序签名替换净增21/11，五源RX均正；它仅定位源特征损伤，不自动晋级或直接启动新G。F主2560/显式原头12800，smokeF4/head4，0新G/权重加载/优化/TX/V/target；F状态不变。2048恢复观测、12800原头特征行及104分组独立复算。默认DURA与完整目标保持。

详见[完整分析](analysis.md)。执行ac309dd9540295bfc2456fc697258b07c1f9a1f3，PID1667506已自然结束。PA失败保持，不以其他分支替选，不以特征oracle当作可部署IQ。
