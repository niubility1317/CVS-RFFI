# PAMF第三阶段：21台源辅助设备数据

- run_id：`20261029-phase3-source-oe-diverse-data-r01`
- group_id：`pamf-phase3-source-diverse-oe-data`；类别：`data`；阶段：`Phase3-source-21-device-auxiliary-exposure-data`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：ANALYZED；真实数据VERIFIED/VALIDATED_ONCE，无模型性能结果。

## 目的与对照

Full中compact外24TX保留3novel，其余21训练备选在原5RX×3day每cell最多32、少量全取，6134唯一source记录；CPU纯数据，不读目标或模型。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

6134条源辅助数据已完成且独立VERIFIED；完整[数据文档](E:/type10-7/docs/PAMF_PHASE3_SOURCE_OE_DIVERSE_DATA_20261029.md)和[实际摘要](E:/type10-7/local_artifacts/pamf_phase3_source_oe_diverse_data_20261029_r01/analysis_summary.json)可核查。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。

## 版本与本地验证

实际 code/config commit：`2f41fc2835798f583c54900ab4c791d762d0101c`。5项相关测试、5个subtests通过；一次新数据候选P0/P1审查PASS。独立21辅助source TX（含原4），cap32少量全取，预期6134记录／220非空95空；不按IQ/score挑选。原Full158PKL验证复用，15source容器每个只读一次，无新rawhash/CRC/20ZIP扫描。source L/U/V、novel3、正式unknown4不进入输出；不载F/G、channel、训练、预测或评分，因此模型checkpoint smoke不适用。CLI实际PID/CWD/argv JSON与原生Win32_Process/log-growth读回核对，输出不可覆盖，CPU-only/新增GPU0/其他进程保持。未来domain加权readout不被本数据builder消费，尚在独立实现，不作为本次额外门槛。

## 实际构建完成

6134条、220非空／95空cell、15source容器，18行／1467B完整日志无异常；VALIDATED_ONCE句柄保留，原L/U/V／novelIQ／F／channel／formal target／GPU／foreignstop0。只有数据构建完成，不宣称模型拒识／适应／注册成功。
