# CCOI-PA-V2需求—实现追踪表

| ID | V1证据/设计要求 | V2变更 | 验证 | 状态 |
|---|---|---|---|---|
| V2-01 | prediction中receiver全为`-1` | 递归读取嵌套`meta.rx_i` | 嵌套tensor/list单测；真实prediction receiver集合检查 | N607_VERIFIED_0_TO_11 |
| V2-02 | 无效receiver仍可进入scorer | runner与scorer双重拒绝负值/unknown | scorer负测；五行真实评分闭合 | N607_VERIFIED |
| V2-03 | 实际融合偏离设计公式 | 去中心RMS尺度对齐凸融合 | 公式单测；`alpha=0`基线等价 | LOCAL_VERIFIED |
| V2-04 | 旁路梯度被小alpha缩弱 | 算子分类器独立CE训练 | 聚焦训练单测与历史记录字段 | LOCAL_VERIFIED |
| V2-05 | alpha并列偏向初始值 | 网格包含0并列取最小alpha | 纯函数单测 | LOCAL_VERIFIED |
| V2-06 | 48码中单码占约49.5% | 有效码数与最大均值概率有界正则 | soft effective=`35.216/48`，但hard仅`4/48`且top=`70.52%` | SCIENTIFIC_FAILURE_HARD_COLLAPSE |
| V2-07 | V1 artifact schema无法表达scale | sidecar/manifest记录V2 schema、alpha、scale | 五行manifest、calibration和metrics真实读回 | N607_VERIFIED |
| V2-08 | V1总体链路健康 | 不改变Core90、source roles、C0–C4、四场景 | 35项回归；真实checkpoint smoke；16,320,000行prediction/truth | N607_VERIFIED |
| V2-09 | 科学增益未达门槛 | 同seed同row独立scorer复验 | 最佳LEO均值`+0.0095pp`、floor`+0.0361pp`，均未达`+0.30pp` | SCIENTIFIC_FAILURE_NO_PROMOTION |
| V2-10 | 禁止目标/query参与训练或校准 | `L_s/U_s/V_cal/V_select`边界保持不变 | 既有协议负测 | LOCAL_VERIFIED |

V2只处理以上十项。Soft-DTW、partial OT、多机制算子、状态子空间和多seed确认均为`NONBLOCKING`后续项，不得延迟本次最小矩阵。

最终实验为`PHASE1_CCOI_PA_V2_S20260824_20260825A`。工程闭环完整，但V2没有取得可晋级的分类增益；后续修复应直接作用于锐化后的近硬分配，而不是继续只约束soft均值分布。
