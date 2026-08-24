# CCOI-PA-V2需求—实现追踪表

| ID | V1证据/设计要求 | V2变更 | 验证 | 状态 |
|---|---|---|---|---|
| V2-01 | prediction中receiver全为`-1` | 递归读取嵌套`meta.rx_i` | 嵌套tensor/list单测；真实prediction receiver集合检查 | LOCAL_VERIFIED |
| V2-02 | 无效receiver仍可进入scorer | runner与scorer双重拒绝负值/unknown | scorer负测 | LOCAL_VERIFIED |
| V2-03 | 实际融合偏离设计公式 | 去中心RMS尺度对齐凸融合 | 公式单测；`alpha=0`基线等价 | LOCAL_VERIFIED |
| V2-04 | 旁路梯度被小alpha缩弱 | 算子分类器独立CE训练 | 聚焦训练单测与历史记录字段 | LOCAL_VERIFIED |
| V2-05 | alpha并列偏向初始值 | 网格包含0并列取最小alpha | 纯函数单测 | LOCAL_VERIFIED |
| V2-06 | 48码中单码占约49.5% | 有效码数与最大均值概率有界正则 | 集中/分散合成分布单测 | LOCAL_VERIFIED |
| V2-07 | V1 artifact schema无法表达scale | sidecar/manifest记录V2 schema、alpha、scale | manifest字段和真实artifact | LOCAL_VERIFIED |
| V2-08 | V1总体链路健康 | 不改变Core90、source roles、C0–C4、四场景 | 全部CCOI回归、真实checkpoint smoke | PENDING_N607_SMOKE |
| V2-09 | 科学增益未达门槛 | 同seed同row独立scorer复验 | clean、三LEO、receiver-floor、holdout、q对照 | PENDING_N607_MATRIX |
| V2-10 | 禁止目标/query参与训练或校准 | `L_s/U_s/V_cal/V_select`边界保持不变 | 既有协议负测 | LOCAL_VERIFIED |

V2只处理以上十项。Soft-DTW、partial OT、多机制算子、状态子空间和多seed确认均为`NONBLOCKING`后续项，不得延迟本次最小矩阵。
