# 在训源日志：联合梯度上限的实际执行

All complete lines of each captured persisted source gzip prefix; unfinished steps/rows not claimed complete; no target/scorer/stdout read

读回时间 Unix 1791754776.576394，覆盖 22 行已写入的 17540 条步骤，含 1108 个实际限幅辅助步。不是完整训练结果，未读取目标评分或调整在训任务。

|行|已写入步骤|辅助/限幅步|s均值/最小|限幅触发次数|限幅后各组最大比例|
|---|---:|---:|---:|---:|---|
|DELAY_M_SHALLOW-s2026092701|888|111/0|N/A|0|shallow:N/A, middle:N/A, rest:N/A|
|DELAY_M_MIDDLE-s2026092701|1554|277/0|N/A|0|shallow:N/A, middle:N/A, rest:N/A|
|DELAY_M_BOTH-s2026092701|811|91/0|N/A|0|shallow:N/A, middle:N/A, rest:N/A|
|DELAY_JOINT-s2026092701|634|47/0|N/A|0|shallow:N/A, middle:N/A, rest:N/A|
|CAP_M_SHALLOW-s2026092701|803|89/89|0.723631/0.126042|54|shallow:0.2, middle:0.2, rest:0.2|
|CAP_M_MIDDLE-s2026092701|1110|166/166|0.29598/0.0386181|154|shallow:0.2, middle:0.2, rest:0.2|
|CAP_M_BOTH-s2026092701|723|69/69|0.437488/0.0472711|56|shallow:0.2, middle:0.2, rest:0.2|
|CAP_JOINT-s2026092701|546|25/25|0.747819/0.25964|14|shallow:0.2, middle:0.2, rest:0.2|
|DELAYCAP_M_SHALLOW-s2026092701|807|90/90|1/1|0|shallow:0.115268, middle:0.137793, rest:0.0540562|
|DELAYCAP_M_MIDDLE-s2026092701|888|111/111|1/1|0|shallow:0.163836, middle:0.14769, rest:0.0750417|
|DELAYCAP_M_BOTH-s2026092701|723|69/69|1/1|0|shallow:0.096679, middle:0.155811, rest:0.0575907|
|DELAYCAP_JOINT-s2026092701|618|43/43|1/1|0|shallow:0.0416538, middle:0.0494948, rest:0.027138|
|DELAY_M_SHALLOW-s2026092702|666|55/0|N/A|0|shallow:N/A, middle:N/A, rest:N/A|
|DELAY_M_MIDDLE-s2026092702|972|132/0|N/A|0|shallow:N/A, middle:N/A, rest:N/A|
|DELAY_M_BOTH-s2026092702|739|73/0|N/A|0|shallow:N/A, middle:N/A, rest:N/A|
|DELAY_JOINT-s2026092702|596|38/0|N/A|0|shallow:N/A, middle:N/A, rest:N/A|
|CAP_M_SHALLOW-s2026092702|647|50/50|0.865082/0.2545|21|shallow:0.2, middle:0.2, rest:0.17681|
|CAP_M_MIDDLE-s2026092702|957|128/128|0.136851/0.00810721|124|shallow:0.2, middle:0.2, rest:0.2|
|CAP_M_BOTH-s2026092702|615|42/42|0.288694/0.0326947|37|shallow:0.2, middle:0.2, rest:0.2|
|CAP_JOINT-s2026092702|546|25/25|0.419779/0.0721846|21|shallow:0.2, middle:0.2, rest:0.179465|
|DELAYCAP_M_SHALLOW-s2026092702|666|55/55|1/1|0|shallow:0.0483973, middle:0.0372689, rest:0.018656|
|DELAYCAP_M_MIDDLE-s2026092702|1031|146/146|1/1|0|shallow:0.136926, middle:0.159453, rest:0.0674912|

完整分组、原始/限幅后比例、标量、求导耗时和源损失统计见 evidence/source_prefix_audit.json。校验覆盖参数数目、联合标量公式、实际加权损失及各组范数界。s变小只表明运行时缩放生效，不证明抑制了有害方向或提高了泛化。该上限不保证AdamW位移。无辅助梯度组/未执行限幅记N/A，不填0冒充有效观测。
