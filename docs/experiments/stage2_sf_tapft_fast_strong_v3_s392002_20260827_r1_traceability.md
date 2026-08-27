# SF-TAPFT Fast-Strong V3设计追踪

来源：用户提供的《SF-TAPFT Fast-Strong V3》设计报告，2026-08-27。状态：`IMPLEMENTED_AND_EXPERIMENTALLY_CLOSED`。

|ID|设计要求|状态|落地与证据|
|---|---|---|---|
|FSV3-01|固定目标head、rho=0.5、scale=8、balanced CE、LOO=0.5、L2-SP=1e-4|verified|H0–H4/H6矩阵固定；H5才单独启用rho_c|
|FSV3-02|禁用Adapter/完整t3/frequency/domain；H6用S02结构|verified|H6仅训练head+t3.norm(weight+bias)，1152元素|
|FSV3-03|H0复跑300/ref4500/warmup0.05|verified|H0闭合，OOF BA81.9445%，选择250步|
|FSV3-04|H1/H2追加150/300步tail|verified|H1/H2闭合；H2未选择额外tail|
|FSV3-05|适配后缓存embedding做head-only polish|verified|H3/H4/H5/H6只有一次cached embedding forward|
|FSV3-06|EMA beta=0.99只覆盖许可delta|verified_negative|H4 BA/NLL回退，不晋级|
|FSV3-07|实现类自适应rho_c公式|verified_negative|H5 rho_c完整审计；OOF无收益|
|FSV3-08|可靠性head anchor lambda=0.01|verified_negative|H5同row测试无收益|
|FSV3-09|M02 OOF teacher不进入每次部署|deferred_as_designed|本轮无teacher依赖|
|FSV3-10|OOF温度不改变argmax，不能救欠拟合|verified|全矩阵温度仅作support诊断；H0–H5未因NLL晋级|
|FSV3-11|保留向量化LOO、稀疏validation、KD=0优化和许可snapshot|verified|81项聚焦测试及真实矩阵通过|
|FSV3-12|full-support单次refit和<10KB delta bundle|verified|H6 delta=4500B；严格loader等价测试通过|
|FSV3-13|只做H0–H6最小矩阵，不做全排列|verified|7/7行闭合，无额外组合|
|FSV3-14|BA/floor/NLL晋级门槛|verified_no_promotion|H6 Q180 NLL通过，BA/floor未通过|
|FSV3-15|约1500变化元素、分钟级、推理不增算|verified_with_limit|H6 1152元素、研究墙钟69.29s；GPU峰值未连续捕获|
|FSV3-16|真实最大Q180 DA0_REG0/DA1_REG0对比|verified|support冻结H6后才预测；Q60+Q120零重叠，180条truth-last闭合|
|FSV3-17|排除rho=1、Adapter、完整t3、P4、KD0.1、ref300、head-first|verified|均未进入正式矩阵|
|FSV3-18|完整记录实现、全矩阵、逐类结果、资源、失败并发布|verified|正式报告已更新并推送至GitHub治理分支|

## 设计偏差与解释

- 报告提出“温度只在高性能候选后使用”。为保持H0–H6概率指标可比，本轮对所有候选计算support-only诊断温度，但它不改变argmax，也不改变BA/floor晋级；只有support侧冻结的H6进入Q180。
- H2预设600步、H3–H5预设520步，但OOF选择允许回到更早的三段步数；表中同时保留预设上限和实际选择，未把未选中步数写成实际部署计算。
- 最大Q180只验证support侧已冻结的H6，而不是用query在H0–H6中选模型，避免query驱动候选选择。
