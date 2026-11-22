# LoRa DURA 在保护共享形状的条件下重训

源域证据表明仅修改六条 chirp 共享的非载波形状会净损伤识别，推理前投影后净纠正高于完整 G。只在推理时投影仍会丢弃原恢复器已经花容量拟合的成分。这里固定比较训练时加入同一保护投影，考察把可学习容量放到允许修正的成分后，独立星地识别能否提高。

三个 G 各从随机初始化开始，沿用原全量 DURA 的相同骨架、种子、1600 更新、batch 24、L/U 源 IQ、采样顺序、门槛、优化器、配准和误差目标。固定同一源 IQ 旁路、同一 1269 个 L 和 1537 个 U 合格输入，并保留原剂量留出的 U。唯一改变是在计算训练误差前对恢复量做共享形状保护投影。增量上限与该 G 投影前的受限增量相同。无需重新拟合剂量器。G 只读源 L/U IQ，不读 V 身份、识别器、任何目标。冻结 G 后，源波形审计和接入识别器分别执行。

预登记的四路为直通、原固定 G、同一原 G 加推理投影、从训练起加投影的 G。四路连接同种子第 168 轮已冻结 Tri，固定 30000 个包、15 个未见接收机、clean 与六景 practical，共 84 个评分单元。记录真值最后评分、各景净变化、逐 TX/RX、clean 激活及误改。训练过程保存同批直通误差、应用恢复后的误差、真实步数与支持记录。

N607 普通账户、既有 LoRa Python，发布 `python -m experiments.pamf_lora_main.full_dura_publish --projected` 并传原数据、G 与第 168 轮 F 的 release receipt。CWD 为 `CV-SincNet/releases/20261122-phase1-lora-dura-train-protected-m3-r01-release`，输出为 `CV-SincNet/runs/20261122-phase1-lora-dura-train-protected-m3-r01`。分布在 GPU0–6，每卡最多一个本 run worker，避免 GPU7。发布器锁定实际 Git commit。正式生产先跑真实 G 的纯源 IQ smoke。数据/祖先、训练出现非有限值、清洁旁路或盲输出计数错误时停止派发。健康任务保留，低分不提前停止。冻结 84 单元后由独立评分程序核验并报告。
