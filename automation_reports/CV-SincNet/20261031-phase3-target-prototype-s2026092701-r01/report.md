# PAMF 目标原型开放投影适配

PLANNED，无新性能。完整目标 ACTIVE。

Run `20261031-phase3-target-prototype-s2026092701-r01`，actualcode `35d41680b2039452562baa76d407e7d0ef0dacdb`。两独立训练臂SOURCE/TARGET_PCD各444；六列×三DURA唯一主PCD_predicted_ratio，固定公式、数据、预算、科学门槛和停止规则见 `docs/PAMF_PHASE3_TARGET_PROTOTYPE_PLAN_20261031.md`，逐row/六类seed在experiment.json唯一维护。

N607解释器 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/20261031-phase3-target-prototype-s2026092701-r01-release`，owner `codex-phase3-target-prototype-20261031`。命令 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -B -u -m experiments.pamf_phase3_open_world.star_target_prototype_launch start --runtime /home/szu2070436088/2510044040/CV-SincNet/releases/20261031-phase3-target-prototype-s2026092701-r01-release/runtime.json`。不可覆盖root `/home/szu2070436088/2510044040/CV-SincNet/runs/20261031-phase3-target-prototype-s2026092701-r01`。

Source-only r03初始化，冻结232033编码器，仅训练102688投影；TARGET另2560aux原型（K16事前容量非trueK）。source原crossset/OE+KD10，两armsource索引匹配；TARGET+.1(global+local)平衡原型强弱view，source控制在U打开前完成。原F0/defaulthead/G/r03不改，SOURCE无U祖先、TARGET及源聚合有U祖先；无V/NV/Q神经训练。两arm全冻后各sourceGMM和V四new6tau，old4af两old6tau exact，全部τ冻后NV和两Q全6x3预测，再独立CPU3truth/Utruth0，source科学失败仍全部Q。原基线和native-energy延后状态保持，无发现/匿名REG，AN/HOWNA。
