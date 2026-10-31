# PAMF 公共类传输对齐

PLANNED，无新性能，完整目标ACTIVE。

Run `20261031-phase3-common-transport-s2026092701-r01` / actualcode `4038105cb7f7c2defe75eb7fbfa825e7f932b40f`。只新增TARGET_CCD444；源r03初始化，固定Sinc/buffers/F0/defaulthead/G，保留sourceKD10及PCD.1，step101加入固定源六原型adaptive-UOT CCD.1。旧SOURCE/PCD训练a336/r01、六列预测及统计4273/r02精确复用。新known90/OE32源fit1，V两组六τ12新值/旧36exact全冻，再八列三DURA全部预测后CPU三truth；Utruth0。完整公式/数据/祖先/三对照晋级/成本/停止规则见`docs/PAMF_PHASE3_COMMON_TRANSPORT_PLAN_20261031.md`，六类seed/row在experiment.json。

Interpreter `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`；CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/20261031-phase3-common-transport-s2026092701-r01-release`；owner `codex-phase3-common-transport-20261031`；command `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -B -u -m experiments.pamf_phase3_open_world.star_common_transport_launch start --runtime /home/szu2070436088/2510044040/CV-SincNet/releases/20261031-phase3-common-transport-s2026092701-r01-release/runtime.json`。新root `/home/szu2070436088/2510044040/CV-SincNet/runs/20261031-phase3-common-transport-s2026092701-r01`。科学低分仍完成预登记全测试，无formalRX/发现/匿名REG，A_N/H_OW=N/A，原主基线保持。
