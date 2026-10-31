# PAMF 冻结Sinc的目标原型骨干适配

PLANNED，无新性能。完整目标 ACTIVE。

Run `20261031-phase3-body-prototype-s2026092701-r01` / actualcode `a33612e978948da25bd733bcb9edba3ac9d74e9a`。SOURCE/TARGET_PCD各444，冻结Sinc48与buffers，body231985＋projection102688真实IQ训练；源192L/64OE exact matched，crossset/OE＋KD10，目标.1(global＋local)及K16/全部seed保持。原F0/defaulthead/G/r03不改。矩阵、祖先、成本、门槛与停止规则见`docs/PAMF_PHASE3_BODY_PROTOTYPE_PLAN_20261031.md`及experiment.json。

N607 `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，CWD `/home/szu2070436088/2510044040/CV-SincNet/releases/20261031-phase3-body-prototype-s2026092701-r01-release`，owner `codex-phase3-body-prototype-20261031`；command `/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python -B -u -m experiments.pamf_phase3_open_world.star_body_prototype_launch start --runtime /home/szu2070436088/2510044040/CV-SincNet/releases/20261031-phase3-body-prototype-s2026092701-r01-release/runtime.json`。SOURCE完成才开U，两模型冻→源GMM冻→源V四组六τ及old两组exact全冻→六列三DURA全pred→独立CPU3truth/Utruth0，source科学失败仍Q。完整goal ACTIVE，AN/HOW NA，无注册/正式RX访问。
