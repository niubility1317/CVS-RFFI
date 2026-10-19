# 纯 F rawAs E20：固定 L 面板源拟合诊断

状态：PLANNED。逐行矩阵、seed、输入与命令以 experiment.json 为准。

固定问题：把第一条 L 监督输入从 ordinary 改为 raw 后，RAW_AS 对训练期命名流重建观测、同 L 新信道及源 V 新物理样本的拟合是否改善？只读取 RAW_NOAS/RAW_AS 自身 scratch FINAL20；等本轮 source_matrix_decision 固定、两行 source completion/qualification 完成才运行。该诊断不增加资格门槛，不救援源判 FAIL 的配方。

固定面板：原诊断 20261019-diagnostic-F-observedChannel-e20-s2026092701-m4-r01 已冻结的 180 个 L physicalID，保留 B1 E20 第 4219 至 4440 步的实际 step/slot；不重新抽样。复用旧 B0/B1 源预测和元数据，不重复其神经前向。L6300/U56700（TX 隐藏）/V27000、RX1/3/4/6/8、day1/2/3、split392005 不变；IQ 读取前校验合法 L 成员及 base metadata。

按相同原训练 Config/session/seed/命名流重建训练期 As，再按原两个新命名流、三个原训练场景生成同 L 新观测。RAW_AS 核对实际训练消费 pid/step/slot 与每步 As128；没有新 RAW_AS 逐包 scene 历史记录或历史 As IQ，因此只称确定性重建。RAW_NOAS 没有训练过这些 As。原 180 面板与 scene 对齐，不能称作历史 IQ 字节核验。

骨干、默认分类头、EMA 和 optimizer 均 0 更新，冻结 state 前后完全相等；只补两个模型各 1440 个 L 前向包，首步真实 checkpoint 合法源单包 smoke 另 1，共 2881。源 V 复用本轮固定资格读出，0 新 V 前向；正式目标 IQ、预测与 truth 均不读取。报告 accuracy、默认头 CE（区分训练 smoothing0.01）、n、场景/RX、修正/新增错误、raw 正确保留及观测/新信道差；n=1 seed，不称重复收益或物理上限。

普通 N607 既有 Python3.10/Torch2.1、CPU2、FP32/TF32off，GPU6 单诊断进程，含其他 owner 每 GPU 最多 2 进程且空闲至少 12000 MB；沿用共享 GPU 锁。不干预健康训练。唯一 owner codex/root/F-rawAs-sourceFit20。CWD PROJECT/releases/cvs_F_rawAs_E20_20261019_r01；命令见 experiment.json。脚本 analysis/cvs_pure_F_rawAs20_source_fit_20261019.py 位于独立发布目录 PROJECT/releases/cvs_F_rawAs_sourceFit_E20_20261019_r01。输出根 PROJECT/runs/20261019-diagnostic-F-rawAs-sourceFit-e20-s2026092701-m2-r01 首次独占创建；保存配置、metadata、重建核对、source_smoke、源预测、analysis、complete/failure 及日志。

角色/来源、非有限值、模型状态变化、流/接口或输出矛盾即技术停止并保留现场，不自动重启。仅进行窄协议负测、一次 P0/P1 delta 审查、一次归档传输比较、远端编译与启动后读回；首步真实 checkpoint 源 smoke 通过后直接继续，不增加封印或逐文件哈希。

本地验证 PASS：py_compile、原180条真实metadata纯函数核验，以及非法L角色、pid不匹配、越界base_index、非E20 step四项在IQ访问前拒绝。唯一一次独立P0/P1 delta审查 PASS；仅覆盖此次源读出直接缺陷，无远端执行。预检确认新run/release/archive均不存在，GPU6空闲24124 MB；源判定及RAW_AS终轮尚未就绪，继续等待。
