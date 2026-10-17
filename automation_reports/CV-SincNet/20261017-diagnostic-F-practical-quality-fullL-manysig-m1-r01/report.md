# 完整源 L：practical 接收质量耦合检验

状态：已实现、已登记，未启动。当前唯一假设：white-input 期望功率与有限窗 H*x 功率的差异，实质改变原接收锁定／残差补偿。不是已确认的缺陷，不预告身份识别收益。

全部原 L6300、源 RX1/3/4/6/8、day1/2/3、6TX、split392005，无训练 RX 留出。现有 exact-match source_cache 只用 L 的 x/y/rx/day/ids；U 只读 opaque 排除 ID，V 和旧父派生特征反序列化后不用。零身份模型、CP、NN、训练／EMA／头拟合／target。E30 训练预算保持，此检验不追加训练。

每个 L 在原 low suburban／mid urban／low urban 三个训练场景各取一次固定 seed，共 18900 个配对。两种质量口径共享同一 raw IQ、h、几何、状态、虚拟 RX 和参考噪声方差。质量变化可依既有规则共同改变锁定、频率／相位跟踪与 AGC。两个输出各自从同一个 raw 开始，不连续叠加信道，不修改测试 received IQ、SNR 或样本。波形功率包含原采集噪声，不声称真实总 SNR。

预定物理判据：lock 状态改变或 residual scale 比值至少 1.10，整体占比 ≥5%，且每个源 RX ≥3%。失败关闭固定方向，不改阈值／窗口／SNR／seed 救跑；通过只说明物理近似相关，后续须由同预算默认头识别证明，不记为 Ag、As 或联合成功。

本地已验证项目 Python3.10.0／NumPy2.2.6：19 协议负例、2 解析功率例、24 合成实际引擎配对／48 residual 数值调用 PASS；0 真实 IQ／CP／NN／更新；7 个 Python 文件 compile PASS。一次独立限定 P0/P1 审查 PASS。发布 first-step 再在 pinned runtime 执行同一合成数值 smoke 后立即继续源检验。无模型消费，真实 CP smoke 不适用，不额外创建模型。

owner codex/root/F-practical-quality-source；唯一不可覆盖 output PROJECT/runs/20261017-diagnostic-F-practical-quality-fullL-manysig-m1-r01，CWD PROJECT/releases/cvs_F_practical_quality_fullL_20261017_r01。命令 CVS-RFFI/bin/python -u experiments/cvs_practical_quality_source17/runner.py；N607 普通账户 CPU2、CUDA 为空、无 GPU 分配。执行 Git commit 写入 release_commit／resolved_config；代码依赖固定训练 R03 practical runtime。一个 N607 preflight、一个归档与一次传输核对、远端 compile、PID/CWD/argv/CPU／日志增长按原最小流程执行。

停止规则：数据角色／来源不匹配、非有限或缺失记录、不可覆盖输出冲突、系统技术失败。保留产物，不自动重试，不影响其他任务。产物：全部 18900 physical_pairs、RX/TX/day/scene/state 分层、result／completion、实际模块路径与日志、同 row 报告／登记。当前没有目标预测或身份准确率。
