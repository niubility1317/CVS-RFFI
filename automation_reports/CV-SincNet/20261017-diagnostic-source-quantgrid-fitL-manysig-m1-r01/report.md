# 接收观测格点：源域固定反证

状态 PLANNED／本地实现完成，未读取真实IQ。run_id=20261017-diagnostic-source-quantgrid-fitL-manysig-m1-r01。当前F＋跨RX同TX／defaultcos30正常联合训练／E20保持；本次不训练、不载入模型或checkpoint、0NN/优化/头拟合/IQ变换/target。

当前唯一假设：原eq1＋入网前RMS归一化的源IQ仍存在足够可观测、跨TX/day稳定且有足够幅度的轴向仿射格点，值得单独检验接收侧去格点观察，而非继续增加损失/分支。官方WiSig说明提供原始和均衡后两种256点表示：[数据集官方说明](https://cores.ee.ucla.edu/wisig/overview/)。当前实际wisig_equalized=1、native normalize=True；不能把它称为原ADC码、根据假定位数加噪，或在已处理IQ上套用subtractive-dither独立误差定理。

最小实验仅原fitL5040、RX3/4/6/8、day1/2/3、6TX、72cell各70，无U值或真值/V/heldRX1/target。复用既有VALIDATED_ONCE source_cache；载入旧carrier L6300和未用V特征，但只消费精确effective角色ID的5040x2x256；旧features不用。数据路径/角色/seed/命令和逐row唯一配置以experiment.json为准，不改split或重新校验数据。

每包测量I/Q轴共同 affine grid：候选来自相邻distinct值差16最小＋17分位除1..32，最大选取残差99分位≤.02格距且可分辨于float32的格距；最小相对格距2^-16、float32分辨界64eps×maxabs、最大相对格距.5、distinct≥32。该格点可能来自存储/处理，不能唯一归因RX硬件。未测格点的包完整保留，不做可观察子集救援。

先要求全部72cell可观测比例均≥95%；满足后才用cell log(q/componentRMS)中位数做18个leaveTX/day描述性RX条件均值风险比较（非身份/RX分类器，无参数化头），至少15/18正折且汇总平方风险降低≥20%；全体可测包等效独立均匀dither SNR中位数≤50dB。最后一个只是原先固定的候选幅度可辨理由，不证明实际去量化误差或必要增强强度。全部满足只支持另登记源身份安全检验；任一失败关闭该固定可观察且有幅度的轴向格点理由，不调候选/容差/窗/RX/seed/阈值补救，不自动E20。

合成24个已知格距/增益/偏置/归一化正例全部正确，36连续64/32bit和周期负例0误报；13协议反例/compile PASS，0真实IQ/CP/NN。开发阶段JSON np-scalar打印错误与coarse-grid上界alias保留，均在源读取前修复，不属于科学选择。第一次实际运行先合成noquery estimator smoke立即继续source；无模型CP因此不伪称真实CP前向smoke。唯一限定P0/P1正在进行，无额外hash/receipt/gate。

固定CPU2/CUDA隐藏、现有N607普通账户环境、一份release archive/一次SHA比较/compile、独占不可覆盖run根和owner；一次预检/启动后PID-CWD-argv-loggrowth。技术失败保留产物，不自动重跑或干预其他owner。完整5040逐包/72cell、条件crossfit、全量独立复算、实际cost与图、登记/Git交付；原始IQ/逐包ID/weights/logs/图和机器状态不入Git。正式测试与完整目标保持，当前不证明新Ag、联合或跨seed收益。
