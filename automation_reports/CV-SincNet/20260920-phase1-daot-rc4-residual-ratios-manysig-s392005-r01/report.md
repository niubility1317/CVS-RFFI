# DAOT＋FastTrust-RC4 residual环境比例六组

状态：LOCAL_VERIFIED，尚未启动。实际代码固定于`1e4984924ae804183545681e358bbc205a54145c`；逐行配置、完整命令、六类seed和存储位置以[experiment.json](experiment.json)为准。

用户最终范围：六个新训练（high/low_urban与mid/low_urban分别75:25、50:50、25:75），不重训旧配方对照；旧四checkpoint统一clean＋六个Practical环境测试。所有新训练保持seed392005、虚拟RX硬件seed2027、相同物理source L/U/V与200epoch，scratch初始化；旧权重仅评估。

只改变拼接分支场景组合/权重，DAOT内部student high、teacher mid/low_urban与其视图数和损失不变。原p=.3/.6/.8日程、E80卫星CE启用点、0.68卫星CE系数不变。比例是已应用的拼接星地视图条件比例，不是全部训练输入的比例。E91以后困难占拼接位置的理论比例为0.5×0.8×困难权重；其余还包含clean、简单以及DAOT内部视图。

执行加速统一启用：FIR核缓存、物理常量/配置复用、C递推、轻量训练元数据、恒等滤波短路、2个信道进程、输入/输出缓冲、clean验证复用、增量日志、identity-only评估、梯度快照、固定验证/测试视图磁盘缓存和双进程固定评估预取。训练视图保持动态，不跨训练batch预取。

E100起每10epoch统一评估clean和六个Practical环境，E200保留原LEO_WEAK最终参考。以新组E200 low_urban性能与clean退化为主要比较，同时完整列出六环境；epoch轨迹仅描述，禁止挑选目标最佳checkpoint或回流调参。单seed与地面代理数据限制保留。旧四checkpoint预算分别为full无均衡E150、ZF E130、MMSE E160、residual E200，不能冒充等预算四模型对比。

验证：实际parser与加权场景消费者测试、与原residual选项差异白名单、原Practical回归通过；本次独立P0/P1启动审查无问题。每row服务器启动前做scratch checkpoint真实模型无query smoke，启动后核对PID/CWD/命令、resolved配置、source_contract与日志增长。
