# CVS_META_ADAPTER_TRI_R4_V1设计追踪表

状态说明：`specified`表示设计已固化但尚未实现；`pending`表示等待对应代码或实验验证。

|ID|设计来源|要求|目标文件／模块|状态|验证方式|备注|
|---|---|---|---|---|---|---|
|META-01|用户确认方案；设计规格§2|Phase1只使用source角色，Phase2严格`p2_min_v1/VALIDATED_ONCE`|`dataset_wisig.py`、Phase2 runner|specified|协议负测＋真实checkpoint无query smoke|严格实现|
|META-02|用户扩展query要求；设计规格§3～4|query支持类别、receiver、day/channel和LEO多层级任务|`meta_episodes.py`|implemented|`conda run -n ssr-gpu python -m pytest code/tests/test_meta_episode_sampler_v1.py code/tests/test_meta_ssl_split.py code/tests/test_meta_ssl_train_loop.py -q`：22 passed|五类domain关系、`capture_block`代理变量和固定seed采样已验证|
|META-03|设计规格§4.1|区分`Y_adapt`和`Y_guard`，不得把query-only类伪装成已适配类|`meta_episodes.py`、`meta_objectives.py`|implemented|上述GREEN命令：22 passed；覆盖部分类模式、guard路由和标签置换计数不变|本任务仅实现episode侧路由，`meta_objectives.py`仍由后续Task实现|
|META-04|设计规格§2、§12|support/query按physical sample ID隔离，同物理样本视图不得跨边界|`dataset_wisig.py`、`meta_episodes.py`|implemented|上述GREEN命令：22 passed；Task2单测覆盖support/query物理ID互斥和视图隔离|WiSig物理样本ID与capture block代理metadata来自Task1；本任务完成episode侧联合隔离断言|
|META-05|设计报告adapter-BOIL路线；设计规格§5|time/freq/fusion插入rank-4残差adapter|`meta_adapter.py`、`model.py`|specified|形状、初始化和forward回归|由rank-8优化为≤1%的rank-4|
|META-06|用户轻型快速要求；设计规格§5、§9|Phase2训练参数≤1%，固定3步，最多5步，硬上限40步|adapter参数白名单、Phase2 runner|specified|真实checkpoint参数计数和step审计|严格实现|
|META-07|设计报告元学习路线；设计规格§6|真实FOMAML内外循环和模块级Meta-SGD|`meta_inner_loop.py`、`meta_trainer.py`|specified|梯度更新、可复现和有限差分测试|一阶近似|
|META-08|APSTA复盘；设计规格§7～8|用独立query outer loss学习support梯度泛化，记录梯度余弦|`meta_objectives.py`、`meta_trainer.py`|specified|source meta-validation曲线|严格实现outer目标；余弦仅诊断|
|META-09|项目Phase1协议；设计规格§6.1|复用ADV3B02 Core90 LEO_WEAK日程|Phase1 launcher/config|specified|配置单测和日志字段|严格复用|
|META-10|设计规格§8|checkpoint只由source meta-validation选择|`meta_trainer.py`、selection summary|specified|target query不可达负测|严格实现|
|META-11|用户禁止D92头；设计规格§2、§9|不创建或训练协方差、LDA和持久分类头|模型和Phase2 runner|specified|state dict/参数白名单审计|严格实现|
|META-12|项目四状态规范；设计规格§9～10|显式输出`DA0_REG0/DA1_REG0`，Stage2-C接口保留`DA0_REG1/DA1_REG1`|Phase2 runner和scorer schema|specified|schema单测和同row配对|REG0新类指标为`N/A`|
|META-13|设计规格§10|Stage2-C先冻结旧类适配状态，再交给现有注册链|Stage2-C接口|specified|状态复用测试|V1不让新类support参与encoder更新|
|META-14|项目最小实验工作流；设计规格§12～14|RED→GREEN→邻近回归→真实smoke→一次审查→Git/N607发布→truth-last评分|tests、报告、release脚本|specified|逐阶段真实artifact|严格实现|
|META-15|用户晋级目标；设计规格§13|Target5达到mean`+1.0pp`且floor`+0.5pp`才进入Target25|Phase2 aggregate/scorer|specified|同row独立评分|低性能记科学失败|

## 当前一致性结论

- 设计报告核心路线已严格保留：元训练真实support更新、独立query outer优化、adapter-only内循环、模块级Meta-SGD和Phase2少步适配。
- 为满足不超过1%的运行时参数预算，三个rank-8 adapter被优化为三个rank-4 adapter；这是有依据的结构收缩，不是静默偏离。
- 设计报告中的二阶MAML、`z_dom`条件初始化、多专家、IQ输入adapter和新类support联合适配均明确延期，不属于V1缺失实现。
- 当前尚无代码、训练或性能证据，所有实现与实验状态保持`pending`。
