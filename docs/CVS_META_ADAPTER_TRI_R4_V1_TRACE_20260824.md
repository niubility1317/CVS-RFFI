# CVS_META_ADAPTER_TRI_R4_V1设计追踪表

状态说明：`specified`表示设计已固化但尚未实现；`pending`表示等待对应代码或实验验证。

|ID|设计来源|要求|目标文件／模块|状态|验证方式|备注|
|---|---|---|---|---|---|---|
|META-01|用户确认方案；设计规格§2|Phase1只使用source角色，Phase2严格`p2_min_v1/VALIDATED_ONCE`|`dataset_wisig.py`、Phase2 runner|specified|协议负测＋真实checkpoint无query smoke|严格实现|
|META-02|用户扩展query要求；设计规格§3～4|query支持类别、receiver、day/channel和LEO多层级任务|`meta_episodes.py`|implemented|`conda run -n ssr-gpu python -m pytest code/tests/test_meta_episode_sampler_v1.py code/tests/test_meta_ssl_split.py code/tests/test_meta_ssl_train_loop.py -q`：22 passed|五类domain关系、`capture_block`代理变量和固定seed采样已验证|
|META-03|设计规格§4.1|区分`Y_adapt`和`Y_guard`，不得把query-only类伪装成已适配类|`meta_episodes.py`、`meta_objectives.py`|implemented|Task5聚焦测试：mask隔离、重叠/长度/标签/shape负测、空集有限标量、adapt/guard计数均通过|已实现纯目标函数侧`Y_adapt/Y_guard`隔离；真实inner loop和Phase2更新仍待Task6/9|
|META-04|设计规格§2、§12|support/query按physical sample ID隔离，同物理样本视图不得跨边界|`dataset_wisig.py`、`meta_episodes.py`|implemented|上述GREEN命令：22 passed；Task2单测覆盖support/query物理ID互斥和视图隔离|WiSig物理样本ID与capture block代理metadata来自Task1；本任务完成episode侧联合隔离断言|
|META-05|设计报告adapter-BOIL路线；设计规格§5|time/freq/fusion插入rank-4残差adapter|`meta_adapter.py`、`model.py`、`meta_checkpoint.py`|implemented|Task3模型/邻近回归通过；Task4严格bundle由真实`build_model(model_args)`重建并逐键加载；legacy缺失集合仅为目标state中`meta_adapter_time/freq/fusion`三站点|Task3完成三hook；Task4不改变adapter结构或legacy默认路径|
|META-06|用户轻型快速要求；设计规格§5、§9|Phase2训练参数≤1%，固定3步，最多5步，硬上限40步|adapter参数白名单、`meta_checkpoint.py`、Phase2 runner|partial|Task4单测验证真实总参数计数、inner`down/up/gate`白名单、`log_step_size`冻结和>1%失败；Task6聚焦22项测试验证函数式固定步更新、clone隔离、只读fast映射、原模型参数/buffer不变及非有限保护；Phase2固定步数审计待Task9|Task6实现source诊断步数`0～10`；不实现Phase2 wrapper、optimizer或Phase2更新|
|META-07|设计报告元学习路线；设计规格§6|真实FOMAML内外循环和模块级Meta-SGD|`meta_inner_loop.py`、`meta_trainer.py`|implemented|`conda run -n ssr-gpu python -m pytest code/tests/test_meta_trainer_v1.py code/tests/test_meta_inner_loop_v1.py code/tests/test_meta_objectives_v1.py -q --disable-warnings`：70 passed；聚焦训练步验证4-episode平均outer loss、一次optimizer step、adapter和`log_step_size`白名单、非白名单参数/buffer不变、optimizer异常回滚|Task7 Fix Round1补齐Task2物理ID分区重验和显式source receiver allowlist；Task8 CLI、真实checkpoint训练与性能证据仍未完成|
|META-08|APSTA复盘；设计规格§7～8|用独立query outer loss学习support梯度泛化，记录梯度余弦|`meta_objectives.py`、`meta_trainer.py`|implemented|同上70项邻近回归通过；训练日志逐episode记录`episode_kind/k_shot/inner_steps/loss_adapt/loss_guard/loss_floor/grad_cos_support_query`，source curve固定输出A(0/1/3/5/10)，调用方已有grad和module training flags均精确恢复|outer loss仍只消费source`L_s`meta-query；`V_cal/V_select`仅生成丢弃式fast state，不执行optimizer或持久状态更新|
|META-09|项目Phase1协议；设计规格§6.1|复用ADV3B02 Core90 LEO_WEAK日程|Phase1 launcher/config|specified|配置单测和日志字段|严格复用|
|META-10|设计规格§8|checkpoint只由source meta-validation选择|`meta_trainer.py`、`meta_checkpoint.py`、selection summary|implemented|Task7聚焦测试覆盖source`V_cal/V_select`角色、显式receiver allowlist与target/query字段拒绝、固定A(0/1/3/5/10)curve、clean step0与guard floor门槛、从类型化source holdout曲线派生worst A(3)-A(0)及参数/延迟/ID决胜|选择器拒绝claimed与派生值不一致的候选；Task8/14负责CLI、真实checkpoint和完整clean+LEO评估，不在本任务提前声明|
|META-11|用户禁止D92头；设计规格§2、§9|不创建或训练协方差、LDA和持久分类头|模型、`meta_checkpoint.py`、`meta_objectives.py`和Phase2 runner|partial|Task3/4冻结与白名单证据保持；Task5测试验证fixed-head logits、冻结prototype、adapter-only L2-SP，并拒绝`cls_head/classifier/lda/cov/log_step_size`、伪嵌套/假站点/非法后缀；Task6测试验证fast state仅含`down/up/gate`且原模型head/base/buffer不变|本任务不创建optimizer、分类头、prototype参数、LDA或协方差状态；Phase2 runner侧审计待Task9|
|META-12|项目四状态规范；设计规格§9～10|显式输出`DA0_REG0/DA1_REG0`，Stage2-C接口保留`DA0_REG1/DA1_REG1`|Phase2 runner和scorer schema|specified|schema单测和同row配对|REG0新类指标为`N/A`|
|META-13|设计规格§10|Stage2-C先冻结旧类适配状态，再交给现有注册链|Stage2-C接口|specified|状态复用测试|V1不让新类support参与encoder更新|
|META-14|项目最小实验工作流；设计规格§12～14|RED→GREEN→邻近回归→真实smoke→一次审查→Git/N607发布→truth-last评分|tests、报告、release脚本|specified|逐阶段真实artifact|严格实现|
|META-15|用户晋级目标；设计规格§13|Target5达到mean`+1.0pp`且floor`+0.5pp`才进入Target25|Phase2 aggregate/scorer|specified|同row独立评分|低性能记科学失败|

## 当前一致性结论

- 设计报告核心路线已严格保留：元训练真实support更新、独立query outer优化、adapter-only内循环、模块级Meta-SGD和Phase2少步适配。
- 为满足不超过1%的运行时参数预算，三个rank-8 adapter被优化为三个rank-4 adapter；这是有依据的结构收缩，不是静默偏离。
- 设计报告中的二阶MAML、`z_dom`条件初始化、多专家、IQ输入adapter和新类support联合适配均明确延期，不属于V1缺失实现。
- 截至Task7 Fix Round1，`meta_adapter.py`、CVSincNet三站点hook、参数白名单、函数式一阶inner loop、纯outer objective、Phase1-B/C训练步及source-only选择器已有代码、单测和邻近回归证据；META-05、META-07、META-08、META-10为`implemented`，META-06与META-11仍为`partial`。
- Task7只实现机制和聚焦测试，不运行真实训练或改变Phase2权限；Task8/9/14的CLI、Phase2固定步数wrapper、真实checkpoint无query smoke、完整clean+三LEO评估、训练和性能证据仍未完成，当前不声明实验完成或性能结果。
