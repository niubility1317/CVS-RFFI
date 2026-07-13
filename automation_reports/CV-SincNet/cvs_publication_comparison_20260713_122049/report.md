# CVS论文级方法对比实验Git交接

- 根报告：`E:\type10-7\automation_reports\CV-SincNet\cvs_publication_comparison_20260713_122049\report.md`
- 追踪表：`analysis/cvs_publication_comparison_traceability_20260713.md`
- 统一协议：`docs/CVS_PUBLICATION_COMPARISON_PROTOCOL_20260713.md`
- 当前阶段：Phase1 seed713101三个baseline仍在训练；Stage2-C K5 seed713101中CSIL和MoPC-HR已完成，Orthogonal仍在运行；Stage2-B三种监督DA的低步数smoke已完成。
- Stage2-C已实现CSIL、MoPC-HR、Orthogonal Incremental统一runner、三种LEO测试、固定query的seeded nested K-shot split、sample score及四层详细统计。正式K5锚点已完成的CSIL/MoPC-HR分别得到`H_old_new=0.1808/0.2459`，单seed且Orthogonal未完成，禁止作正式排序。
- Stage2-B已实现ProtoNet CDA、MRIOR-SDA、DADDA-SDA统一监督runner。每个run只适应一个target receiver，仅有标签target-old support可训练；query只评估。三方法smoke均输出360条score、57条四层明细且全测试星地增强。
- 监督DA聚焦回归`10 passed`；新增runner提交`c157754`。N607同步hash、remote py_compile和dry-run均通过。
- 声明边界：smoke只证明机制与artifact契约；Phase1终局详细后评估、Stage2-C完整K/seed矩阵、Stage2-B五接收机K/seed矩阵和CVS同协议结果仍未完成，因此不构成论文最终性能结论或部署成功证据。

## CVS同协议入口与训练预算决议（13:50）

- Phase1当前CVS候选不是普通监督基线：`phase1_dgleo_jointp0_leoweak8r2_20260713`以`ADV3B02_CORE90_SOFT_E200`为初始化，使用`rho_label=0.08`有标签与0.72源域无标签，包含三种星地信道训练视图和source-val-only选择。论文表必须显式报告其额外无标签访问预算，不能把它写成仅算法结构差异。
- Phase2提出方法固定为两个与项目谱系一致的入口：Stage2-B使用冻结CVS特征上的`CVS-OPGAC`监督support-only原型高斯校准；Stage2-C使用冻结`ADV3B02_CORE90_SOFT_E200`特征上的`CVS-qKNNV42`。后者参数固定为int8 support code、类内top-1、prototype权重0.45、old anchor0.001、8轮support-clamped label propagation权重0.025；unknown拒识不进入Phase2主线。
- 新增`paper_reproduction/cvs_aligned/cvs_method_runner.py`，强制单target receiver、三种正式LEO缓存、`seeded_nested` K={1,2,5,10,20}、support pool maxK=20后固定query、query标签不训练/不选模、sample score和四层明细、有限support-fit trace。
- Stage2-B与Stage2-C正式矩阵均从3个对比方法扩为“CVS+3个对比方法”，每阶段为`4方法 x 5接收机 x 5K x 5seed=500`行。新dry-run manifest分别生成500行。
- 训练预算采用双层报告：主表使用CVS任务下的common-budget，以控制训练计算量并保证同一数据、K、seed、receiver和query配对；论文原生epoch/batch只作为方法谱系敏感性附表，不作为主表直接混排。该选择避免Orthogonal的100/50 epoch与CSIL/MoPC短训练配置造成计算预算不等，但主表必须标为CVS extension而非论文原始结果。
- 本地验证：`cvs_method_runner.py`与matrix worker py_compile通过；新增CVS runner与matrix聚焦测试`4 passed`。当前尚未生成三场景ADV3B02正式feature cache，也未启动500行矩阵。

## ADV3B02正式feature cache导出计划（13:48）

- 目标：为CVS-OPGAC与CVS-qKNNV42生成同一冻结`ADV3B02_CORE90_SOFT_E200`、同一原始样本集合的三份星地场景feature cache；每份包含clean source、五个target receiver的target-old与两个target-new TX。
- 本地脚本：`paper_reproduction/scripts/export_cvs_publication_adv3b02_features_20260713.sh`；本地`bash -n`和`--dry-run`通过。导出只做冻结模型前向，不训练、不更新checkpoint。
- checkpoint：`runs/phase1_adv3_mechanism32_queue_20260701/ADV3B02_CORE90_SOFT_E200/best_joint_safe_ssdg.pth`；特征`z_id`；ManySig old TX索引0-5；ManyTx new TX=`1-16,1-18`；target receivers=`20-1,3-19,7-14,7-7,8-8`；target day0；每TX最多400条。
- 三个并行短任务分别使用GPU3/4/5，输出`runs/cvs_publication_adv3b02_feature_cache_20260713/{leo_clear_weak,leo_low_elev_weak,leo_rain_weak}.npz`，日志位于`paper_reproduction/logs/cvs_publication_adv3b02_feature_cache_20260713/`。
- 精确服务器入口：`bash paper_reproduction/scripts/export_cvs_publication_adv3b02_features_20260713.sh`，环境`/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python`，工作目录`/home/szu2070436088/2510044040/CV-SincNet`。
- 成功条件：三份NPZ非空；target-old/new行的`sat_scenarios`分别全部等于对应场景；五个receiver每个old/new TX至少包含40条以满足maxK20+query20；三场景sample ID集合一致；checkpoint hash和manifest可读取。任一条件失败则不启动CVS正式矩阵。

### feature cache完成与CVS K5锚点计划

- 导出PID3796469正常完成；三份NPZ均5300行、160维，target-old/new无unknown/proxy行。正式验证`PASS`：三场景target行均只含对应星地场景；五receiver每个old/new TX均满足maxK20+query20覆盖；三场景共享完全相同的5300个sample ID。
- 本地回收：`local_artifacts/cvs_publication_adv3b02_feature_cache_20260713/`。SHA256：clear=`c5639f3c...a867`、low-elev=`912219b0...a872`、rain=`8f0b4e58...eb1a`。
- 下一步先跑receiver20-1、K5、seed713101的CVS-OPGAC与CVS-qKNNV42正式锚点，输出根`paper_reproduction/runs/cvs_publication_cvs_anchor_k5_seed713101_20260713/{cvs_opgac,cvs_qknnv42}`。两者使用同一三场景cache；OPGAC只登记target-old support，qKNNV42登记target-old+target-new support；query标签不进入适应或选模。
- 精确命令为`python -m paper_reproduction.cvs_aligned.cvs_method_runner --config paper_reproduction/configs/cvs_proposed_stage2_publication_features_n607.json --run-dir <method_dir> --method <cvs_opgac|cvs_qknnv42> --target-receiver 20-1 --seed 713101 --split-seed 713101 --k-shot 5 --device cpu`。成功条件为8个artifact、三场景、四层明细、finite trace、support/query无重叠及全测试星地增强。

### CVS K5锚点v1审计与修正

- v1两方法artifact链路PASS：OPGAC为360条score/57条明细/3条finite trace，qKNNV42为480条score/78条明细/3条finite trace；两者旧类support逐ID一致，均无重叠且全测试星地增强。
- v1数值：CVS-OPGAC适应前0.6722、适应后0.7361、delta+0.0639；CVS-qKNNV42 old0.6056、seen-new0.6333、H0.6170、forgetting0.0861。
- 反向审计发现v1 qKNNV42遗漏技术报告中固定的`diag_whiten_fisher` support-only变换（strength0.1），因此v1 qKNNV42降级为实现诊断，不能进入主表。已补入严格support-only的类间/类内Fisher对角缩放与对角whitening；三场景中每个已登记类都有同场景support，故`scenario_residual_weight=0.5`按公式为零并显式记录，而不是静默省略。
- 修正后py_compile及CVS runner+matrix测试仍为`4 passed`。需同步后写入新根`cvs_publication_cvs_anchor_k5_seed713101_v2_20260713`，不得覆盖v1。
