# D50全局锚定类级中位数证据融合探针

## 1.状态与目标

- 实验ID：`d50_centered_median_evidence_fusion_probe_20260719`。
- 当前状态：`IMPLEMENTED_VERIFIED_NOT_RUN`。
- 目标：综合D45全局LOO融合的稳定性与D46类级LOO对new floor的真实改善，修复D46的类级均值易受少量support held rank拉偏、D47收缩完全退回D45、D48截距残差过强、D49全局cosine nested CE失配。
- development cell固定为receiver`20-1`、seed`713101`、K10/new5、`leo_clear_weak/leo_low_elev_weak/leo_rain_weak`×5 outer折；实际outer fit K8。只复用`VALIDATED_ONCE`的`p2_min_v1`固定received-IQ capsule/split，方法变化不触发数据重验。
- 本轮仅本地开发探针；不访问N607，不运行第二development seed或125。

## 2.唯一方法

继承D45/D46相同的B20、全288d shrinkage-LDA head、3-block shrinkage-LDA head、support RMS、canonical gauge、一次FP32融合和int8/FP16生命周期。对每个匿名注册类`c`及inner held rank`r`定义：

`d_{r,c}=CE_block,r,c-CE_full,r,c`。

D45全局log-odds锚点保持为：

`z0=C×mean_{r,c}(d_{r,c})`。

类级稳健位置与中心偏差为：

`m_c=median_r(d_{r,c})`，`delta_c=K×(m_c-mean_j(m_j))`。

最终：

`z_c=z0+delta_c`，`w_full,c=sigmoid(z_c)`，`w_block,c=1-w_full,c`。

由构造可得`mean_c(z_c)=z0`，因此类级差异不会移动D45的全局log-odds中心。median对偶数K使用两个中间值的算术平均。K1逐位回退D45；K2时full/block均为单位协方差等价head，必须得到精确0.5/0.5，否则fail-close。

该方法只称`D45-anchored centered median rank evidence fusion`，不宣称median是校准posterior、泛化误差估计或场景不变量。

## 3.协议、对称性与禁止项

- `d_{r,c}`只来自合法support标签和严格inner train/held分区；不读取outer-held、query、clean/source、receiver、scene、handle、class ID、old/new角色或任何class quota。
- 每类使用完全相同公式，class-label置换必须等变，rank顺序置换必须不变。
- query继续对全部注册类逐样本独立一次性argmax；truth、role Oracle、true batch class count、quota、global reassignment、query-dependent batch optimization和dense query graph均禁止。
- 不增加temperature、clip、阈值、sign gate、trim比例、权重扫描、第二arm或post-hoc选择；不得根据本轮outer结果切换mean/median。
- sigmoid权重必须有限、严格位于(0,1)且逐类和为1；median、中心、锚点或canonical state任一闭合失败即停止。

## 4.资源预注册

D50不新增B20、LDA fit、optimizer step、query state或sidecar，继承D46的K8 before/final共36次LDA和一个`C×288+C`state。median排序比较单列审计；数值运算保守沿用D47的`O(CK)`、K8两state`1,256`scalar MAC-equivalent上界，因此总适配开销预计不超过D47的`1,077,329,226`MAC-equivalent，query仍为6,624 MAC，参数2,016，state 8,583B，epoch/step为20/20。真实artifact必须报告实测CUDA peak；host FP64 peak未测时继续标未测。

## 5.开发晋级门

D50必须同时满足：

1. 相对D45至少改变1条final outer prediction；否则机制无决策价值。
2. 相对D45的总体和clear/low-elev/rain各自after-old、seen-new、同row H、joint、min-old、min-new均不退化，forgetting不增加。
3. seen-new和min-new至少达到D46的84.67%/73.33%，同时rain after-old/forgetting至少达到D42的78.33%/≤10.00pp；总体forgetting不得高于D42的8.89pp。
4. old→new/new→old/new→new混淆不超过D42的26/10/18。
5. before/final/margin的int8相对FP32翻转为0/0/0；协议、source、ground、state、资源、artifact全部闭包。

任一失败即记`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`，不追加公式变体、第二seed或125。全部通过也仅进入另行formalize，不能直接宣称正式性能。

## 6.完成后详细性能账

实验完成后必须在本报告及根目录镜像写入：7候选总体表、3场景表、11类表、15个outer行、相对D42/D45/D46同条件差值、D50权重/median/均值偏差/锚点分布、20步训练轨迹、逐向混淆、FP32/int8误差与top-tie、资源闭包、全部artifact大小/SHA、缺陷机理和下一轮决策。不得只写失败原因或边际最大值。

## 7.版本与执行占位

- Git承载面：`E:\type10-7\github_publish\CVS-RFFI-repo`，分支`codex/cvs-rffi-release-20260626`；工作树有用户无关改动，本轮只stage D50自有文件。
- 根目录`E:\type10-7`的`.git`不可用，根报告仅为运行镜像，不宣称根目录版本化。
- 计划实现：`code/scripts/probe_d50_centered_median_evidence_fusion.py`；测试：`tests/test_probe_d50_centered_median_evidence_fusion.py`；追踪：`analysis/d50_centered_median_evidence_fusion_traceability_20260719.md`。
- 输出、代码提交、clean worktree、输入SHA、exact command、PID/GPU、日志和最终artifact在首次运行前补锁。

## 8.实现与运行前验证

D50通过D46的`reliability_strategy`入口实现，只替换support证据聚合；B20、两个LDA head、RMS、canonical affine融合、int8/FP16编译和runner均未改动。运行后verifier将从每个artifact的逐fold CE重算mean、median、D45锚点、中心偏差、post-log-odds和权重，再把audit临时还原为D46标准权重调用既有D46完整verifier，从而同时验证D50新增公式与D46分区/融合/资源闭包。D50 source closure绑定D47/D46/D45/D44/D43全部helper SHA及D50探针本身。

本地验证：D50定向`8 passed`；D46＋D47＋D50联合`45 passed`；D42–D50全链`152 passed`；`py_compile`通过，所有pytest退出码为0。代码复核未发现P0/P1：K1返回D45等权fallback；K2只允许等价head；rank置换不变、class置换等变；`mean(post_log_odds)=z0`在`1e-12`内fail-close；非有限/分区/CE闭合均拒绝。当前仍未读取outer结果。

## 9.执行锁与exact command

- 实现提交：`003f0babd6791302bbcdcaf03a15e8cedc439c35`；clean detached worktree为`E:\type10-7\code\snapshots\d50wt`，`git status -sb`仅`## HEAD (no branch)`。
- clean worktree探针SHA256：`65e4b31b3add10463744faf6dab9b2e74ef9a2183aba19988cba021dbd5acf53`。主工作树文本因Windows行尾转换具有不同工作树字节SHA，运行只使用上述clean artifact。
- runtime继续只读使用历史锁定`E:\type10-7\code\snapshots\d41wt`；D50 bootstrap对12个runtime模块的内置source closure通过。首次预检在closure通过后因打印了错误诊断属性名退出，未创建输出、未执行fold；改用正确只读字段重跑通过。
- before/after seal、before/after envelope、component manifest、class binding实际SHA依次匹配`53ace286…d9f75`、`c70aedf3…b50ff`、`31a2ad99…ceb0e`、`a2483d6e…be76`、`15b5e144…629c`、`bb89a1db…c901f`。
- 输出`E:\type10-7\automation_reports\CV-SincNet\d50_centered_median_evidence_fusion_probe_20260719\centered_median_evidence_fusion`启动前不存在。本地串行`device=auto`；不访问N607、不生成125。

```powershell
& 'C:\Users\lh594\.conda\envs\ssr-gpu\python.exe' `
  'E:\type10-7\code\snapshots\d50wt\code\scripts\probe_d50_centered_median_evidence_fusion.py' `
  --d50-arm centered_median_evidence_fusion `
  --runtime-root 'E:\type10-7\code\snapshots\d41wt' `
  --probe-root 'E:\type10-7\code\snapshots\d50wt' `
  --before-root 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\phase2_capsule_k10_new5\predictor\before\enrollment_only' `
  --before-seal 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\phase2_capsule_k10_new5\seals\before_enrollment.seal.json' `
  --before-seal-sha256 53ace2863c9da6c2f6cc855d602c99f581df6de3d30a9a3ecb89eb6b6f0d9f75 `
  --before-formal-policy 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\formal_execution_policy.json' `
  --before-formal-policy-authorization 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\runtime_authorization_k10_new5\before_formal_policy_authorization.v2.json' `
  --before-signed-policy-authorization-envelope 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\runtime_authorization_k10_new5\before_signed_policy_authorization_envelope.v2.json' `
  --before-signed-policy-authorization-envelope-sha256 31a2ad9918f061b25d5a7ed0cc135df70ae02460c094b2f396bf314817bceb0e `
  --after-root 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\phase2_capsule_k10_new5\predictor\after\enrollment_only' `
  --after-seal 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\phase2_capsule_k10_new5\seals\after_enrollment.seal.json' `
  --after-seal-sha256 c70aedf3a8f059e756806201758c1933a2f3e1ba4df415e69a1c776b1a2b50ff `
  --after-formal-policy 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\formal_execution_policy.json' `
  --after-formal-policy-authorization 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\runtime_authorization_k10_new5\after_formal_policy_authorization.v2.json' `
  --after-signed-policy-authorization-envelope 'E:\type10-7\automation_reports\CV-SincNet\d18_formal_k10_new5_rx20_1_seed713101_20260717_085303\runtime_authorization_k10_new5\after_signed_policy_authorization_envelope.v2.json' `
  --after-signed-policy-authorization-envelope-sha256 a2483d6e9c9c362d89397029ff1e43f48358be3bdb3a05d717ee112b70a0be76 `
  --component-dir 'E:\type10-7\automation_reports\CV-SincNet\d22_int8_anchor_lifecycle_20260717\input\int8_component' `
  --component-manifest-sha256 15b5e144f9af3989421d8e925c17758479c327be47e79222f6363dc63994629c `
  --class-binding 'E:\type10-7\github_publish\CVS-RFFI-repo\analysis\d19_adv3b02_class_binding_20260717.json' `
  --class-binding-sha256 bb89a1dbb831acb374fccfc596ae98b660b496b449bdca577dabb962121c901f `
  --output 'E:\type10-7\automation_reports\CV-SincNet\d50_centered_median_evidence_fusion_probe_20260719\centered_median_evidence_fusion' `
  --device auto --mode development_select_unverified_component --candidate-set d42_v1
```
