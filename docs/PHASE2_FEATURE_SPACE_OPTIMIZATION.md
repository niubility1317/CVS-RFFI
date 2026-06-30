# Phase2地面特征空间优化设计与实现

生成时间：2026-06-29
状态：已实现默认关闭的训练期特征空间优化；拒识评分层扩展仍按Phase2计划分阶段推进。
最高约束：本方案只作用于地面训练的`z_id`或`--generalization_feature`指定的identity特征，不改变`项目.md`中的Stage2-A/B/C协议，不让`z_dom`进入TX原型距离。

## 文献依据与筛选结论

本轮检索的有效特征空间方法可以分为四类：

| 方法族 | 代表工作 | 可借鉴点 | 本地落地判断 |
|---|---|---|---|
| 监督对比学习 | Khosla et al.,Supervised Contrastive Learning(NeurIPS 2020) | 拉近同类样本、推远异类样本，适合提升embedding可分性 | 本地已有`domain_aware_supcon_loss`，保留复用；不重复实现SupCon |
| 角度margin分类 | Deng et al.,ArcFace(CVPR 2019) | 在归一化球面上扩大类间角间隔，和cosine/angular原型头一致 | 借鉴角度margin思想，但不替换现有分类头 |
| 类内紧凑损失 | Wen et al.,Center Loss(ECCV 2016) | 显式约束类内半径，降低原型方差 | 本轮实现为batch级类内角半径约束 |
| 原型/距离式开集 | Snell et al.,Prototypical Networks(NeurIPS 2017)、Bendale and Boult,OpenMax(CVPR 2016)、CAC distance-based open-set loss | 原型距离、半径和开放集判定高度相关 | 本轮优化训练期原型几何；OpenMax/EVT、Mahalanobis、energy保留在拒识评分层 |

补充拒识相关方法包括Mahalanobis OOD和Energy-based OOD detection。它们对unknown rejection有价值，但主要是推理期评分、校准或阈值选择方法。CVS当前更紧迫的问题是地面训练得到的`z_id`空间是否具有足够的类内紧凑性和类间角间隔，否则后续Phase2原型导出、新类注册和unknown拒识都会被半径重叠拖垮。因此，本轮先落地训练期特征几何优化，而不是把OpenMax/Mahalanobis/energy强行写进训练主循环。

主要来源：

- Khosla et al.,Supervised Contrastive Learning：https://arxiv.org/abs/2004.11362
- Deng et al.,ArcFace: Additive Angular Margin Loss for Deep Face Recognition：https://arxiv.org/abs/1801.07698
- Wen et al.,A Discriminative Feature Learning Approach for Deep Face Recognition：https://link.springer.com/chapter/10.1007/978-3-319-46478-7_31
- Snell et al.,Prototypical Networks for Few-shot Learning：https://arxiv.org/abs/1703.05175
- Bendale and Boult,Towards Open Set Deep Networks/OpenMax：https://arxiv.org/abs/1511.06233
- Liu et al.,Energy-based Out-of-distribution Detection：https://arxiv.org/abs/2010.03759
- Lee et al.,A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks：https://arxiv.org/abs/1807.03888
- Class Anchor Clustering for distance-based open-set recognition：https://arxiv.org/abs/2004.02434

## 本地代码适配判断

本地已有三类近似能力：

1. `code/cvsrffi/losses.py::domain_aware_supcon_loss`：已做同TX跨domain正样本的SupCon，解决“同类跨接收机拉近”的一部分问题。
2. `code/cvsrffi/losses.py::PrototypeMemoryBank`：训练时EMA原型正则，已有class pull、domain align和prototype push，但它是有状态memory bank，不适合作为新的地面特征空间主约束。
3. `code/cvsrffi/open_world_head.py::OpenWorldMultiPrototypeHead`和`code/cvsrffi/phase2_prototypes.py`：Phase2离线原型、半径和open-world head已经使用cosine/angular距离。

因此，本轮不新增平行memory bank、不改模型forward、不改loader、不改checkpoint语义。新增方法落在`code/cvsrffi/losses.py`，由`code/train.py`在已有`dg_feat = select_generalization_feature(...)`之后默认关闭接入。

## 已实现方法：Open-World Angular Feature Space Loss

记归一化identity特征为`z_i`，类别中心为`c_y`，角距离为：

```text
theta(a,b)=acos(clamp(a^T b,-1,1))
```

总损失：

```text
L_ow = L_compact + L_inter + L_sample + alpha_dom * L_domain
```

其中：

```text
L_compact = mean_i [max(0, theta(z_i,c_yi)-r)]^2
L_inter = mean_{a<b} [max(0, m_inter-theta(c_a,c_b))]^2
L_sample = mean_i [max(0, theta(z_i,c_yi)+m_sample-min_{k!=yi}theta(z_i,c_k))]^2
L_domain = mean_{y,d} theta(c_{y,d}, stopgrad(c_y))
```

设计含义：

- `L_compact`降低旧类原型半径，直接服务Phase2 old-class calibration和unknown拒识。
- `L_inter`扩大类中心角间隔，降低old/new注册重叠。
- `L_sample`约束单样本离最近负类中心的安全间隔，避免只优化中心但留下边界样本。
- `L_domain`可选，只在显式设置`--ow_feat_domain_align_weight>0`且batch中有domain label时启用，用于同TX跨domain中心对齐；它不使用`z_dom`，也不把domain特征并入TX距离。

## 本地接入点

| 文件 | 新增/修改 | 与已有数据流关系 |
|---|---|---|
| `code/cvsrffi/losses.py` | 新增`open_world_feature_space_loss` | 复用`safe_l2_normalize`和现有loss工具，返回graph-safe zero，默认无状态 |
| `code/train.py` | 新增`--lambda_open_world_feat`及`--ow_feat_*`参数 | 在`select_generalization_feature`后接入；默认`0.0`不计算、不改变训练结果 |
| `code/cvsrffi/logging.py` | 新增`[LOSS-OW-FEAT]`日志行和weighted loss top项 | 训练时可看到compact/inter/sample/domain几何指标 |
| `code/tests/test_open_world_feature_space_loss.py` | 新增synthetic tensor测试 | 覆盖塌缩几何惩罚、domain中心错位、类别不足时graph-safe zero |
| `code/tests/test_phase2_train_cli.py` | 扩展CLI默认关闭测试 | 检查默认关闭参数和训练接入字符串 |

## 推荐实验入口

默认不启用。需要地面训练特征空间优化时，可先使用保守设置：

```powershell
python code\train.py `
  --lambda_open_world_feat 0.01 `
  --ow_feat_radius_deg 12 `
  --ow_feat_inter_margin_deg 55 `
  --ow_feat_sample_margin_deg 5 `
  --ow_feat_domain_align_weight 0.05
```

调参边界：

- 先在source validation、strict UDU、worst receiver和Phase2离线prototype半径上验证，不得用target query调参。
- 若`train_ow_feat_active_classes`长期低于2，说明batch采样不足以形成类间角约束，应优先检查sampler，而不是提高loss权重。
- 若`train_ow_feat_min_inter_deg`提升但`val_tx_acc`下降，说明margin过强，应先降低`--lambda_open_world_feat`或`--ow_feat_inter_margin_deg`。
- 若跨receiver半径仍大，可小幅提高`--ow_feat_domain_align_weight`，但它必须保持在identity feature内，不得改成使用`z_dom`。

## 延期项

| 项目 | 延期原因 | 后续落点 |
|---|---|---|
| Mahalanobis协方差门 | 需要稳定离线原型导出和足够类内样本估计协方差 | `code/cvsrffi/open_world_head.py`或独立评估adapter |
| OpenMax/EVT tail fitting | 属于拒识阈值层，不能用unknown query拟合 | Phase2离线评估CLI |
| Energy OOD | 更适合作为logit/score层baseline，不直接优化`z_id`几何 | Phase2评估表 |
| full episodic meta-refiner | 会改变训练范式和方法声明，风险高 | 完成P1-P3闭环后再评估 |

## 验证状态

已执行：

```powershell
conda activate ssr-gpu
python -m pytest code\tests\test_open_world_feature_space_loss.py code\tests\test_phase2_train_cli.py -q
python -m pytest code\tests\test_open_world_feature_space_loss.py code\tests\test_phase2_train_cli.py code\tests\test_epoch_timing_logging.py code\tests\test_open_world_head.py code\tests\test_phase2_prototypes.py -q
python -m py_compile code\cvsrffi\losses.py code\cvsrffi\logging.py code\train.py
python code\train.py --help | Select-String -Pattern "open_world_feat|ow_feat"
```

结果：Git镜像19 passed；根目录关键验证5 passed，另有`.pytest_cache`权限警告，不影响测试结论。
