# ADV3B02-FCR Task10报告：四阶段训练、权限与总损失

## Status

`LOCAL_VERIFIED`。四阶段日程、显式候选入口、L_s/U_s/V权限、Task6/7/8总损失组合和日志已完成本地TDD与聚焦回归。此状态不表示真实checkpoint smoke、N607训练或clean/三种LEO_WEAK最终评测已完成；这些属于Task11及后续实验。

## 设计追溯

| ID | 要求 | 实现与证据 | 状态 |
|---|---|---|---|
| T10-01 | 4段精确边界 | 8个边界、E0/E201和非200epoch测试 | verified |
| T10-02 | ramp与step交替 | E41为0、E90到配置值；偶数step普通、奇数step necessity | verified |
| T10-03 | 普通ADV3B02关闭FCR | `use_fcr=False`且8个有效lambda全0 | verified |
| T10-04 | L_s显式标签权限 | label-dependent项只读`label_mask` | verified |
| T10-05 | U_s无标签边界 | `labels=-1`、无identity/factor/transplant且允许项可反传 | verified |
| T10-06 | V/query边界 | V只读且state不变；query角色拒绝 | verified |
| T10-07 | 复用Task6/7/8 | 严格L_s necessity和U_s完整objective测试 | verified |
| T10-08 | Task9 detach | 内容参数无`delta_f`梯度、operator有梯度 | verified |
| T10-09 | E80/LEO兼容 | 原stage和`lambda_sat_cls`逐值不变 | verified |
| T10-10 | 可观测性 | stage/scales、9个loss组、pair、reason和freeze日志 | implemented |
| T10-11 | 限定追踪范围 | 只修改FCR-14/16/20/21/24 | verified |

## TDD与回归

RED先于实现：首次运行三个Task10测试文件时，三个文件均因`cvsrffi.phase1_fcr_schedule`不存在而在收集阶段失败。随后补最小实现，11项通过。U_s因子项增加负测后先得到`factor=4.0`而失败，再收紧为精确0。最终三个Task10文件共13项全部通过，耗时4.55秒。

全部既有`test_phase1_fcr_*`回归共59项通过。普通ADV3B02、BaselineOriginSatView、concat-sat和Meta训练聚焦组18项通过。`py_compile`、CLI help和`git diff --check`通过。

另有一项非Task10旧断言未通过：`test_meta_ssl_cli_defaults.py`期待`masked_y_minus_1_true_tx_in_meta_only`，当前Task2和项目协议实际为`masked_y_minus_1_no_reversible_tx_metadata`。该旧断言要求在U_s元数据保留可逆真TX，与当前禁令冲突；本任务没有修改该测试，也没有恢复隐藏TX。

## 精确日程

| epoch | FCR活动组 | scale规则 | necessity冻结 |
|---|---|---|---|
| E1-40 | `id,self,eta` | `self/eta`为配置值 | 否 |
| E41-90 | 增加`swap,shared,latent_cycle` | 乘`(epoch-41)/49`；E41为0，E90完整 | 否 |
| E91-150偶数optimizer step | normal real-combination及全部因子组 | 完整配置值 | 否 |
| E91-150奇数optimizer step | `transplant,necessity` | 只用`need` | 是 |
| E151-200 | 保留全部组和identity DG | `self/swap`为E90的0.25，其余完整 | 否 |

optimizer step只在安全反传实际完成后递增；跳过batch不改变奇偶相位。原ADV3B02分类、DG、E80卫星辅助CE和LEO_WEAK日程在原控制器中独立执行。necessity step只切换FCR附加项，不删除原ADV3B02核心loss。

## 8个CLI lambda

| CLI | 有效键 | 既有实现组件 |
|---|---|---|
| `--lambda_fcr_self` | `self` | Task7 self NLL加Task6 MRSTFT/phase |
| `--lambda_fcr_swap` | `swap` | Task7双向swap |
| `--lambda_fcr_shared` | `shared` | Task7双向stop-gradient shared |
| `--lambda_fcr_cycle` | `latent_cycle` | Task7重新编码cycle |
| `--lambda_fcr_eta` | `eta` | Task7已知且shape匹配的eta |
| `--lambda_fcr_factor` | `factor` | Task7 factor加anti-collapse；U_s为0 |
| `--lambda_fcr_need` | `need` | Task8 transplant/necessity |
| `--lambda_fcr_phys` | `phys` | Task6能量、平滑、边界及Fisher门控固定物理特征 |

配置默认值为1.0，但只有同时显式设置`--phase1_method adv3b02_fcr --use_fcr`才形成非零有效值。普通`adv3b02`即使收到非零FCR flag，有效值仍全部为0。方法名/开关不一致、非200epoch、非centralized或与concat-sat混用均在训练前失败。模型只在显式候选路径接收`use_fcr=True`和`FCRConfig(input_len=...)`。

## 权限和总损失

- L_s：原ADV3B02 identity CE和严格有效Task8 transplant可运行。新增label-dependent聚合只认`label_mask=True`。实际训练不重复计算identity CE，`fcr.id`保持0，原`loss_cls`继续作为身份主监督。
- U_s：必须全为`labels=-1`且`label_mask=False`，fingerprint index/mask全失效。只允许self、有效swap/shared/cycle、eta和phys；identity、factor、transplant精确0。不读真TX、硬伪标签或query。
- V：只读helper使用`eval+no_grad`，逐项state_dict不变，结束后恢复进入前模式；无optimizer、BN、prototype或memory更新。
- query：权限解析器显式抛错，训练函数没有query参数、路径或打开动作。

Task2无效pair仍由Task7/8返回连接图的精确有限0；没有identity、shuffle、label-derived或transplant fallback。真实Fingerprint Pair能力仍为`blocked`，因此无严格pair时移植项必须为0，日志只记录失效原因，不写隐藏TX真值。

## Task9梯度路由

Task9为规避Task4原地autograd版本冲突采用`fingerprint_excitation=content.s_hat.detach()`。Task10的cross decoder、Task8 adapter和fingerprint residual callback保持相同边界。因此`delta_f`可更新FingerprintFactorEncoder/Operator，但不存在`G_f→E_s`梯度；内容编码器仍经Decoder重构路径获得梯度。

## 日志、自查与追踪

每个FCR epoch记录stage、scales、`id/self/swap/shared/latent_cycle/eta/factor/transplant/phys`、active pair、失效reason和freeze状态，不输出TX标签、隐藏metadata或query事实。

自查确认：普通路径不构造FCR loss；8个边界无off-by-one；E80和原`lambda_sat_cls`未改；U_s不用`labels>=0`推断权限；V无持久更新；无query；identity CE未双计；necessity临时冻结后恢复Decoder；冻结普通ADV3B02身份backbone不进入optimizer。

追踪表只更新FCR-14、FCR-16、FCR-20、FCR-21和FCR-24。FCR-20仅在本地U_s训练路由范围标为`verified`；FCR-24保留`pending`，因为Task11真实checkpoint无query smoke及clean/`leo_clear_weak`/`leo_low_elev_weak`/`leo_rain_weak`最终评测尚未执行。

## Git发布

本报告与Task10拥有文件以`feat:add-FCR-training-schedule`同一提交发布。提交后的本地HEAD、push结果和远端OID将在任务完成回执中独立给出；不把未来OID回写同一提交，避免改变提交对象。

## Task11接口与关注点

- Task11使用显式方法、开关和8个唯一lambda，不得从普通ADV3B02隐式启用。
- 首次真实checkpoint smoke应核对stage日志、FCR state、pair计数、reason、无query和prediction闭合。FCR-13仍blocked时active fingerprint pair为0是正确结果。
- 当前LEO view复合nuisance通常为9字段，而Task9`eta_pred`为3字段。Task10只在shape精确匹配时启用eta，否则mask无效并返回0；Task11应记录能力事实，不得错误切片或伪装已知eta。
- Task11仍负责checkpoint往返、正式launcher、诊断和clean/三种LEO_WEAK最终评测；不得提前把FCR-24写成实验完成。
