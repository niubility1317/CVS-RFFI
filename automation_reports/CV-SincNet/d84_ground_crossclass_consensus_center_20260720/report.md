# D84地面跨类一致域模板稳健中心

状态：`PREREGISTERED_LOCAL_VERIFIED_QUERY_NOT_OPENED`。实验ID：`d84_ground_crossclass_consensus_center_20260720`；时间：2026-07-20 06:35 HKT；操作者：Codex。

## 目标与历史边界

D83的统一协方差loading相对D81在15/15 outer rows上零预测变化，却增加114.26M adaptation MAC；D78/D79直接地面切向logit residual曾用旧类收益交换新类损失；D80/D82直接修改协方差或残差均已判负。D84不重启这些路线，只沿D81已在两个seed复现的center-only正收益机制继续研发。

## 单一机制差异

地面bundle包含14 domains×6 old classes的84个int8中心。D84先对每个ground class跨domain去中心，随后在每个domain内对6类残差求共同漂移`g_d`；以

```text
rho_d = ||g_d||^2 / (||g_d||^2 + mean_c ||r_dc - g_d||^2 + eps)
pi_d = rho_d / sum_d rho_d
energy_i = sum_d pi_d * <x_i - mean_y, normalize(g_d)>^2
```

替换D81的全84-cell协方差特征谱。ground类中心和类特有漂移在生成模板后丢弃，不做target identity映射、不产生旧类分数。target每类仍使用一步Cauchy权重，只平移z160类共同中心，类内残差和FFT96/RF32逐位保持；最终仍由target support D62产生单个INT8 affine head。

## 创新性与效率假设

- 跨地面类一致性只保留能够跨6个旧类复现的domain drift，过滤Phase1旧类身份交互，预期比D81的混合协方差对新类更公平。
- 14个domain模板直接闭式构造，无160×160协方差和特征分解、无rank/weight/强度扫描；target translation维度不超过14。预期地面统计从D83的90.52M MAC降到低于0.2M，且不再有1.84M covariance loading。
- K≤2精确identity；新旧类同式；query/clean/source/role/quota/global assignment访问0；ground只读且不写回。

## 预注册开发门

固定复用D18 `VALIDATED_ONCE` cell：receiver=`20-1`、seed=`713101`、K10（actual K8）、new5、3场景×5fold。相对D81要求总体及每场景B/A/N/H/J、全部class floors和mean-row floors不退化、F不升、三类混淆不增加，且A/H/F、rain、最差旧类或新类中至少一项严格改善。未通过即停止第二seed和125；ground组件仍为`UNVERIFIED_UNDER_CURRENT_PROTOCOL`，任何结果仅为development diagnostic。

## 版本、验证与待办

`E:\type10-7`根不是Git仓库；实现位于隔离Git worktree`E:\type10-7\code\snapshots\d81wt`，将精确提交并cherry-pick到`E:\type10-7\github_publish\CVS-RFFI-repo`。核心SHA256=`5bebd29643767ee349059dbf80e0208a349a5e8870ea599c3d5e96cd63e17dff`；probe SHA256=`0642252d34d9c50c90aecb1478163f66304f355415ed3a4c8e86cedb4305db30`。D84专项12/12、D42/D62/D81-D84相邻链87/87 PASS，`py_compile`与`git diff --check`PASS。

真实ground只读加载确认：26个registry domain槽位中14个有效domain、84个有效cell，保留14个一致域模板；跨类一致性`rho` min/mean/max=`0.14591/0.17696/0.21422`，模板权重`0.05889..0.08647`；template SHA256=`9a36dd9b85841282987fb8093dd3fe5daf8003c945995dfd5b0f651f8caa6eb4`，weight SHA256=`adcbb4724213c5d80d825756b4b17c3605c18ea8c35d0ccf4a38f9be264aafea`。地面统计上界179,200 MAC，较D83的90.52M减少99.80%；加上预计21.89M center translation，D84新增适配约22.07M MAC，较D83的114.26M减少约80.7%。

下一步在独立输出`ground_crossclass_consensus_center/`执行真实105-row实验；未完成前无性能结果。
