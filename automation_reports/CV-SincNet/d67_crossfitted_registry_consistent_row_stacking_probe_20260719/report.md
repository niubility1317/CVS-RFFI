# D67交叉拟合registry-consistent连续行堆叠探针

## 1.执行前登记

- 实验ID：`d67_crossfitted_registry_consistent_row_stacking_probe_20260719`；operator：Codex；状态：`PREREGISTERED_IMPLEMENTATION_PENDING`。
- 目标：综合D62的旧/新联合表现与D65的旧类保持信号，用support-only、类别身份无关的连续闭式行融合提高after-old、遗忘和旧类floor，同时不牺牲seen-new、H、joint和新类floor。
- 当前联合最强D62：before92.78%、after82.22%、new84.67%、H82.62%、forget10.56pp、joint26.67%、min-before/min-after/min-new=80.00%/53.33%/73.33%，混淆23/8/15。
- D65正/负信号：after86.11%、forget6.11pp、min-after70%，但new59.33%、min-new46.67%；说明冻结旧决策有价值，而其新类标尺不可直接采用。
- cell：receiver`20-1`、seed`713101`、K10/new5、三场景×五outer fold、实际K8；直接复用同一`VALIDATED_ONCE/p2_min_v1`D18 enrollment-only support，不重验数据。
- 根目录`E:\type10-7`非Git；代码、追踪和本报告镜像进入`E:\type10-7\github_publish\CVS-RFFI-repo`。执行前主分支含D66最终报告提交`4b9819fa`和三轮回顾提交`eb8e8661`，工作树的其余大量修改均不属于本轮。

## 2.唯一机制与公式

对每个stage和每个匿名注册类`c`，分别生成D62与D65仿射专家`g_e,c(x)`。在四个预锁定physical-rank cross-fit折中，每折held两个rank、train六个rank；所有专家只在train support拟合。

对每个专家行，以train support的一对多统计计算：

```text
center_e,c = (mean_positive + mean_negative) / 2
within_e,c = sqrt((var_positive + var_negative) / 2)
gap_e,c = abs(mean_positive - mean_negative) / 2
scale_e,c = max(within_e,c, gap_e,c, float32_eps)
z_e,c(x) = (g_e,c(x) - center_e,c) / scale_e,c
```

inner-held目标为正类`+1`、其他类`-1`，每类正/负总权重各0.5。令`d=z_65-z_62`，闭式权重为：

```text
alpha_c = clip(sum_i w_i d_i (target_i-z_62,i) / sum_i w_i d_i^2, 0, 1)
h_c(x) = (1-alpha_c) z_62,c(x) + alpha_c z_65,c(x)
```

若分母不大于机器精度，`alpha_c=0`，即D62。full support只重算两个专家的center/scale并使用已锁定`alpha_c`；所有`h_c`再删除类公共仿射项并编译为一个全注册类affine state。没有role/class ID/scene/receiver分支，没有阈值、温度、alpha扫描、难类名单、outer-held/query拟合或跨query操作。K≤4精确回退D62，避免不合法的小K四折估计。

## 3.假设、判门与停止条件

- 假设：D65旧类保持信号可通过匿名support-held残差获得非零连续权重；其新类失配行会自动得到接近0的权重并回到D62。该判断必须由同一公式产生，不能读取old/new角色。
- 主门：相对D62总体before/after/new/H/joint、三项全局class floor、三场景同类指标、遗忘和三类混淆不得交换伤害，并至少严格改善after、forgetting、joint或任一floor。
- 量化：INT8相对matched FP32的before/final support与outer argmax变化、margin sign flip都必须为0；全部分数有限。
- 结构：四折partition exact-once，所有held rank不得参与对应专家拟合/归一化；`alpha∈[0,1]`，类置换等变；最终只保留单一affine state，query额外MAC/state为0。
- 失败即`COMPLETED_DIAGNOSTIC_NEGATIVE_NOT_PROMOTABLE`并停止D62/D65连续行融合；不扫描fold数、alpha温度、ridge、阈值或按角色设权。即使通过也先运行第二development seed，不直接启动125。

## 4.实施与预期证据

待新增独立D67 probe、专项测试和专属摘要；不修改D62/D65历史实现或artifact。测试至少覆盖：四折physical-rank exact-once、闭式权重解析解、类置换等变、K≤4精确D62回退、D65 final专家由旧support冻结协方差后追加、full affine编译等价、无角色/场景/outer-held/query分支及资源闭包。

真实运行前补齐Git提交、干净worktree、完整回归、精确命令、输出路径和资源估计。运行完成后必须在本报告补齐七候选、三场景、11类、15fold、alpha分布/专家贡献、量化、训练、资源、artifact、D62/D64/D65/D66同排对照和目标缺口；不得只报告缺陷。

本轮先本地开发和验证，不访问N607。只有本地锁定候选需要大规模独立seed/matrix时，才按`AGENTS.md`执行N607 preflight、报告、Git、SCP及短连接闭环。
