# D69冻结D62旧行并追加同族新行探针

## 1.执行前登记

- 实验ID：`d69_frozen_d62_old_append_d62_new_probe_20260719`；operator：Codex；状态：`PREREGISTERED_IMPLEMENTATION_PENDING`。
- 目标：保留当前联合最强D62的绝对跨类尺度，检验D65式Stage2-B旧行冻结能否减少注册遗忘，同时由D62同族final head提供新类行。
- 当前最强D62：B/A/N/H/F/J=92.78/82.22/84.67/82.62/10.56/26.67，min-B/A/N=80.00/53.33/73.33，混淆23/8/15。
- D68已完成105/105行并以B/A/N/H=58.89/51.67/14.00/18.66否决；其低F=7.22是注册前B先塌陷形成的伪改善。D68最终证据提交为`19c4603b`。
- 根目录`E:\type10-7`非Git；代码、追踪和本报告镜像进入`E:\type10-7\github_publish\CVS-RFFI-repo`。工作树中其他大量改动与D69无关，提交只暂存D69拥有路径。

## 2.唯一方法锁

Stage2-B执行完整D62并冻结6个旧类行`(W_B,b_B)`。Stage2-C在11类support上执行同一D62得到`(W_C,b_C)`，只追加其中5个新类行：

```text
W_final=concat(W_B[old],W_C[new])
b_final=concat(b_B[old],b_C[new])
```

不做逐行标准化、符号翻转、alpha融合、温度、offset、角色门、class名单、scene/receiver分支或超参数扫描。K1沿用D62自身精确D46 fallback。最终仍为一个全注册类affine head。

## 3.假设、可观察结果与停止条件

- 假设：D62的绝对行尺度已经包含有效joint竞争信息；只冻结旧行而让新行来自同族D62 final，可能比D65的异族block-LDA追加更兼容。
- before state、预测和全部指标必须与D62匹配；final旧FP32行与before逐bit相同，final新FP32行与D62 final逐bit相同。
- 相对D62必须无A/N/H/J/min-A/min-N交换，并至少严格改善A、F、J或floor之一；否则首seed即停止。
- INT8相对matched FP32的before/final argmax变化及margin sign flip必须为0；资源须保持正式上限。
- 真实105行完成后详细报告全部候选、场景、类、fold、混淆、训练、量化、资源、artifact和同排历史对照。失败不做第二seed或125。

## 4.数据与协议

- 固定development cell：receiver`20-1`、seed`713101`、K10/new5、三场景×五outer fold、实际K8。
- 复用D18`VALIDATED_ONCE/p2_min_v1`enrollment-only capsule；方法变化不触发数据重验。
- query只评分一次且不参与拟合；每query独立面对全部已注册类。clean/source、role Oracle、quota、batch assignment和dense query graph均禁止。
- ground实际输入锁为0。D22尚未达到正式Phase2资格；D66读取84个int8 cell仍为负交换，D69不以协议无效依赖换取旧类指标。

## 5.实施计划

新增独立D69 lifecycle wrapper、probe和专项测试，不修改D62历史实现或artifact。先验证：对称support、before精确D62、旧行bitwise冻结、新行精确D62 final、类置换等变、K1 fallback、调用配对、量化state旧行不变、禁止分支和资源闭包；随后运行D42–D69完整链。代码验证、提交和干净worktree复跑后，才登记并执行真实105行命令。

## 6.实现与本地验证

- `code/cvsrffi/stage2_d69_frozen_d62_append.py`：纯生命周期core，Stage2-B缓存D62旧行，Stage2-C只追加joint D62新行，记录旧/新行hash、支持准确率和禁止访问审计。
- `code/scripts/probe_d69_frozen_d62_old_append_d62_new.py`：复用锁定D62数学实现与D42 runner，增加编译后INT8/FP32旧state逐bit检查、资源闭包、source closure和D69 metadata。
- `tests/test_stage2_d69_frozen_d62_append.py`、`tests/test_probe_d69_frozen_d62_old_append_d62_new.py`：10项专项，覆盖before精确同D62、K1、append identity、新support不能改旧行、类置换、非法support、生命周期、state全部字段及禁止分支。
- 首次测试命令因本机Conda实际位于`F:\App\miniconda3`而不是旧路径失败，未进入pytest；改用正确hook并显式`conda activate ssr-gpu`。随后类置换测试的验证索引把“原类→新类”误当成逆置换，修正测试索引后专项10/10通过；算法实现未为此改动或放宽。
- D42–D69完整链335/335通过，用时81.1s；包含D42集成测试20项和D43–D69全部相关专项。测试运行目录为`local_artifacts/d69_full_chain_335`。
- source SHA256：core`bb59c3828ce63cdd168c00fe26a2ca82a2d7a37fade8105dc57f1dc9e6ec3bbf`；probe`f999f02523d150eb11e3cb872acd3df35271b44a4e9df8fc21cbce325c2f37a1`；未修改的D62 helper`c685e60402b5b172a0b2ed77e647e3aa506048b759f637fd88dcfb31ca114bcd`。

当前只有代码和合成验证，尚无真实outer性能。下一步提交精确文件，建立干净worktree并复跑335项；真实运行前继续保持状态`PREREGISTERED_IMPLEMENTATION_VALIDATED_PERFORMANCE_PENDING`。
