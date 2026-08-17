# ADV3B02 MRIOR预适应CI对比预注册

- Run ID：`adv3b02_mrior_preadapt_ci_20260817_v1`
- 状态：`LOCAL_VERIFIED`；尚未同步或启动N607。
- 提交：待填（Task4提交后补入SHA）。
- 操作角色：Task4运行闭环实现者；N607唯一runner待主任务交接。

## 目标与假设

在不改变CSIL或MoPC-HR原有类增量机制、矩阵、LEO输入及support/query划分的前提下，先对相同`receiver/seed/K/scene`的target-old support完成冻结MRIOR-SDA预适应，再执行既有Task3 truth-free注册预测。比较对象是不可变v7无预适应reference。每个可报告比较必须保持`receiver`、`seed`、`K-shot`、`new_class_count`、方法和scene相同。

这是一项论文方法对比闭环，使用外部比较方法允许的source访问；不作为`p2_min_v1`主方法晋级或真实在轨验证声明。新类support与query仍须保持LEO输入。

## 冻结运行面

- Plan schema：`cvs.phase2.adv3b02_mrior_preadapt_ci_plan.v1`。
- 固定矩阵：300个MRIOR预适应job、800个CI cell、2400个唯一scene row。
- 分片：严格8个确定性shard；每个预适应artifact key为`receiver/seed/K/scene`，不得重复。
- 预适应锁：200steps、Adam lr=0.0006、estimate=7、target CE=1.0、DV-KL=0.005、mu=0.5。
- 预适应输入：已封存source cache与同一target-old K-shot support；不打开query。
- Task3 predictor：`mrior_sda_then_csil_paper_full`或`mrior_sda_then_mopc_hr_paper_full`，必须提供三scene MRIOR bindings。

## 本地文件与核验

| 文件 | 目的 | 状态 |
|---|---|---|
| `paper_reproduction/scripts/run_adv3b02_mrior_preadapt_ci_plan.py` | 不覆盖run root、8分片、MRIOR artifact、Task3 smoke predictor/scorer、smoke授权与命令收据 | 已核验 |
| `tests/test_run_adv3b02_mrior_preadapt_ci_plan.py` | smoke门、run-root边界、300/800/2400、8分片与两行技术停机 | 已通过 |

本地命令：

```text
C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m pytest -q tests/test_run_adv3b02_mrior_preadapt_ci_plan.py
C:\Users\lh594\.conda\envs\ssr-gpu\python.exe -m py_compile paper_reproduction/scripts/run_adv3b02_mrior_preadapt_ci_plan.py
```

结果：focused runner测试6passed；`py_compile`通过。

## N607交接命令

前置条件：Task2生成并同步新的MRIOR plan；v7 source plan固定为`/home/szu2070436088/2510044040/CV-SincNet/runs/adv3b02_unfrozen_paperfull_ci_20260723_v7/protocol_plan/paper_full_plan_authorized.json`，SHA256为`1c5fb08231bc9d150d625e5360162c3ee287bdf778ac1c16a069ac187b96d65b`。source cache路径由该MRIOR plan封存，不在命令行替换。

```bash
ROOT=/home/szu2070436088/2510044040/CV-SincNet
PLAN=/home/szu2070436088/2510044040/CV-SincNet/runs/adv3b02_mrior_preadapt_ci_20260817_v1/protocol_plan/mrior_preadapt_ci_plan.json
PY=$ROOT/.conda/envs/CVS-RFFI/bin/python
cd "$ROOT"
"$PY" paper_reproduction/scripts/run_adv3b02_mrior_preadapt_ci_plan.py --plan "$PLAN" --project-root "$ROOT" --stage prepare --device cuda:0
```

8个预适应分片必须由同一唯一runner以独立、短连接调度；`i`与GPU索引相同：

```bash
for i in 0 1 2 3 4 5 6 7; do
  "$PY" paper_reproduction/scripts/run_adv3b02_mrior_preadapt_ci_plan.py --plan "$PLAN" --project-root "$ROOT" --stage preadapt_shard --shard-index "$i" --shard-count 8 --device "cuda:$i" >"$ROOT/runs/adv3b02_mrior_preadapt_ci_20260817_v1/logs/preadapt_shard_${i}.out" 2>&1 &
done
wait
"$PY" paper_reproduction/scripts/run_adv3b02_mrior_preadapt_ci_plan.py --plan "$PLAN" --project-root "$ROOT" --stage smoke --shard-index 0 --shard-count 8 --device cuda:0 >"$ROOT/runs/adv3b02_mrior_preadapt_ci_20260817_v1/logs/smoke.out" 2>&1
```

`smoke`只有在300个不可变artifact均有合法job receipt后才打开Task3 predictor/scorer；输出包括`smoke_receipt.json`、每cell的predictor/scorer精确命令收据、prediction、score和query-boundary receipt。

## 停机、成功与风险

- 停机：P0协议/覆盖违例，或两个不同outer row出现相同normalized、预测产生前异常指纹；状态必须写作`NO_PERFORMANCE_RESULT`。绝不以accuracy、H、BA或任何性能指标停机。
- smoke成功：300个预适应artifact可核验，所有smoke cell的prediction与scoring receipt存在，query在model lock后才打开。
- 风险：本提交只释放到smoke闭环；800-cell全量CI派发、最终2400行闭合和paired analyzer仍待后续Task4补充，不能据此作性能结论。

## 四状态结果表模板

| candidate/run | method | receiver | seed | K-shot | new count | scene | DA0_REG0 old | DA1_REG0 old | DA0_REG1 old/new/H | DA1_REG1 old/new/H | CSIL/MoPC DA effect | verdict |
|---|---|---|---:|---:|---:|---|---:|---:|---|---|---|---|
| 待artifact | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | `DA1_REG1-DA0_REG1` | 待分析 |

`REG0`的new accuracy与`H_old_new`必须写作`N/A`；不得跨方法拼接best值。
