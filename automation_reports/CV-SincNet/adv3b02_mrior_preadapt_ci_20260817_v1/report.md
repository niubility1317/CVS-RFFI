# ADV3B02 MRIOR预适应CI对比预注册

- Run ID：`adv3b02_mrior_preadapt_ci_20260817_v1`
- 状态：`N607_FIRST_BUILD_STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。首次真实远端builder因`target-old support identity drift across new-count packages`停止；当时external PLAN与run root均未创建，未产生预测、评分或性能结果。
- 冻结代码提交：本报告与本次1200-job热修一起提交；提交SHA以最终Git交接为准。
- 操作角色：Task4运行闭环实现者；N607唯一runner待主任务交接。

## 目标与假设

在不改变CSIL或MoPC-HR原有类增量机制、矩阵、LEO输入及support/query划分的前提下，先对同一`receiver/seed/new_class_count/K/scene`package的target-old support完成冻结MRIOR-SDA预适应，再执行既有Task3 truth-free注册预测。比较对象是不可变v7无预适应reference。每个可报告比较必须保持`receiver`、`seed`、`K-shot`、`new_class_count`、方法和scene相同。

这是一项论文方法对比闭环，使用外部比较方法允许的source访问；不作为`p2_min_v1`主方法晋级或真实在轨验证声明。新类support与query仍须保持LEO输入。

## 冻结运行面

- Plan schema：`cvs.phase2.adv3b02_mrior_preadapt_ci_plan.v1`。
- 固定矩阵：1200个MRIOR预适应job、800个CI cell、2400个唯一scene row。
- `preadapt_scope=receiver_seed_newcount_k_scene`。每个预适应artifact identity为`receiver/seed/new_class_count/K/scene`，job ID为既有`preadapt_key`后追加`__new_<n>`；两个downstream method只在同一new-count package内共享该artifact。
- 分片：严格8个确定性shard。跨new-count的old-support token或package seal不得比较、拒绝或复用；每个cell必须映射本身`new_class_count`的三scene job。
- 预适应锁：200steps、Adam lr=0.0006、estimate=7、target CE=1.0、DV-KL=0.005、mu=0.5。
- 预适应输入：已封存source cache与同一target-old K-shot support；不打开query。
- Task3 predictor：`mrior_sda_then_csil_paper_full`或`mrior_sda_then_mopc_hr_paper_full`，必须提供三scene MRIOR bindings。

## 本地文件与核验

| 文件 | 目的 | 状态 |
|---|---|---|
| `paper_reproduction/scripts/build_adv3b02_mrior_preadapt_ci_plan.py` | 1200个new-count-specific MRIOR job、各自package seal与old-support token绑定、6-job smoke集合 | 已核验 |
| `paper_reproduction/scripts/run_adv3b02_mrior_preadapt_ci_plan.py` | 不覆盖run root、`preadapt_smoke`、smoke授权、8分片、MRIOR artifact与Task3 predictor/scorer命令收据 | 已核验 |
| `tests/test_build_adv3b02_mrior_preadapt_ci_plan.py` | new-count-specific绑定、合法old-support差异、1200-job formal合同 | 已通过 |
| `tests/test_run_adv3b02_mrior_preadapt_ci_plan.py` | 6-job smoke授权、1200/800/2400、8分片与两行技术停机 | 已通过 |

同步目标为N607项目根`/home/szu2070436088/2510044040/CV-SincNet`中的同名相对路径；同步文件包括Task1至Task4的4个生产脚本/模块，以及本报告。source cache固定为`/home/szu2070436088/2510044040/CV-SincNet/runs/adv3b02_three_da_leoweakonly_20260715_v1/phase1_caches/source/cache_set.json`，SHA256=`dea3bdb01d4f5791d7e92a01dbdcdb7f3d66b26bf134a375264b88eff8c6e4c4`。

本地命令：

```text
conda run -n ssr-gpu python -m pytest -q tests/test_build_adv3b02_mrior_preadapt_ci_plan.py tests/test_run_adv3b02_mrior_preadapt_ci_plan.py
conda run -n ssr-gpu python -m py_compile paper_reproduction/scripts/build_adv3b02_mrior_preadapt_ci_plan.py paper_reproduction/scripts/run_adv3b02_mrior_preadapt_ci_plan.py tests/test_build_adv3b02_mrior_preadapt_ci_plan.py tests/test_run_adv3b02_mrior_preadapt_ci_plan.py
```

结果：fresh focused builder+runner测试17passed；四个目标文件`py_compile`通过；`git diff --check`无空白错误。

## N607交接命令

前置条件：先在external PLAN路径生成并核验新的MRIOR plan；v7 source plan固定为`/home/szu2070436088/2510044040/CV-SincNet/runs/adv3b02_unfrozen_paperfull_ci_20260723_v7/protocol_plan/paper_full_plan_authorized.json`，SHA256为`1c5fb08231bc9d150d625e5360162c3ee287bdf778ac1c16a069ac187b96d65b`。source cache路径由MRIOR plan封存，不在命令行替换。

```bash
ROOT=/home/szu2070436088/2510044040/CV-SincNet
PY=/home/szu2070436088/.conda/envs/CVS-RFFI/bin/python
RUN_ROOT=$ROOT/runs/adv3b02_mrior_preadapt_ci_20260817_v1
PLAN=$ROOT/protocol_plans/adv3b02_mrior_preadapt_ci_20260817_v1/mrior_preadapt_ci_plan.json
SOURCE_PLAN=$ROOT/runs/adv3b02_unfrozen_paperfull_ci_20260723_v7/protocol_plan/paper_full_plan_authorized.json
SOURCE_CACHE=$ROOT/runs/adv3b02_three_da_leoweakonly_20260715_v1/phase1_caches/source/cache_set.json
cd "$ROOT"
"$PY" paper_reproduction/scripts/build_adv3b02_mrior_preadapt_ci_plan.py --source-plan "$SOURCE_PLAN" --expected-source-plan-sha256 1c5fb08231bc9d150d625e5360162c3ee287bdf778ac1c16a069ac187b96d65b --source-cache-manifest "$SOURCE_CACHE" --expected-source-cache-manifest-sha256 dea3bdb01d4f5791d7e92a01dbdcdb7f3d66b26bf134a375264b88eff8c6e4c4 --run-root "$RUN_ROOT" --output "$PLAN"
"$PY" paper_reproduction/scripts/run_adv3b02_mrior_preadapt_ci_plan.py --plan "$PLAN" --project-root "$ROOT" --stage prepare --device cuda:0
mkdir -p "$RUN_ROOT/logs"
"$PY" paper_reproduction/scripts/run_adv3b02_mrior_preadapt_ci_plan.py --plan "$PLAN" --project-root "$ROOT" --stage preadapt_smoke --shard-index 0 --shard-count 8 --device cuda:0 >"$RUN_ROOT/logs/preadapt_smoke.out" 2>&1
"$PY" paper_reproduction/scripts/run_adv3b02_mrior_preadapt_ci_plan.py --plan "$PLAN" --project-root "$ROOT" --stage smoke --shard-index 0 --shard-count 8 --device cuda:0 >"$RUN_ROOT/logs/smoke.out" 2>&1
```

只有上述`preadapt_smoke(cuda:0)`完成6个唯一job且`smoke(cuda:0)`写出PASS receipt后，8个预适应分片才能由同一唯一runner以独立、短连接调度；`i`与GPU索引相同：

```bash
for i in 0 1 2 3 4 5 6 7; do
  "$PY" paper_reproduction/scripts/run_adv3b02_mrior_preadapt_ci_plan.py --plan "$PLAN" --project-root "$ROOT" --stage preadapt_shard --shard-index "$i" --shard-count 8 --device "cuda:$i" >"$ROOT/runs/adv3b02_mrior_preadapt_ci_20260817_v1/logs/preadapt_shard_${i}.out" 2>&1 &
done
wait
"$PY" paper_reproduction/scripts/run_adv3b02_mrior_preadapt_ci_plan.py --plan "$PLAN" --project-root "$ROOT" --stage smoke --shard-index 0 --shard-count 8 --device cuda:0 >"$ROOT/runs/adv3b02_mrior_preadapt_ci_20260817_v1/logs/smoke.out" 2>&1
```

`smoke`只要求`smoke_preadapt_job_ids`中的6个不可变artifact均有合法job receipt，并运行既有4个smoke cell。PASS后写入`smoke_receipt.json`；`preadapt_shard`先核对此receipt和plan contract，再运行全部1200job。6个已完成job由不可变receipt核验后跳过。输出包括每cell的predictor/scorer精确命令收据、prediction、score和query-boundary receipt。

## 停机、成功与风险

- 停机：P0协议/覆盖违例，或两个不同outer row出现相同normalized、预测产生前异常指纹；状态必须写作`NO_PERFORMANCE_RESULT`。绝不以accuracy、H、BA或任何性能指标停机。
- smoke成功：6个声明的预适应artifact可核验，4个smoke cell的prediction与scoring receipt存在，query在model lock后才打开。
- 风险：本提交只释放技术smoke与其后的1200-job预适应派发；800-cell全量CI预测/评分、最终2400行闭合和paired analyzer仍待现有后续lane完成，不能据此作性能结论。

## 热修追溯

| ID | 要求 | 落点 | 状态 | 验证 |
|---|---|---|---|---|
| H1 | 去除跨new-count anchor复用与identity漂移拒绝 | builder、builder tests | 已实现 | new-count-specific RED→GREEN |
| H2 | identity=`receiver,seed,new_count,K,scene`，formal=1200/800/2400 | builder、runner、两组tests | 已实现 | focused tests |
| H3 | `preadapt_smoke`先运行6个job，`smoke`再运行4个cell | runner、runner tests | 已实现 | smoke receipt RED→GREEN |
| H4 | `preadapt_shard`必须先验证smoke authority，已完成job可receipt跳过 | runner | 已实现 | focused runner tests |

## 四状态结果表模板

| candidate/run | method | receiver | seed | K-shot | new count | scene | DA0_REG0 old | DA1_REG0 old | DA0_REG1 old/new/H | DA1_REG1 old/new/H | CSIL/MoPC DA effect | verdict |
|---|---|---|---:|---:|---:|---|---:|---:|---|---|---|---|
| 待artifact | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 | `DA1_REG1-DA0_REG1` | 待分析 |

`REG0`的new accuracy与`H_old_new`必须写作`N/A`；不得跨方法拼接best值。
