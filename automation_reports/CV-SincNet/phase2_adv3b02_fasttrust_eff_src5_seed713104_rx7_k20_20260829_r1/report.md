# Phase2七接收机域目标确认报告

## 预登记

- run_id：`phase2_adv3b02_fasttrust_eff_src5_seed713104_rx7_k20_20260829_r1`
- 状态：`LOCAL_VERIFIED`
- 候选：冻结`S713104_ADV3B02_FASTTRUST_EFF`
- checkpoint：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase1_adv3b02_fasttrust_eff_src5_seed8_e200_20260828_r1/S713104_ADV3B02_FASTTRUST_EFF/final_ssdg.pth`
- 方法：`freq_f3_proj`，1步support-only late-block adaptation，`learning_rate=0.0005`，可训练比例范围`[0.03,0.15]`
- 数据：`protocol_schema=p2_min_v1`，`phase2_data_status=VALIDATED_ONCE`，`capsule_id=536fb610302e0298fe98b4708d2e6d51eb81aef676126c01d8de6ff1a67985f2`，`split_id=260f7bc291e8dbfe53e68f58997414a7d89c8f15b55d59793de506fb434fac25`
- K-shot：`K=20`
- 状态口径：本次仅确认冻结最终方法的`DA1_REG1`；`DA0_REG0/DA1_REG0/DA0_REG1`均为`NOT_RUN`，不得据此声称四状态因果效应。
- 接收机：`1-1,14-7,2-1,20-1,7-14,7-7,8-8`
- 场景：`leo_clear_weak,leo_low_elev_weak,leo_rain_weak`
- 矩阵：7个接收机×3个场景，共21行；每行1352个query，26类（6个旧类+20个注册新类）。
- prediction-first：21行prediction全部原子发布并完成schema、1352行、有限值、26类范围和唯一query ID验证后，独立scorer才可打开第一份truth。
- 选择规则：无目标域选种、调参、重训或选择性重跑；本次只报告冻结方法在所有接收机域的结果。

## 版本与本地验证

- Git提交：`9c8ade0fb6ad775a5847577f29efac71f3845523`
- 分支：`codex/phase1-fasttrust-eff-src5-20260828`
- 本地验证：46项聚焦测试通过；Python编译通过；CLI入口通过；`git diff --check`通过。
- 独立P0/P1审查：无P0；原2项P1已定点修复。冻结checkpoint与seed成对绑定；prediction临时写入后原子发布，全局评分门在truth前完整验证21行。
- release归档：`E:\type10-7\local_artifacts\releases\phase2_adv3b02_fasttrust_eff_src5_seed713104_rx7_k20_20260829_r1\release.zip`
- release SHA256：`1364fab5256756fbcddcdef628194329ccba61765421ab16b7ccd5b83debaa14`
- 远端release：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase2_adv3b02_fasttrust_eff_src5_seed713104_rx7_k20_20260829_r1/release.zip`
- 远端解压目录：`/home/szu2070436088/2510044040/CV-SincNet/releases/phase2_adv3b02_fasttrust_eff_src5_seed713104_rx7_k20_20260829_r1/unpacked`

## N607命令与路径

- 账户：普通`N607`账户`szu2070436088`，禁止管理员账户。
- 环境：`ssr-gpu`
- 代码CWD：远端release解压目录。
- canonical输入：`/home/szu2070436088/2510044040/CV-SincNet/runs/P2_CANONICAL_UNION_SMOKE_V1_20260828/cache/cache_set.json`
- 输出根：`/home/szu2070436088/2510044040/CV-SincNet/runs/phase2_adv3b02_fasttrust_eff_src5_seed713104_rx7_k20_20260829_r1`
- stage命令：`python code/scripts/run_phase2_fasttrust_receiver_confirmation.py stage --cache-set <cache_set.json> --run-root <output_root> --checkpoint <frozen_checkpoint>`
- prediction命令：每个接收机执行`python code/scripts/run_phase2_fasttrust_receiver_confirmation.py predict-receiver --matrix <output_root>/matrix.json --receiver <receiver> --device cuda:0`。
- GPU：`1-1→GPU0`，`14-7→GPU1`，`2-1→GPU2`，`20-1→GPU3`，`7-14→GPU4`，`7-7→GPU5`，`8-8→GPU6`；每个进程在同卡顺序执行3个场景，GPU7保留。
- score命令：`python code/scripts/run_phase2_fasttrust_receiver_confirmation.py score-all --matrix <output_root>/matrix.json`

## 预期artifact与停止规则

- 预期：`matrix.json`、`stage_receipt.json`、7份`prediction_receipt.json`、21份`predictions.npz`、21份`score.json`和`summary.json`。
- 允许停止：错误checkpoint/seed/K/receiver/scenario/split，query/truth边界违规，输出碰撞，错误checkout，不能启动，确定性重复异常，无合法prediction闭合或scorer连接错误。
- 禁止停止：中间或最终性能偏低。不得重启健康行、不得热补丁、不得影响无关进程。

## 正式结果

待21行prediction-first闭合并由独立scorer连接truth后填写。
