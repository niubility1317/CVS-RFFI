# phase1_adv3b02_open_boundary_distill_20260706

## Objective

R8/R9/R10的Stage2-C qknn8协同评估已经形成负证据：最佳同row仅为`R8_SHELL, M_budget=1`，old_acc35.41%、min_old5.00%、seen_new2.37%、unknown_reject40.94%。严格`exact_k`同事件5接收机也不可用，因为当前特征证据中不存在5接收机共同观测组。

下一步不再把主要增益押在协同融合。我们需要重塑地面模型的`z_id`空间，使叠加LEO星地信道后的未知类不再落在已知类核心附近，同时保持旧类source/test TX准确率。该路线命名为`ADV3B02_OPEN_BOUNDARY_DISTILL`。

## Protocol Boundary

|item|rule|
|---|---|
|teacher|`ADV3B02_CORE90_SOFT_E200`|
|student|CV-SincNet/CVS source-only backbone|
|training data|ManySig/source receivers only|
|forbidden data|真实`Y_unknown`、ManyTx unknown query、target receiver样本、target receiver统计、target阈值、target early stopping|
|allowed proxy|source-only leave-one-class-out proxy、source prototype interpolation、source low-density synthetic boundary、teacher-uncertain source augmentations|
|deployment eval|仍用Stage2-C qknn8 old+seen-new+unknown query；unknown query eval-only|

This is not open-set training on the true unknown classes. It is source-only boundary shaping.

## Algorithm

Let the teacher produce frozen embeddings and logits:

```text
z_T = f_T(x), p_T = softmax(g_T(z_T) / T)
z_S = f_S(x), p_S = softmax(g_S(z_S) / T)
```

For each source class`c`, build a teacher core prototype`mu_c^T` and a robust class radius`r_c^T` from high-confidence source samples:

```text
C_c = {x: y=c, max(p_T) >= q_core, teacher_correct(x)}
mu_c^T = mean(normalize(z_T(x)) for x in C_c)
r_c^T = Quantile(||normalize(z_T(x))-mu_c^T||_2, q_radius)
```

The student uses four loss groups:

```text
L = L_ce + lambda_kd L_kd + lambda_core L_core
    + lambda_boundary L_boundary + lambda_tail L_tail
    + lambda_overflow L_overflow
```

Identity retention:

```text
L_ce = CE(y, g_S(z_S))
L_kd = T^2 * KL(p_T || p_S)
L_core = mean_c ||mu_c^S - stopgrad(mu_c^T)||_2^2
```

Compact old-class cores:

```text
L_tail = mean_x max(0, ||z_S(x)-mu_y^S||_2 - alpha_r r_y^T)^2
```

Source-only boundary negatives are generated without true unknown classes:

```text
z_bridge = beta mu_a^T + (1-beta) mu_b^T + eps, a != b
z_shell  = mu_c^T + gamma r_c^T u, gamma > 1
z_looo   = source leave-one-class-out proxy episode
```

The boundary loss pushes bridge/shell/leave-out proxy samples outside all known acceptance envelopes:

```text
s_known(z) = max_c cos(z, mu_c^S)
L_boundary = mean_zneg max(0, s_known(zneg) - tau_reject)^2
```

Overflow suppression directly targets the observed failure mode:

```text
overflow(x) = 1[ ||z_S(x)-mu_y^S||_2 > r_y^T ]
L_overflow = mean_x max(0, ||z_S(x)-mu_y^S||_2 / r_y^T - rho_overflow)^2
```

The open-boundary monitor must report the same metrics that failed in R8/R9/R10:

|metric|desired movement|reason|
|---|---|---|
|`proxy_vaccept`|down|proxy/virtual boundary samples should not be accepted as known|
|`source_overflow`|down|old-class tails should not spill outside compact source envelope|
|`bridge_accept_rate`|down|inter-class bridge samples should be rejected|
|`low_density_accept_rate`|down|low-density old-neighborhood samples should defer/reject|
|`tail_accept_loss`|down|accepted tail mass must shrink|
|`overflow_accept_loss`|down|overflow samples must stop being accepted as core known|
|`radius_to_inter_ratio`|down|class radius must shrink relative to inter-class distance|
|`zid_p95/p99/tail_cvar`|down|tail angle must contract without reducing source/test TX accuracy|

## Launch Gate

The route is launchable only if the dry-run proves:

1. `WISIG_PKL` resolves to`ManySig.pkl`.
2. No`ManyTx.pkl` or target receiver file is used in training.
3. `real_unknown_classes_in_training=0`.
4. `target_receiver_samples_in_training=0`.
5. Teacher is exactly`ADV3B02_CORE90_SOFT_E200`.
6. Result claims are blocked: `stage2_success_claim=0` and `deployment_success_claim=0`.

## Success Criteria

Stage1 launch success is not paper success. A candidate can proceed toStage2-C qknn8 only if it satisfies both:

```text
test_tx_acc >= ADV3B02/R10 reference - 1 pp
proxy_vaccept, source_overflow, bridge_accept_rate, low_density_accept_rate all improve over R10_GENTLE
```

Final success remains the original Stage2-C same-row target: old99% with min_old95%, seen-new97% with min_seen93%, and unknown_reject99%. If the student only improves proxy metrics but fails qknn8 Stage2-C, the route is diagnostic.

## Next Implementation

Implement `launch_phase1_adv3b02_open_boundary_distill_20260706.sh` with two first candidates:

|candidate|focus|
|---|---|
|`OBD_CORE_TAIL_LOCK`|strong old core retention, aggressive source tail contraction|
|`OBD_BRIDGE_SHELL_REJECT`|bridge/shell rejection and low-density accept suppression|

Both candidates must export`phase2_zid_prototypes.pt/json` for the same qknn8 Stage2-C evaluator used in this report.
