# ADV3B02-TS-DRQKNN-BCRR/r2-affine-bcr2-zidtotal1-bindfix1 DESIGN_FROZEN

## 状态与边界

- 状态：`LOCAL_VERIFIED / NO_PERFORMANCE_RESULT`
- parent：`ADV3B02-TS-DRQKNN-BCRR/r2-affine-bcr2-zidtotal1`
- 独立可行性裁决：`MERGE_TECHFIX / 当前P0=0、P1=1；实现目标P0=0、P1=0`
- 独立实现终审：`MERGE / P0=0、P1=0、P2=2`；P2均不阻塞完整125发布
- 协议：继续复用`p2_min_v1`、`VALIDATED_ONCE`和GEOFF/r8 archive/manifest/parity/coverage，不改变received IQ、物理ID、receiver/TX、scene、K、support/query split或schema，不触发数据重验
- 科学机制：DA、`z_id/z_dom`双qKNN、BCRR、四臂、bank/codec、Stage2-C append、资源门和完整125均不变

## 触发证据与根因

parent run=`adv3b02_ts_drqknn_bcrr_r2_affine_bcr2_zidtotal1_full125_21ffdabf_20260723_234716`在两个不同K10/new20 row的prediction前触发同一`z_id repair/state teacher binding drift`。健康门已停止本run；成功6/125、partial prediction/score=`48/1000、72/1500`，终态=`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`，partial性能禁用。

设`N(x)=float32(float64(x)/||x||₂)`。repair receipt绑定`N(raw_zid)`；现有before-state先形成`ordered=N(raw_zid)`，随后`_make_actual_branch`再次调用raw helper，实际绑定`N(ordered)=N(N(raw_zid))`。FP32单位化不是严格字节幂等；构造的160维稀疏向量可稳定产生1 ULP差异。token/class排序保持row-token配对，不是根因。

## 唯一技术delta

1. `build_int8_qknn_state`显式区分`ordered_raw`与`ordered_unit=N(ordered_raw)`。
2. `ordered_unit`继续构造原affine INT8 bank、codec和决策状态。
3. `_affine_margin_audit`的teacher和`_make_actual_branch`只接收repair后的`ordered_raw`FP32 teacher；各消费点只单位化一次。
4. `_make_actual_branch`入口必须验证raw teacher为有限FP32`[N,160]`，并继续用token-bound单位化SHA闭合receipt。
5. 预排序raw output SHA仍由已有validator严格验证；canonical reorder后继续使用排序无关的token-unit binding，不以容差、自动raw/unit猜测或receipt SHA复制替代。
6. Stage2-C完整teacher遵循同一raw-teacher合同；before/after、old/new、class和scene完全同式。

## 决策与资源不变量

affine bank codes/scales/offsets、qKNN wire、metric、BCR权重codes/scales及正式query预测必须与parent相同；BCR权重仍只由同一decoded bank拟合。允许改变的仅是audit SHA、branch teacher binding和由其派生的state/receipt digest。query MAC、optimizer step和持久状态主体布局均不变；revision schema使JSON header增加9B，但不改变数组主体、资源公式或256KiB门。

## 冻结改动范围

- `code/cvsrffi/stage2_adv3b02_ts_drqknn_bcrr.py`
- `code/scripts/run_adv3b02_ts_drqknn_bcrr_125.py`（只更新launcher revision schema，不改调度、健康门、矩阵或scorer）
- `tests/test_stage2_adv3b02_ts_drqknn_bcrr.py`
- 本设计文档、目标文档、活动报告和后续新run报告

正式125 runner的调度、健康门、矩阵和scorer不得改变；只允许更新launcher revision schema并读取更新后的candidate/schema常量。禁止修改模型、数据、authority、coverage、DA/qKNN/BCRR公式、四臂、rank、K、fallback或资源上限。

## 冻结测试与立即证伪

- 显式非幂等FP32向量必须先证明`N(x)`与`N(N(x))`字节不同，再通过repair→before-state→runtime binding。
- K5/K10分别覆盖无零identity和单零medoid；donor可为非幂等行。
- support/token联合置换及class顺序置换后，teacher binding、bank wire、BCR权重和固定query prediction保持等价。
- token-row错配、raw teacher 1 ULP漂移或误传`ordered_unit`到raw-teacher接口必须失败关闭。
- interleaved Stage2-C append必须闭合repair/runtime/append三层binding和旧bank前缀。
- parent与bindfix1在相同输入上的bank wire、BCR部署权重和固定query logits必须逐字节一致；任一变化立即停止。
- 真实checkpoint support-only无query smoke覆盖两个触发row三场景，读取query/truth为0；随后独立review达到`P0=0、P1=0`才允许新commit、新run完整125。

## 本地闭合证据

- `ssr-gpu`下3个改动文件`py_compile`通过；目标测试`72 passed、3 Windows POSIX skipped、0 failed`，相邻DSSC测试`36 passed、0 failed`，`git diff --check`通过。
- K5/K10×无零/单零的bank wire、codes/scales/offsets、两级BCR部署权重和固定query logits均与parent逐字节一致；误传unit teacher、raw teacher 1 ULP漂移、token错配和float64输入均失败关闭。
- 真实checkpoint SHA=`2699eedcafe8cec880828592d2d65ba3781a9948939da5cf5c82b47143d59c98`的support-only smoke覆盖两个原触发row×3个scene，6/6 state与binding通过；query/truth/apply打开数均为0，fit query rows=0。
- smoke receipt=`automation_reports/CV-SincNet/ground_prototype_da_research_20260720/artifacts/adv3b02_r2_affine_bcr2_zidtotal1_bindfix1_real_checkpoint_support_only_smoke_20260723T164927Z.json`，SHA256=`b5e232d48fc07dbb1c744133204265e4b0d6634ef1ff299142bb23180c051474`。
- 独立终审=`MERGE / P0=0 / P1=0 / P2=2`。两个P2仅为schema header增加9B和后续可补commit-bound golden，不得延迟本次完整125。
