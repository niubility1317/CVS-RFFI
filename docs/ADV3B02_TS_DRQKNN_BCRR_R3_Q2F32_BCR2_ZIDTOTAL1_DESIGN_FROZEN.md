# ADV3B02-TS-DRQKNN-BCRR/r3-q2f32-bcr2-zidtotal1 DESIGN_FROZEN

## 状态与首源

- 状态：`LOCAL_VERIFIED / NO_PERFORMANCE_RESULT`
- 独立监督：`MERGE / P0=0 / P1=0`
- parent：`ADV3B02-TS-DRQKNN-BCRR/r2-affine-bcr2-zidtotal1-bindfix1`
- parent run：`adv3b02_ts_drqknn_bcrr_r2_affine_bcr2_zidtotal1_bindfix1_full125_00b81000_20260724_005555`
- `DATA_PROTOCOL=PRESENT_REUSED / NOT_REVALIDATED`

parent在两个Stage2-C row的prediction前触发`affine actual branch audit/state drift`并健康止损。本地真实checkpoint support-only复算确定qKNN一平面codec在`K10/new10`的160条after support上出现1个小margin翻转，top1=`159/160=0.99375`、large-margin flip=`0`；BCR top1=`1.0`、flip=`0`。这是真实INT8生命周期失败，不是teacher binding、DA、BCRR、数据或query问题。

## 唯一机制delta

对每条修复后的raw support先计算`u=N(z_id)`，固定部署codec为：

1. 保留现有逐行affine INT8主平面、FP16 scale和FP16 offset，解码为`d0`。
2. residual=`u-d0`；`s_r=FP16(max(max(abs(residual))/127,2^-24))`；`q_r=clip(round(residual/FP32(s_r)),-127,127)`。
3. 部署support仅由`d=d0+FP32(q_r)*FP32(s_r)`重建，不持久化raw/FP32 support或FP32 residual。
4. 既有闭式class bandwidth仅在support build期计算为`h_raw`；持久状态固定为`h_hi=FP16(h_raw)`、`h_lo=FP16(h_raw-FP32(h_hi))`，部署使用`FP32(h_hi)+FP32(h_lo)`。
5. 禁止FP32 bandwidth sidecar、第三平面、按K/scene/class/audit切换codec、阈值变化或fallback。

## 不变量与生命周期

- K1/K5/K10同一codec，0参数、0 optimizer step，不增加物理样本。
- Stage2-C只append新类主平面、residual平面、逐行scale及`h_hi/h_lo`；旧类全部wire前缀逐字节冻结。
- DA、`z_id/z_dom`双qKNN公式、BCRR两级codec、四臂、zidtotal1 repair、全注册类统一竞争和完整125不变。
- query不得进入量化、bandwidth、audit、fallback或state更新。
- 正式support门不变：qKNN top1`>=0.995`、large-margin flip=`0`；BCR top1=`1.0`、any/large flip=`0`。
- 实际序列化state必须`<=256KiB`；26类K10额外state=`42,172B`，真实checkpoint smoke最大总wire=`159,691B`。

## 固定证据与立即证伪

两触发row×3场景真实checkpoint support-only spike为6/6 qKNN top1=`1.0`、large flip=`0`、最大logit误差`0.00108–0.00181`；BCR top1=`1.0`、flip=`0`；bandwidth重构误差`2.4e-8–4.3e-8`，query/truth读取0。单独增加第二平面而沿用旧bandwidth的方案已被falsifier拒绝，不得实现。

实现必须覆盖codec/wire round-trip、scale floor、K1/K5/K10、support/class置换、bit/row tamper、Stage2-C old-prefix、资源公式和两个触发row三场景真实checkpoint无query smoke。任一INT8/BCR/wire/old-prefix/资源/协议门失败即`REJECT`，不得再补第三平面、阈值或fallback。

本地`ssr-gpu`最终结果为目标测试`77 passed、3 Windows POSIX skipped、0 failed`，相邻DSSC`36 passed、0 failed`，`py_compile`与`git diff --check`通过。真实checkpoint复跑两触发row×三场景×before/after共12个state，qKNN与BCR top1均为`1.0`、翻转均为0、最大logit误差=`0.001622`、最大wire=`159,691B`，query/truth/apply打开数均为0。独立Terra终裁=`MERGE / P0=0 / P1=0`；完整bandwidth数值列表已从audit/append/state receipt移除，仅保留不可逆SHA和类数。

## 冻结文件范围

- `code/cvsrffi/stage2_adv3b02_ts_drqknn_bcrr.py`
- `code/scripts/run_adv3b02_ts_drqknn_bcrr_125.py`：只更新candidate/schema/revision对应校验，不改调度、健康门、矩阵或scorer
- `tests/test_stage2_adv3b02_ts_drqknn_bcrr.py`
- 本设计文档、目标文档、活动报告和新run报告

禁止修改模型、数据、coverage、authority、DA/BCRR公式、四臂、K、repair、scorer或完整125矩阵。
