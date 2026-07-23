# ADV3B02-TS-DRQKNN-BCRR/r2-affine DESIGN_FROZEN

## 1.身份、状态与裁决

- candidate：`ADV3B02-TS-DRQKNN-BCRR/r2-affine`
- 状态：`DESIGN_DRAFT -> FEASIBILITY_REVIEW -> DESIGN_FROZEN`
- 独立监督：`MERGE / P0=0 / P1=0`，只批准进入本地实现，不代表release或性能成功。
- 上一revision：`r1`因真实checkpoint support-only检查在seed713104 after的clear与low-elev触发共享qKNN INT8门，同时独立review发现after teacher自指和125无系统故障停派，状态为`SUPERSEDED_TECHNICAL_REVISION / NO_PERFORMANCE_RESULT`。
- 外部设计输入：`C:\Users\lh594\Downloads\ADV3B02_双分支双注册qKNN快速适应设计报告_20260723.md`，SHA256=`cfe29eb87519c7582a2822ddbd2c8d2e80c363bede471d06b8c310c50a5a42a1`。
- 数据：只复用现有`p2_min_v1`、`VALIDATED_ONCE`、GEOFF/r8 archive与coverage；本revision不改变数据，不得重复验证。

## 2.唯一机制与四臂

域适应、qKNN和OTHER保持r1冻结公式：

1.域适应：target-old support在`z_dom`中构造TX抑制的固定2槽类内域邻域；Stage2-C冻结旧状态并append新类。
2.qKNN：`z_id`Student-t qKNN对全部注册旧类和新类执行逐query统一竞争。
3.OTHER：连续BCRR只处理qKNN剩余类级残差、old/new竞争和floor，不读取`z_dom`。

|arm|决策|
|---|---|
|`M0`|仿射INT8 `z_id`Student-t qKNN|
|`M_DA`|TX抑制`z_dom`类内条件化＋同一仿射INT8 `z_id`qKNN|
|`M_OTHER`|`M0`＋BCRR|
|`M_JOINT`|`M_DA`＋BCRR|

四臂逐字节共享同一仿射codec合同；codec是共同部署底座的技术修复，不是第五个性能机制。DA、BCRR、K集合、rank、阈值、fallback、场景、数据和完整125性能门均不变。

## 3.仿射INT8 support codec

对每条合法support的`z_id`向量：

```text
x      = L2Normalize(z_id)
scale  = float16((max(x)-min(x))/254)
offset = float16((max(x)+min(x))/2)
code   = clip(rint((x-float32(offset))/float32(scale)),-127,127).astype(int8)
x_hat  = L2Normalize(float32(code)*float32(scale)+float32(offset))
```

冻结数值合同：

- `rint`为IEEE round-to-nearest-even；code范围固定`[-127,127]`。
- wire中的code为1字节有符号整数，scale/offset为little-endian FP16；不得持久化FP32 support或残差sidecar。
- 非有限输入、零范数向量、`max(x)-min(x)<=0`、FP16 scale下溢为0或decode非有限均fail-closed。
- class bandwidth仍由解码后的support按既有统一公式产生并存FP16；Student-t kernel、temperature、类归一化和全注册类竞争不变。
- K、receiver、seed、scene、class、old/new角色或审计结果不得改变codec公式。
- 旧对称codec必须保持显式只读兼容；r2 wire/schema必须加法版本化，禁止静默猜测、重编码或混用receipt。

## 4.量化审计与真实teacher

量化门沿用目标文档既有定义：

```text
top1_agreement >= 0.995
large_margin_flip_count == 0
large = teacher_margin > 2 * row_max_abs_logit_error
```

- teacher support必须是当前state完整、未量化的FP32 support，并以FP64计算teacher logits。
- Stage2-C runner可把同一after enrollment包已经计算的完整`all_zid`传给只读audit；append更新仍只能使用new support，旧state不得重估。
- deployment logits必须从实际序列化affine bytes反解后重算。
- `any_margin_flip_count`只作诊断，不替代large-margin门，也不得被隐藏。
- receipt必须绑定完整teacher SHA、serialized wire SHA、codes/scales/offsets、class scale、support token顺序、top1、any flip、large flip和`query_rows_used_for_fit=0`。

## 5.K1/K5/K10、决策几何与资源

- K1：域散度不可辨识，精确满足`M_DA=M0`、`M_JOINT=M_OTHER`；codec仍按同一公式编码support。
- K5：首个DA性能falsifier；不增加rank、参数或codec分支。
- K10：按同一公式确认；不得依据support审计或125结果切换codec。
- 仿射codec会相对对称INT8改变近似余弦几何；只能声明serialized-byte审计满足部署保真门，不能宣称几何不变。
- 参数与optimizer step均为0；每条support只比单scale增加1个FP16 offset，即2B。
- C=26时K1/K5/K10分别增加52B、260B、520B；必须实测单arm wire目标`<128KiB`且硬限`<=256KiB`。

support-only可行性证据覆盖3个失败seed×before/after×3场景共18包，未打开query/truth。最低top1为`259/260=0.996154`；仅713105、713106 rain-after各有1个any flip，对应teacher margin为`0.005615/0.003006`、row最大logit误差为`0.124130/0.093048`，large-margin flip均为0。该证据不是性能结果。

## 6.完整125健康退出

矩阵固定为125 jobs、375 scene slices、1500 score rows和1000 arm-state prediction artifacts。调度器必须采用有界增量派发，最多保持GPU worker数目的活动row：

1.任一P0协议/安全错误立即停止新增派发。
2.两个不同row在prediction发布前出现同一确定性异常fingerprint时，立即停止新增派发。
3.取消尚未启动的future；对已启动row先核对PID、CWD、cmdline和run root，再只终止本run进程树。
4.保留partial artifact、日志、已启动/完成/失败计数和fingerprint receipt，终态为`STOPPED_EARLY_SYSTEMIC_TECHNICAL_FAILURE / NO_PERFORMANCE_RESULT`。
5.不得按accuracy、H、BA或其它性能值早停；技术健康后必须跑完完整125。

## 7.立即falsifier

出现任一项即停止r2-affine，不得发布N607：

- 真实checkpoint no-query smoke任一预注册support包`top1<0.995`或large-margin flip>0。
- teacher不是完整未量化FP32 support，after仍使用decoded-old，或query/truth打开数非0。
- K1不再精确identity，Stage2-C改写旧codes/scales/offsets、旧token前缀或旧域状态。
- wire roundtrip、类轴/行置换、round/端序、零range或receipt负例失败。
- 单armwire超过256KiB，出现持久FP32 sidecar或未审计资源。
- 注入两个不同row的同fingerprint prediction前失败后仍有新row被派发，或终止了非本run进程。

满足本地falsifier只表示`LOCAL_VERIFIED`。必须再经独立`P0=0/P1=0`review、Git提交和全新不可覆盖run ID，才允许唯一Terra runner发布完整125。

## 8.冻结改动范围

允许修改：

1.`docs/STAGE2_METHOD_RESEARCH_GOAL.md`
2.`docs/ADV3B02_TS_DRQKNN_BCRR_R2_AFFINE_DESIGN_FROZEN.md`
3.`code/cvsrffi/stage2_adv3b02_ts_drqknn_bcrr.py`
4.`code/scripts/run_adv3b02_ts_drqknn_bcrr_125.py`
5.`tests/test_stage2_adv3b02_ts_drqknn_bcrr.py`
6.仅在无法候选内闭合wire时，以加法版本化方式修改共享qKNN序列化器及其专项测试；旧codec必须字节兼容。

不得修改模型、Phase1 checkpoint、数据builder、authority、coverage、GEOFF/r8、scorer、DA公式或BCRR公式。若实现需要改变核心输入、loss、DA、head或适应规则，必须创建新revision并重新可行性审查。
