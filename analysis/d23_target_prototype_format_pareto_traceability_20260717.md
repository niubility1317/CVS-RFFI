# D23 target-support prototype格式Pareto追踪

## 范围与锁定原则

D23只处理由目标receiver的合法`LEO_weak` support生成的旧类与新类prototype/operational class vector持久化格式。FP32是准确率基线；FP16和INT8是压缩候选。三者必须共享完全相同的support行、训练结果、类别顺序、半径/计数规则和逐样本全注册类决策，仅改变存储格式与对应score kernel。

格式选择只能使用K10 development support-fold结果和预登记batch=1部署代理基准，不得打开query，不得使用query标签、role、真实批类别数、quota、全局assignment或query图。选择锁定后K1/K5不得重新调scale、阈值或格式。

## 数学定义

对第`c`类support-derived单位向量`p_c∈R^D`：

- FP32：`s_c(x)=<normalize(x),p_c>`，持久化`p_c`为float32。
- FP16：持久化`h_c=fp16(p_c)`，评分时按目标kernel执行混合精度点积；不允许另存持久化FP32副本。
- INT8：`a_c=max_j|p_cj|/127`，`q_cj=clip(round(p_cj/a_c),-127,127)`，持久化`q_c∈int8^D`与`a_c∈fp16`，禁止值`-128`；评分为`s_c(x)=a_c<normalize(x),q_c>`。

每类同时记录统一的`radius`和`support_count`。K>=2时radius只由对应类support的leave-one-out余弦距离估计；K=1使用method-lock预登记`r0`，不得用self-distance伪造方差。

## Pareto锁定

在所有15个development场景×fold单元上，候选必须先满足相对FP32的逐类原子非劣和old-prefix freeze；然后按以下联合顺序选择：

1. floor与`H_old_new`不低于FP32容差门；
2. 持久状态更小；
3. batch=1平均/P95延迟更低；
4. 临时RAM/显存更小。

若INT8状态最小但延迟或floor受损，允许锁定FP16；不得仅因“位宽更低”宣称效率更高。

## 追踪矩阵

|ID|Source section|Requirement|Target files|Status|Verification|Notes|
|---|---|---|---|---|---|---|
|D23-01|用户新增要求|旧类与新类使用同一prototype bank schema，支持FP32/FP16/INT8|`code/cvsrffi/stage2_target_prototype_bank.py`|verified|`tests/test_stage2_target_prototype_bank.py`，14项PASS|bank API无query输入|
|D23-02|数学定义|INT8使用per-vector fp16 scale、禁止-128；FP16/FP32不保留额外高精度副本|同上|verified|dtype/member/state-byte tests PASS|零向量fail closed|
|D23-03|半径与K|记录radius/count；K1使用method-locked r0，K>=2仅support估计|同上|verified|K1与append锁定测试PASS|正式K1/K5 capsule隔离仍属D23-11/12|
|D23-04|冻结注册|Stage2-B旧类bank冻结，Stage2-C只能append；old prefix bytes/hash不变|同上；runner|implemented|codec prefix/hash/append测试PASS|runner raw score列证据仍pending|
|D23-05|可达路径|B3 FP32、FP16、INT8三格式候选进入同一support-only15fold runner|`code/scripts/run_d19_support_only_ciaf.py`|pending|runner tests+N607 log|不能只实现未调用模块|
|D23-06|评分|三格式对每个样本独立评分全部注册类，无query graph/role/quota/global assignment|codec；runner|pending|API schema/query-unreachable tests|INT8 MAC与rescale ops分开报告|
|D23-07|资源|报告实际状态、FP32等价状态、压缩倍数、MAC、平均/P95延迟、scratch、峰值显存|runner；`resource_audit.json`|pending|resource formula tests|逻辑状态与serialized bytes分列|
|D23-08|量化保真|support-only报告score误差、top1一致率、margin flip与逐类/fold差异|runner；training log|pending|paired support tests|不得用query调scale/门限|
|D23-09|持久artifact|输出before/after bank与path-free manifest，receipt绑定SHA|runner|pending|artifact hash/allowlist tests|NPZ必须`allow_pickle=false`|
|D23-10|Pareto选择|格式选择以support性能/floor优先，再状态/延迟；不预设INT8胜出|runner；selection|pending|selection tests|选择仅K10 development support|
|D23-11|协议|保持single-LEO、clean/source不可达、query只测和全Oracle禁令|runner；tests；report|pending|existing+new guard tests|任何缺失为`LOCAL_PROTOCOL_REPAIR_REQUIRED`|
|D23-12|N607验证|本地验证、Git提交、同步、短SSH运行、完整日志与报告|launcher；active report|pending|hash/bash-n/full-log audit|正式125前仅development support screen|

## 当前最高风险

B3的高均值来自FP32 operational weights，但其8/15fold原子非劣失败和0%floor说明压缩不是主要精度瓶颈。D23必须把“压缩保真”与“floor机制优化”分开：若FP16/INT8能配对复现B3，下一轮仍需专门优化`14-10/14-7/6-15/8-20`与新类N1的floor。
