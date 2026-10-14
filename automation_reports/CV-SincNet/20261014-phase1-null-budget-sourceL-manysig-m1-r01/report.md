# 周期参考正交补＋观测停止预算：源可行性预登记

本候选检验新的无标签监督目标。旧24维相对参考预测已因三个RX误差反向关闭，保留旧结果。新版本固定投影公共20点周期参考的占用频率与DC的正交补，不拟合响应增益/延迟；不声称实际RX信息被删除或TX独占。固定 bins7..13 共7复、14实坐标。visible `[80:100],[100:120],[140:160]`；held `[120:140]`。输入/校准只用visible，留出与outside全部zero，目标全detach。

可见三个观测均值m、样本方差v决定 s=mean(m²)、b=4v/3；b只是IID噪声动机的容差，实际缓存上游均衡和噪声独立性未知。候选损失 D=s+b+1e−8、w=s/D、a=relu((MSE−b)/D)，L=.1*sum(w*a/(1+a))/完整包数。满足预算时梯度0，每包贡献小于.1w；不按active子集归一化。当前仅固定观测预测的下游损失统计，未训练decoder或骨干。

## 唯一矩阵与先验源条件

Run `20261014-phase1-null-budget-sourceL-manysig-m1-r01`，唯一row L6300。复用原合法L_s，TX0至5、RX1/3/4/6/8、day1/2/3，90cell各70包；固定物理ID契约与nativeexactmatch。0U/V/target读取，标签仅在预测固定后用于分组/126000固定关系，0checkpoint/optimizer/EMA/头拟合。三baseline为zero_reference、cycle1、visiblemean，相同valid定义，不新增接受链。

必须有效覆盖≥95%，mean相对zero pooled MSE下降≥10%且所有RX不反向，固定跨RX身份gap≥.02并至少4/5 RX正；预算 meanweight≥.05，mean预测active≥5%overall且每RX≥1%。全部条件先于数据固定。失败保留完整证据并关闭，不扫bin、mask、方差倍数、eps、lambda或阈值。通过也只证明必要源可行性，进入训练需实际骨干梯度与default头同预算消融，不等于泛化有效。

## 发布与执行

实际代码/config Git身份是release_commit.txt（本次提交后填写）；当前普通非特权N607环境Python与前一源诊断相同。CWD `releases/cvs_null_budget_sourceL_20261014_r01`，CPU2threads、0GPU；源处理预计6300IQ读取、2零IQsmoke、0网络IQ前向/训练。发布 `python -X utf8 E:/type10-7/.codex_tmp/native_polyphase_runner.py experiments.cvs_repetition_feasibility.publish_null`；远端自动无查询fixture后直接source，command `python -u experiments/cvs_repetition_feasibility/runner.py resolved_request.json`。

唯一输出 `runs/20261014-phase1-null-budget-sourceL-manysig-m1-r01/L6300/source`；已有根不覆盖。一次preflight、archive一次SHA传输校验、remotecompile、一次独立livePID/CWD/argv/loggrowth，完整summary/census保留。只按预登记系统技术失败停止确切run-owned进程，不触碰健康任务；源科学失败不再训练扩展。预期runtime.json、observation_census.pt、summary.json、完整log及独立汇总复算。原dataVALIDATED_ONCE复用，不加hash/receipt/approval。该raw观测阶段不消费checkpoint，真实checkpoint smoke属于后续实际模型阶段。

SIG_TIME等已有正收益按各自配方独立保留，默认头原joint训练方式、三seed84%目标继续active；本候选尚无真实性能结果。

本地六套聚焦检查均VERIFIED（观测/损失/14维与旧24兼容/固定关系/两个源入口负例）。一次独立P0/P1 review PASS，未发现直接使下一实际诊断错误、越权、破坏或不可发布的缺陷；无额外审查或报告许可。
