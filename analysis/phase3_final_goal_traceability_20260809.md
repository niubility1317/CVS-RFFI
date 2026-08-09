# Phase3阶段职责最终目标追溯表

状态：`G0_LOCAL_VERIFIED_INDEPENDENT_REVIEW_ALLOW`

日期：2026-08-09

权威来源：

- `E:\codex\home\attachments\c75febfd-60b9-42bb-9825-a0b3b9eda0bb\goal-objective.md`
- `E:\type10-7\项目.md`
- `E:\type10-7\AGENTS.md`

判定规则：`verified`必须有当前代码、测试、不可变artifact或完整同口径报告直接证明；设计、接口草稿、局部technical run、source proxy或历史Oracle只能记为`pending／deferred／rejected／blocked`，不得替代正式证据。

|ID|来源章节|可验收要求|目标文件／证据面|状态|验证|备注|
|---|---|---|---|---|---|---|
|P1-01|目标一、项目4|Phase1保持地面weak-label／semi-supervised source-domain DG，`rho_label<=0.1`且不执行多节点推理|Phase1数据划分、训练launcher、receipt|partial|CP矩阵source-only已闭合，完整`rho_label`目标待核|不得混入`R_t`或Phase3 unknown|
|P1-02|目标二、项目4.2|`source_known_train_tx／validation_tx／proxy_unknown_tx`身份互斥，proxy unknown全样本排除训练与选择回流|TX partition receipt、负测|partial|CP矩阵TX／checkpoint／NPZ绑定成立，但候选已拒绝|不同RX／信道view不得伪装unknown|
|P1-03|目标一|特征提取器同时输出可用的`z_id、z_dom、q、d_class、e_unknown、p_local`或严格等价接口|模型、exporter、bundle schema、真实checkpoint smoke|pending|仅`z_id／tx_logits`局部可达；其余字段缺失|报告字段不等于可达实现|
|P1-04|目标一、二|已知跨接收机／LEO floor、类内紧致／类间margin、proxy unknown低过度置信性得到同候选证据|完整Phase1矩阵和same-row指标|pending|CP-SFCEv2 clean门过，但LEO和proxy非补偿门失败|不得跨run拼最大值|
|P1-05|目标二|候选满足五项窄晋级条件且可从真实checkpoint导出deployment bundle|候选终态、postfreeze、bundle manifest|pending|当前G候选正式拒绝；没有可晋级checkpoint|source proxy不能写成真实unknown|
|P1-06|目标一、项目4.1|不可变bundle含特征提取器、注册类几何、半径／能量／尾部先验、质量／域不确定性|bundle schema、hash、runtime parity|pending|当前本地artifact无完整checkpoint绑定bundle／NPZ／runtime|只读模型知识，不含样本级cache|
|L-01|目标三、项目7.1|每节点先形成不可变本地证据，单节点只输出registered／unknown／defer且零query oracle|local evidence exporter、runtime、负测|partial|CARE严格接口与60行合成技术工件闭合；缺真实bundle本地runtime|`N_sat=1`基线必须独立可运行|
|P3-01|目标四、项目7.2-7.3|证明same-event并处理RX／信道／SNR差异、缺失／延迟、冲突和相关证据|CIRF event ledger、双轴融合、transcript测试|pending|设计已冻结，待实现|平均／投票仅基线|
|P3-02|目标5.1|协同unknown拒识，正式FAR≤5%、safe rejection≥95%，registered reject/defer计错|正式G2 prediction、truth-side scorer、风险证书|pending|待实现／采集|proxy不能替代真实unknown门|
|P3-03|目标5.2|比较独立适应、共享平均、质量加权、完整协同域状态并满足K10旧类门|Stage2-B/Phase3协同适应矩阵|pending|当前无support驱动协同域状态、四臂实现或结果|共享但不抹平节点差异|
|P3-04|目标5.3、项目7.4|unknown跨节点／过境关联为anonymous entity，不形成语义身份且零event回写|MHT实现、OOS／visibility测试、track artifact|pending|设计已冻结，待实现|registered不得生成anonymous track|
|P3-05|目标5.4|融合RFFI与位置／轨迹／认证／登记／调度证据，输出可信确权字段和`registration_authorized`|credential lifecycle、truth-blind fixture、不可变receipt|partial|CARE合成credential字段闭合；未接可信registry／真实外部证据|必须报告证据独立性和冲突|
|P3-06|目标5.5、项目7.4|授权后重新采集fresh-K独立事件、新`split_id`，交Stage2-C；历史unknown永不变support|lifecycle实现、fresh-K负测、Stage2-C handoff|partial|合成状态到`FRESH_K_READY_FOR_STAGE2_C`；未实际执行Stage2-C|Phase3不得直接更新Phase2|
|M-01|目标六|固定身份节点评价`N_sat in {1,2,3,4,5}`及节点子集、缺失、边际饱和|matrix launcher、subset receipt、same-event cache|verified_technical_only|CARE五档、N1恒等与60行合成矩阵闭合|尚无真实代理性能|
|M-02|目标六、项目7.3|一个event的多reception仍只计一个shot|event/reception schema、计数负测|pending|待审计|不得把五节点变五shot|
|M-03|目标六、项目8|非同步现有数据统一标`PROXY_MULTI_RECEIVER`，不声称在轨同步|artifact/report claim marker|pending|待审计|G1与G2必须隔离|
|M-04|目标七、项目7.5|同输入口径完成A/B/C/D，并报告`B-A、C-A、D-B-C+A`|matrix、pair scorer、interaction table|partial|CARE A/B/C/D合成行闭合；无目标定义的真实同row因果结果|不得只比较A与D|
|M-05|AGENTS四状态|涉及域适应／注册时同时报告`DA0_REG0、DA1_REG0、DA0_REG1、DA1_REG1`及四效应／交互|pair scorer、result schema|pending|代码、schema、测试和报告均无完整四状态执行结果|REG0新类指标为N/A|
|E-01|目标八|本地证据按节点缓存，组合复用cache，不为每个卫星组合重复backbone|export cache、matrix executor、resource receipt|pending|待审计|GPU≤2训练进程／卡|
|E-02|目标八|八卡只用于Phase1候选／节点特征提取等冻结矩阵，融合优先CPU，完整记录时延／字节／能耗|launcher、resource report|pending|待审计|发布须独立runner|
|R-01|目标九|所有阶段表述严格区分Phase1潜力、Phase3部署、授权和Stage2-C正式注册|项目文档、报告、README|pending|待审计|定义不构成完成声明|
|R-02|项目5.4、7|query真值／角色／配额／scorer零fit、零更新、零选择|负测、runtime receipt|pending|待审计|预测先封存、scorer后接truth|
|R-03|完整目标|完成状态必须由代码、完整矩阵和artifact逐项证明，不以设计或technical run替代性能|本追溯表、最终报告|pending|持续审计|任一必需项缺证据则目标保持active|

## CIRF-Track v3 G0当前修复门

以下三项是2026-08-09独立复核对当前未提交G0实现的可复现阻塞。三项全部通过聚焦回归和独立复审前，`P3-01／P3-04`不得从`pending`提升，G0不得发布。

|ID|对应目标|阻塞|验收证据|状态|
|---|---|---|---|---|
|G0-01|P3-01、R-02|网络调度状态仅校验调用方提供的`elapsed_ms`，已消费状态可被回放或重置累计时限|event authority预封存canonical replay store ID，并绑定authority receipt／opportunity／session／初始预算后内部签发root；所有正常终态在独占session锁下追加不可覆盖generation、父snapshot hash和transition chain；同进程及两个独立Python进程重建同session均不能恢复已消费root；外部latest回执锚定、深拷贝、receipt严格校验及恢复闭合|independent_review_verified|
|G0-02|P3-01|词典序QP把接近零但严格为正的特征值吞入零空间，可能破坏第一阶段主目标|二进制浮点目标在精确simplex等式下比较，同一全局谱分解只准精确零曲率方向进入二阶段；`1-2^-53`相邻可表示正定边界仍返回唯一主最优`[0.5,0.5]`|independent_review_verified|
|G0-03|P3-04|从未出现过但`opportunity_index`已越过N-scan截止点的迟到事件仍可新生track|迟到新hash返回`N_SCAN_FINALIZED_AUDIT_ONLY`，tracks／hypotheses／revision／event历史均不变|independent_review_verified|

首轮独立复核返回`P0=0、P1=2、REVISE`：指出外部root／终态重用及相邻可表示正曲率跨face合并仍可复现；G0-03已关闭。第二轮确认QP与同进程状态链关闭，但返回`P0=0、P1=1、REVISE`：类级内存registry在进程重启后可恢复root。第三轮确认跨进程append-only ledger能阻止正常重放，但返回新的`P1=1`：删除最新generation后，仍自洽的旧generation会被误认为最新head。最终实现增加外部封存的`prior_replay_receipt`，existing session恢复必须由该回执精确锚定最新generation／snapshot／transition head；缺回执、换store、删末代或回执与ledger不一致均fail-closed。第四轮冻结字节复核返回`P0=0、P1=0、ALLOW`。

修订后本地验证：`ssr-gpu`下从仓库根执行`py_compile`通过；`test_phase3_care_poe.py＋test_phase3_cirf_track_v3.py`为`49 passed`；跨进程测试显式封存子进程`PYTHONPATH`，真实双子进程同session负测为首进程`REQUEST`、第二进程`UNKNOWN_NETWORK_STATE`；移除末代generation后，携带原latest回执的第三进程被拒绝；CLI首次生成不可覆盖G0工件及两代append-only ledger，第二次同目录写入退出1；`git diff --check`通过（仅Windows换行提示）。上述仅为本地技术证据，不是性能或完整Phase3完成证据。

## 当前审计顺序

1.盘点Phase1真实候选、checkpoint、postfreeze和bundle字段；
2.定位现有Phase3代码、测试、生命周期和矩阵executor；
3.核对automation reports中technical与performance证据边界；
4.选择最大可闭环缺口，只实现一个可运行纵切；
5.本地验证、独立P0／P1复核、Git提交后再决定是否发布N607。
