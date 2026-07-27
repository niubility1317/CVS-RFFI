# Starlink与星上RFFI文献审计

审计日期：2026-07-28
审计角色：卫星通信、物理层安全与RFFI文献审稿人
审计对象：`manuscript.tex`与`references.bib`
修改边界：本文件仅给出文献、论证与措辞建议，不改动主稿或参考文献库。

## 1.审稿结论

Starlink与本文的关联是**系统架构层面的强动机**，不是数据或实验层面的直接证据。最准确的连接点不是笼统的“Starlink属于LEO”，而是Starlink Direct to Cell已经把具备eNodeB功能的接收与处理能力放到LEO卫星上，使普通LTE手机和IoT终端能够直接向星上载荷发送信号。该架构将“地面发射机—高速运动的星载接收机”从设想变成了真实商业系统中的链路方向，与本文“地面训练、星上部署，由星载接收机识别地面发射机”的方向一致。

这种一致性只到任务方向为止。本文没有使用Starlink信号、终端、波形、协议、卫星射频前端或飞行处理器，也没有复现其链路预算和接入流程。WiSig/ManySig仍是地面代理数据，当前LEO算子仍是接收后残余基带仿真。因此，Starlink只能用于说明研究问题为何重要，不能用于说明方法已经在Starlink或真实在轨环境中有效。

星上RFFI的价值应表述为：在星载接收机入口处提供一个与高层标识独立的、由发射硬件产生的辅助身份线索，用于终端注册、凭据与物理发射机一致性检查、可疑重放/欺骗的初筛，以及后续干扰源归因。它应当与密码认证、协议校验和位置/信道证据联合使用，而不是替代密码学。SatIQ明确把密码认证视为首要防线，并把指纹视为在密码不可用、失效或不足以覆盖特定重放情形时增加置信度的手段。

“必要性”不宜写成“Starlink必须采用RFFI”。当前公开材料没有证明Starlink使用RFFI，也没有公开证据表明其现有认证机制存在本文所针对的缺口。更稳妥的顶刊措辞是：大规模LEO直连终端系统使**跨星载接收机、少标签、低状态开销的物理层身份验证成为值得研究的能力**；在不能持续下传原始IQ、接触时间短、接收链频繁变化的条件下，紧凑的星上RFFI具有明确的体系价值，但其必要性和收益仍需真实硬件、真实链路和威胁模型验证。

当前稿件还有一个需要立即澄清的方向性问题：Related Work中的“Satellite Transmitter Fingerprinting”主要讨论PAST-AI和SatIQ。这两项工作由**地面接收机识别卫星发射机，即下行指纹**；本文主要任务则是**星载接收机识别地面发射机，即上行指纹**。二者能够证明卫星链路保留可辨识硬件信息，却不是同一链路方向。建议把小节改为“RFFI over Satellite Links: Downlink Evidence and Uplink Deployment”，并明确指出这一证据迁移边界。

## 2.Starlink与本文场景的准确映射

| Starlink/LEO系统事实 | 本文中的对应抽象 | 仍未覆盖的差距 |
|---|---|---|
| Starlink Direct to Cell在卫星上配置eNodeB功能，直接连接普通LTE手机和IoT设备 | 地面终端是待识别发射机，卫星载荷是部署接收机 | 本文使用WiSig/ManySig代理，不是LTE Direct to Cell波形或Starlink前端 |
| LEO卫星高速运动，链路必须处理Doppler、时延、低终端增益和低上行功率 | 相同发射机身份信号叠加时变传播与同步残差 | 当前算子是残余基带模型，不是完整链路预算，也没有星历驱动的连续过境轨迹 |
| 地面—卫星连接随轨道运动发生高频链路变化和切换 | 同一终端可能被不同卫星、波束和接收链观测，形成跨接收机域偏移 | 当前实验用地面接收机代理卫星接收机，没有真实多星接收前端 |
| 大型星座把大量接入点和计算节点放到轨道上 | 适合采用“地面训练、星上紧凑推理与少样本注册”的生命周期 | 当前只报告模型/分类头状态，尚未测量目标处理器上的延迟、RAM、功耗与容错 |
| 新终端、替换终端和IoT设备持续加入系统 | 部署后的few-shot新类注册 | 仍缺真实终端到星载接收机的新类注册实验和时间漂移实验 |
| 物理层信号先于或伴随高层身份到达接收机 | RFFI可作为独立的硬件身份线索 | 需要定义与现有密码认证、SIM/eSIM、接入控制的融合位置和威胁模型 |

由此得到的核心论证应是：

> Starlink Direct to Cell是本文“星载接收机识别地面终端”方向的商业系统例证；LEO运动和多卫星接入使跨接收机鲁棒性成为系统问题；星上RFFI可以提供紧凑的辅助物理身份线索；本文只研究该问题的代理实现，不宣称Starlink兼容性或在轨有效性。

## 3.逐条claim—source映射

### 证据等级

- **R1—官方监管/标准**：适合证明授权、系统规模、频段和标准信道事实。
- **P1—同行评审原始研究**：适合支撑已经被实验或正式模型验证的技术事实。
- **P2—同行评审综述/愿景论文**：适合支撑体系动机、研究趋势和开放问题。
- **O1—运营商官方材料**：适合证明该运营商公开描述的架构和业务状态；必须使用“SpaceX reports/describes”等归属式措辞，不能作为独立安全评估。
- **A1—预印本**：只用于补充最新方向，不应承担主论点。

| ID | 可支持的claim | 核心来源 | 等级 | 推荐措辞 | 主要风险 |
|---|---|---|---|---|---|
| C1 | Starlink是大型LEO系统的现实例证 | FCC 22-91在2022年授权最多7,500颗Gen2卫星部署于525、530和535 km轨道壳层 | R1 | “A 2022 FCC order authorized a tranche of up to 7,500 Gen2 Starlink spacecraft.” | 不要把授权数写成当前在轨数；数量随时间变化 |
| C2 | Starlink Direct to Cell把蜂窝基站功能放到卫星上 | SpaceX《Starlink Direct to Cell Service Now Available》 | O1 | “SpaceX describes Direct to Cell satellites as orbital cellular base stations connecting ordinary LTE phones.” | 运营商自述具有宣传性；不能据此推断安全架构 |
| C3 | Direct to Cell链路包含星载接收机接收低功率地面终端上行 | 同一SpaceX官方文件：普通4G LTE手机、1.6–2.7 GHz合作频谱、低天线增益和发射功率 | O1 | “The architecture places a moving receiver above low-power terrestrial transmitters.” | 本文不是该频段、波形或链路预算 |
| C4 | LEO运动导致Doppler、时延和地面—卫星链路动态 | SpaceX官方文件；3GPP TR 38.811；Kassing等人的Hypatia | O1+R1+P1 | “Orbital motion changes link geometry and produces ground–satellite link churn.” | Hypatia是网络仿真，不是RF前端测量 |
| C5 | 大型LEO网络中，同一终端可能由变化的卫星/链路服务 | Hypatia对高速轨道运动和ground-to-satellite link churn的分析 | P1 | “A terminal can be observed through changing satellite links and receiver chains.” | “不同卫星必然对应独立硬件域”仍是合理推论，不是该文直接测量结果 |
| C6 | 卫星链路保留可用于指纹识别的发射机硬件信息 | PAST-AI使用589小时、超过100M IQ样本的真实Iridium数据 | P1 | “Real Iridium measurements show that transmitter-specific information survives a LEO downlink.” | 这是卫星发射机到地面接收机的下行，不是本文上行 |
| C7 | 高频率指纹能提高低成本SDR重放的攻击门槛 | SatIQ在真实Iridium数据和有线重放设置下报告EER 0.120、ROC AUC 0.946 | P1 | “SatIQ shows that high-rate fingerprints can raise the cost of SDR replay in a specific downlink setting.” | 不能写成对所有欺骗攻击有效；高采样率攻击仍可能伪造 |
| C8 | 少样本卫星IoE识别可以采用地面训练、LEO部署 | Zhao等人的GSGL把地面训练SEI模型部署到LEO卫星并研究样本不足 | P1 | “GSGL establishes a ground-trained/onboard-deployed SEI precedent under sample scarcity.” | GSGL不等价于本文跨接收机、support-only旧/新类注册协议 |
| C9 | 星上推理有低时延、减少数据搬运和自主处理的体系价值 | Wang和Li的Satellite Computing；Furano等人的edge AI论文 | P2 | “Near-receiver processing can reduce data movement and support timely local decisions.” | 当前稿件没有量化原始IQ下传节省或端到端时延 |
| C10 | 星上计算受功耗、计算、存储和恶劣环境约束 | Wang和Li；Furano等人 | P2 | “Onboard deployment imposes power, compute, memory, and reliability constraints.” | 不能由通用卫星计算文献推断本文已满足这些约束 |
| C11 | 星载观测地面发射机可联合支持识别、认证和归因 | Hendy等人的卫星RF地理定位与发射机指纹综述 | P2 | “Spaceborne RF sensing motivates joint localization and emitter identification.” | 综述不是本文算法的直接实验；文章较新，应核查期刊卷期 |
| C12 | RFFI应作为密码认证的补充，而非替代 | SatIQ的威胁模型和动机部分 | P1 | “RFFI provides an auxiliary hardware-bound cue alongside cryptographic authentication.” | 不要写“RFFI优于密码学”或“Starlink认证不安全” |
| C13 | 当前研究只提供代理证据 | 本文数据协议与稿件限制段 | 内部事实 | “Starlink anchors the systems motivation; no Starlink signal or flight hardware is used.” | 该边界句必须保留，不能被引言宣传性措辞抵消 |

## 4.星上RFFI的意义、作用与必要性

### 4.1意义

1. **把网络身份绑定到实际发射硬件。**高层设备标识、账号或证书描述的是逻辑实体；RFFI观察的是振荡器、混频器、功放、时钟和调制链产生的细微失真。二者联合时，可检查“合法凭据是否由预期硬件发出”。
2. **在星载接收入口提供早期筛查。**身份或异常分数可以在完整上层会话建立前形成，为后续认证、接入控制或人工分析提供额外证据。
3. **支持部署后的终端注册。**LEO直连IoT、应急通信和远端接入会持续遇到训练阶段未见终端。少样本注册允许新增合法设备，同时维持旧设备识别能力。
4. **支持跨卫星持续识别。**终端在卫星、波束或载荷之间切换时，接收链与传播状态变化。若指纹模型只在一个接收机上成立，就难以作为星座级身份线索。
5. **降低原始IQ长期搬运需求。**在星上形成紧凑身份状态和事件级输出，理论上可减少持续下传或保留完整IQ的需求。当前论文尚未量化这项收益。

### 4.2可承担的作用

- 终端或发射机的辅助注册与再识别；
- 凭据—发射硬件一致性检查；
- 低成本SDR重放、仿冒或克隆后的异常筛查；
- 非法或干扰发射机的归档与后续归因；
- 在地面回传中断或时延过大时生成本地告警；
- 为后续密码认证、位置验证、TDoA/FDoA、信道证据和行为分析提供一个独立分量。

### 4.3“必要性”的安全写法

可写：

> The scale and mobility of emerging LEO access systems make receiver-robust, data-efficient physical-layer identity evidence operationally relevant.

可写：

> Onboard RFFI is attractive as a defense-in-depth signal because it can bind an observed waveform to transmitter hardware while retaining only a compact decision state.

不可写：

> Starlink requires RFFI to be secure.

不可写：

> RFFI replaces cryptographic authentication.

不可写：

> Our method prevents Starlink spoofing.

## 5.建议的Introduction论证结构

建议把Starlink放在当前Introduction第一段定义RFFI之后、当前“observable waveform is not a transmitter-only object”之前。逻辑顺序应为：

1. 定义RFFI及其硬件身份价值；
2. 用Starlink/Direct to Cell说明星载接收地面终端已经是现实系统方向；
3. 从LEO运动推导跨卫星接收链和传播变化；
4. 说明星上RFFI的辅助身份、少样本注册和紧凑状态价值；
5. 明确密码学互补关系以及Starlink仅是动机；
6. 再进入本文的接收机—信道混淆、标签稀缺、旧/新类平衡和现有工作空白。

Starlink在正文中出现一至两次即可。除非加入真实Starlink实验，不建议在标题、摘要或贡献列表中写“Starlink RFFI”。

### 5.1推荐的英文展开版

```latex
Large LEO constellations make this deployment problem concrete. In 2022, the U.S. Federal Communications Commission authorized a tranche of up to 7,500 SpaceX Gen2 Starlink spacecraft, while Starlink's Direct to Cell architecture places an eNodeB-capable payload in orbit to connect ordinary LTE phones and IoT modems \cite{fcc2022starlinkgen2,spacex2025directtocell}. Unlike a stationary terrestrial base station, this receiver moves at orbital velocity; Doppler, timing, link budget, and ground--satellite association vary as the serving spacecraft changes \cite{kassing2020hypatia,3gpp38811}. From an RFFI perspective, the same ground transmitter can therefore be observed through changing propagation states and satellite receive chains. Starlink is used here as a systems-level example, not as an experimental platform: our data contain no Starlink waveform, terminal, satellite front end, or protocol trace.

At the satellite ingress, RFFI can supply a hardware-bound identity cue before or alongside higher-layer authentication. Such a cue can support terminal enrollment, credential--emitter consistency checks, and first-stage screening of spoofed or replayed transmissions. Real Iridium studies show that LEO downlinks retain transmitter-specific impairments and that high-rate fingerprints can raise the cost of SDR replay attacks \cite{oligeri2023pastai,smailes2023satiq}; GSGL studies the complementary ground-trained/onboard-deployed SEI direction under sparse observations \cite{zhao2026gsgl}. These results establish feasibility on related satellite links, but they do not solve the present problem: support-only adaptation and old/new registration on a receiver absent from ground training.

Onboard processing is valuable only if the decision state fits the flight processor. Processing observations near the receiver can reduce raw-IQ transport and shorten the path to an identity or anomaly flag, but satellite computing remains constrained by power, computation, storage, and the operating environment \cite{wang2023satellitecomputing,furano2020edgeai}. This motivates our ground-trained, sealed representation and compact deployment-time classifier. It does not establish flight readiness; end-to-end latency, peak memory, energy, numerical equivalence, and radiation-tolerant execution remain unmeasured.
```

### 5.2版面受限时的英文压缩版

```latex
Large LEO access systems make the problem operational rather than hypothetical. Starlink Direct to Cell, for example, places an eNodeB-capable payload in orbit to connect ordinary LTE phones and IoT modems, while orbital motion introduces Doppler, timing, link-budget, and handover variation \cite{spacex2025directtocell,kassing2020hypatia,3gpp38811}. A ground terminal may consequently be observed through changing satellite receive chains, turning receiver robustness into a constellation-level requirement. At this ingress point, RFFI can complement cryptographic authentication with a hardware-bound cue for enrollment, credential--emitter consistency, and spoof/replay screening \cite{oligeri2023pastai,smailes2023satiq}. Onboard use also favors a compact state because satellite computing is power-, memory-, and reliability-constrained \cite{wang2023satellitecomputing}. Starlink anchors this systems motivation only; the present study uses neither Starlink signals nor flight hardware.
```

### 5.3建议重写的卫星相关工作段

```latex
\subsection{RFFI over Satellite Links: Downlink Evidence and Uplink Deployment}

Existing satellite RFFI evidence spans two different link directions. PAST-AI and SatIQ use real Iridium downlinks: a ground receiver fingerprints satellite transmitters, establishing that hardware-dependent information can survive a time-varying LEO path and, in SatIQ, assist replay detection \cite{oligeri2023pastai,smailes2023satiq}. Our deployment direction is the reverse. A spaceborne receiver must identify terrestrial transmitters, as considered in ground--satellite SEI and spaceborne RF-sensing research \cite{zhao2026gsgl,hendy2026beyondgnss}. The distinction matters because the receiver hardware, uplink budget, terminal population, and enrollment process change. None of these studies jointly evaluates a ground-trained representation on an unseen spaceborne receiver, followed by support-only adaptation of old identities and registration of new identities under one all-class decision rule.
```

### 5.4建议在Introduction末尾保留的边界句

```latex
Starlink and other LEO access constellations motivate the receiver-side lifecycle considered here; they are not data sources or validation platforms for this study. WiSig/ManySig remains a terrestrial proxy, and the residual LEO operator is a simulation rather than a Starlink air-interface or in-orbit measurement.
```

## 6.相关工作中的语义修正

当前稿件可作以下定向调整：

| 当前位置 | 当前风险 | 建议 |
|---|---|---|
| Introduction中“A spaceborne monitoring lifecycle...” | 只有抽象卫星背景，缺少现实系统锚点 | 在此前加入Starlink Direct to Cell及跨卫星接收链段落 |
| Related Work的“Satellite Transmitter Fingerprinting” | 容易让读者误以为PAST-AI/SatIQ与本文链路方向相同 | 改为“RFFI over Satellite Links: Downlink Evidence and Uplink Deployment” |
| “Satellite fingerprinting establishes feasibility on real downlinks” | 结论基本正确，但没有说明本文是上行接收 | 补充“related-link feasibility, not uplink validation” |
| 星上资源约束 | 当前只引用GSGL，支撑范围不足 | 增加Satellite Computing或edge AI文献 |
| 星上RFFI安全作用 | 容易被写成替代密码学 | 明确“auxiliary/defense-in-depth cue” |
| Starlink与本文 | 容易产生品牌兼容性暗示 | 每个Starlink段落末尾保留“systems motivation only”边界句 |

## 7.建议新增的BibTeX条目

以下条目均已在2026-07-28通过Crossref、官方页面或出版社页面核对。PAST-AI、SatIQ、GSGL和3GPP TR 38.811已经存在于`references.bib`，不应重复添加。

### 7.1优先级A：建议现在加入

```bibtex
@techreport{fcc2022starlinkgen2,
  author      = {{Federal Communications Commission}},
  title       = {Space Exploration Holdings, LLC: Request for Orbital Deployment and Operating Authority for the {SpaceX Gen2 NGSO} Satellite System},
  institution = {Federal Communications Commission},
  type        = {Order and Authorization},
  number      = {FCC 22-91},
  year        = {2022},
  month       = dec,
  url         = {https://docs.fcc.gov/public/attachments/FCC-22-91A1.pdf}
}

@misc{spacex2025directtocell,
  author       = {{SpaceX}},
  title        = {{Starlink Direct to Cell Service Now Available}},
  year         = {2025},
  month        = feb,
  howpublished = {Online},
  url          = {https://starlink.com/public-files/DIRECT_TO_CELL_SERVICE_FEB_25.pdf},
  note         = {Accessed: Jul. 28, 2026}
}

@inproceedings{kassing2020hypatia,
  author    = {Simon Kassing and Debopam Bhattacherjee and Andr{\'e} Baptista {\'A}guas and Jens Eirik Saethre and Ankit Singla},
  title     = {Exploring the ``Internet from Space'' with {Hypatia}},
  booktitle = {Proc. ACM Internet Measurement Conf. (IMC)},
  year      = {2020},
  pages     = {214--229},
  doi       = {10.1145/3419394.3423635}
}

@article{wang2023satellitecomputing,
  author  = {Shangguang Wang and Qing Li},
  title   = {Satellite Computing: Vision and Challenges},
  journal = {IEEE Internet Things J.},
  year    = {2023},
  volume  = {10},
  number  = {24},
  pages   = {22514--22529},
  doi     = {10.1109/JIOT.2023.3303346}
}
```

### 7.2优先级B：用于资源边界或上行场景补强

```bibtex
@article{furano2020edgeai,
  author  = {Gianluca Furano and Gabriele Meoni and Aubrey Dunne and David Moloney and Veronique Ferlet-Cavrois and Antonis Tavoularis and Jonathan Byrne and Leonie Buckley and Mihalis Psarakis and Kay-Obbe Voss and Luca Fanucci},
  title   = {Towards the Use of Artificial Intelligence on the Edge in Space Systems: Challenges and Opportunities},
  journal = {IEEE Aerosp. Electron. Syst. Mag.},
  year    = {2020},
  volume  = {35},
  number  = {12},
  pages   = {44--56},
  doi     = {10.1109/MAES.2020.3008468}
}

@article{hendy2026beyondgnss,
  author  = {Nermine Hendy and Bisma Manzoor and Ferdi Ganda Kurnia and Fernando Moya Caceres and Akram Al-Hourani},
  title   = {Beyond {GNSS}: A Survey and Tutorial on Satellite-Based Radio Frequency ({RF}) Geolocation and Emitter Fingerprinting},
  journal = {npj Wireless Technology},
  year    = {2026},
  volume  = {2},
  number  = {1},
  pages   = {37},
  doi     = {10.1038/s44459-026-00045-y}
}
```

### 7.3优先级C：实验与部署讨论可选

```bibtex
@inproceedings{liu2025spaceexit,
  author    = {Jiacheng Liu and Xiaozhi Zhu and Tongqiao Xu and Xiaofeng Hou and Chao Li},
  title     = {{SpaceExit}: Enabling Efficient Adaptive Computing in Space with Early Exits},
  booktitle = {Proc. USENIX Annu. Tech. Conf. (USENIX ATC)},
  year      = {2025},
  pages     = {1343--1358},
  publisher = {USENIX Association},
  url       = {https://www.usenix.org/conference/atc25/presentation/liu-jiacheng}
}
```

SpaceExit是地球观测计算系统，不应被写成RFFI证据。它只适合支撑“星上推理必须测量真实硬件资源和采用自适应执行”的Discussion或Future Work。

## 8.来源核验与下载入口

| 来源 | 核验状态 | DOI/正式页面 | 可下载PDF或官方文件 |
|---|---|---|---|
| FCC 22-91 | 官方监管文件 | [FCC正式页面](https://docs.fcc.gov/public/attachments/FCC-22-91A1.pdf) | [PDF](https://docs.fcc.gov/public/attachments/FCC-22-91A1.pdf) |
| Starlink Direct to Cell Service Now Available | SpaceX官方材料，2025-02 | [官方文件](https://starlink.com/public-files/DIRECT_TO_CELL_SERVICE_FEB_25.pdf) | [PDF](https://starlink.com/public-files/DIRECT_TO_CELL_SERVICE_FEB_25.pdf) |
| Exploring the “Internet from Space” with Hypatia | Crossref与ACM元数据一致 | [DOI](https://doi.org/10.1145/3419394.3423635) | [作者公开PDF](https://bdebopam.github.io/papers/imc2020-hypatia.pdf) |
| Satellite Computing: Vision and Challenges | Crossref与IEEE元数据一致 | [DOI](https://doi.org/10.1109/JIOT.2023.3303346) | [作者公开PDF](https://wangshangguang.github.io/assets/Satellite_Computing.pdf) |
| Towards the Use of AI on the Edge in Space Systems | Crossref与IEEE元数据一致 | [DOI](https://doi.org/10.1109/MAES.2020.3008468) | [IEEE记录](https://ieeexplore.ieee.org/document/9288809) |
| PAST-AI | 已在本稿引用；Crossref核验通过 | [DOI](https://doi.org/10.1109/TIFS.2022.3219287) | [CC BY PDF](https://pure.tue.nl/ws/portalfiles/portal/261348390/PAST_AI_Physical_Layer_Authentication_of_Satellite_Transmitters_via_Deep_Learning.pdf) |
| SatIQ/Watch This Space | 已在本稿引用；Crossref与ACM/Oxford元数据一致 | [DOI](https://doi.org/10.1145/3576915.3623135) | [arXiv PDF](https://arxiv.org/pdf/2305.06947) |
| GSGL | 已在本稿引用；Crossref核验通过 | [DOI](https://doi.org/10.1109/JIOT.2025.3600517) | [IEEE记录](https://ieeexplore.ieee.org/document/11130521) |
| Beyond GNSS | Crossref与Nature页面一致 | [DOI](https://doi.org/10.1038/s44459-026-00045-y) | [开放PDF](https://www.nature.com/articles/s44459-026-00045-y.pdf) |
| SpaceExit | USENIX正式开放论文 | [正式页面](https://www.usenix.org/conference/atc25/presentation/liu-jiacheng) | [PDF](https://www.usenix.org/system/files/atc25-liu-jiacheng.pdf) |

本审计没有使用Wikipedia、Reddit、新闻聚合或未经核验的二手博客作为论文论证依据。SpaceX材料只承担其自身系统架构和公开业务状态的事实，不承担独立安全结论。

## 9.高风险或禁止性表述

| 高风险表述 | 问题 | 建议替换 |
|---|---|---|
| “CVS-RFFI secures Starlink.” | 没有Starlink数据、接口或威胁实验 | “Starlink motivates the receiver-side lifecycle studied here.” |
| “Starlink uses RFFI.” | 无公开证据 | 删除 |
| “Starlink authentication is vulnerable.” | 无公开安全审计支持，可能构成无依据指控 | “RFFI can provide defense-in-depth identity evidence in LEO access systems.” |
| “RFFI replaces cryptography.” | 与SatIQ论证相反，也不符合安全工程常识 | “RFFI complements cryptographic authentication.” |
| “Our LEO channel models Starlink.” | 当前只是通用残余基带算子 | “The simulator represents generic post-synchronization LEO residual conditions.” |
| “The model is deployable onboard.” | 只报告状态大小，没有目标硬件实测 | “The state is designed for onboard deployment; flight-processor validation remains future work.” |
| “Real satellite evidence validates our method.” | 真实卫星证据来自PAST-AI/SatIQ的不同链路和方法 | “Real downlink studies motivate the problem but do not validate our uplink proxy.” |
| “Receiver handover proves cross-receiver generalization.” | 网络切换不等于算法已跨射频前端泛化 | “Handover motivates, but does not verify, cross-receiver robustness.” |
| “RFFI prevents spoofing and replay.” | SatIQ只覆盖特定攻击者、硬件和阈值 | “RFFI can raise the cost of selected spoofing/replay attacks.” |
| “Starlink’s 7,500 satellites are in orbit.” | FCC数字是授权上限，不是实时在轨数 | “The FCC authorized a tranche of up to 7,500 Gen2 spacecraft in 2022.” |

## 10.后续实验由该论证直接引出的要求

若Introduction加入Starlink和星上RFFI必要性，审稿人会顺势要求以下证据。主稿不能只增加应用背景而不回应这些验证责任：

1. **链路方向匹配实验**：至少加入一个“地面终端上行—星载或硬件在环接收机”数据源，避免全部真实卫星引用都来自下行。
2. **真实多接收机前端**：同一批终端由两个及以上独立接收前端采集，验证跨接收机而不仅是信道增强。
3. **过境连续性**：按一条LEO过境轨迹连续改变Doppler、SNR、相位噪声和仰角，而不是每条IQ独立随机抽样。
4. **切换实验**：模拟或实测同一发射机从卫星/波束/接收链A切换到B，报告切换前后身份稳定性和校准漂移。
5. **威胁模型**：区分凭据克隆、基带重放、低成本SDR重发、高采样率波形伪造、合法设备硬件老化和纯干扰；为每类攻击说明RFFI能检测什么、不能检测什么。
6. **跨层融合基线**：比较仅密码/协议身份、仅RFFI、RFFI+逻辑凭据、RFFI+位置/信道证据，而不是把RFFI单独等同于完整认证系统。
7. **星上处理器实测**：在候选ARM/FPGA/边缘加速器上报告完整编码器和Phase 2分类头的延迟、吞吐、峰值RAM、模型存储、能耗和INT8数值闭合。
8. **原始IQ搬运收益**：量化在星上输出标签/分数而不是下传IQ时的数据量、带宽和时延节省，否则“减少下传”只能是动机。
9. **真实波形外推**：当前WiFi代理应与LTE/NTN或卫星IoT波形做至少一项外部验证，避免Starlink Direct to Cell背景与WiSig波形之间存在未解释跨度。
10. **时间稳定性与重注册**：跨天、跨温度或跨月份验证指纹漂移，给出何时需要support更新、何时必须拒绝更新的策略。
11. **开放集边界**：若保留“非法终端/干扰源”应用，必须补unknown rejection或异常检测；当前闭集旧/新类注册不能支撑非法发射机检测。
12. **对称性与规模**：新类数量、旧类数量和K值应覆盖星座终端增长；报告每类floor、旧类遗忘、新类识别和全类校准，而不是只报均值。

## 11.最小可执行修改建议

如果主稿本轮只允许有限改动，建议采用以下最小集合：

1. 加入`spacex2025directtocell`、`kassing2020hypatia`和`wang2023satellitecomputing`三条参考文献；
2. 在Introduction第一段后插入第5.2节的压缩段落；
3. 将Related Work小节改名，并加入“PAST-AI/SatIQ为下行、本文为上行”的方向性边界；
4. 在Limitations中新增一句：未使用Starlink信号、波形、终端、卫星前端或飞行处理器；
5. 在实验待办表中加入上行硬件在环、跨接收机切换、真实处理器资源与跨层认证四项。

这五项能够显著增强应用意义，同时避免把一个代理实验包装成Starlink验证。
