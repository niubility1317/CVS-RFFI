# Phase3独立时频恢复器：实际F0与源G10零更新成本

- run_id：`20261101-diagnostic-P3Restore-TF-source-cost-s2026092701-r01`
- group_id：`phase3-independent-restoration-source-cost`；类别：`diagnostic`；阶段：`Phase3`
- 配置与矩阵：[experiment.json](experiment.json)；状态记录：[events.jsonl](events.jsonl)
- 当前登记状态：FAILED（实际状态按events.jsonl及独立证据更新）

## 目的与对照

原L_s/U_s512条G10配对，独立TF和R69图的纯IQ/实际F0指导步骤交替计时，不更新任何权重。

## 数据、seed与模型来源

实际数据契约、权限例外、完整seed角色、checkpoint来源和选择规则见experiment.json。
逐行配置通过config_ref/resolved_config_ref定位；待补项必须在对应生命周期补齐。

## 执行与存储

命令、环境、CWD、commit、launch owner、输出和日志路径见experiment.json。
实际PID/GPU、读取时间、remote readback、失败或替代关系在此追加，并用record命令记录证据指针。

## 结果与覆盖

尚无结果。按预登记artifact逐项记录路径和缺项；保留每row与RX/day/TX/scene/K/seed的对应关系。
源域训练完成、预测完成、评分完成及协议有效性分别陈述。不得用总索引或旧状态证明当前运行。

## 交接

记录已完成、当前run/commit、证据路径、阻塞与下一步；恢复先查原run，不重复启动。


## 固定源矩阵与实际Phase3绑定

PreviousgoalturnPROGRESS:oldfullratepairall30-sevenclosedff853negative/defaultDURAkept. User20261101 explicitlynewGisforPhase3/majorarchitectureallowed/LsUs-only/OEfrozendiagnostic. NewindependentTFc961 sourcegoalimplemented9checks,nottrained/notqualified. ThisdiagnosticusesactualF0,notF50;no newGweights or target feedback.

SourceG10originalprimitiveL6300d319andvalidatedUs56700exactIDs/P0only;no metadata_y/teacher_logits/restored/oldnormalizedIQ. Trustedbuilderus._locate enumeratesrawsourcecoordinates/sampleIDswithouttruth;originalrawManySigwholefileincludesotherrolesandTXnamesbutonlyselected512sourcepacketvaluesindexed. BuilderlocatorsnotpublicGinputs. NewrawRMS512,canonicalobserve2firstLsforaudit,postchannelRMS0.

FirstnewNNworkreal2Ls smoke:strictF0+fixedR69load/newTFidentity/outheadgradient/fullF0existingautograd-inputVJPcap.5/sourcephysicalanddefaulthead/nativeR69-productionclose/stateunchanged;callbackPASScontinueswhole512/benchmark. OriginalF0param-buffers/defaultheadfixed;DURAcopyautogradonly/referenceNOTnewGancestor. F0source50/11100knowledgecostexplicit;0target-PCD/r03teacher/0TXobjective.

PurephysicalIQNMSE+.05mean3STFTrelativeL1(32/64/128 complexfullSTFT);guided+.1cosfullF0fused160,detachedreference/perrecordIQgradientcap.5/globalclip1. Threeblocks2warm6timed/arch-objective/alternatingarchorder/B256resident/interleaved51225Ls75Us. 96forwardbackwardclip/0optimizerstep. Evalnormalizationstatepreserved;differentTFidentityinit-vsR69trainedweights/noquality-equivalentclaim.

Timingincludesobjective/optionalF0inputVJP/backward/clip1;excludesoptimizer.step/IO-transfers-cache-load-eval,thosepreparationtimesseparate. AllmodelsresidentpeaknotTFsinglemodelVRAM. CUDAeventsandwallreported/pairedsameGPUsharedload. Costprimary>=10%medianwallreductionand3blockspositiveBOTHobjectives;scientificFAILdoesnotcompletegoal orcancelalreadyregisteredtrainedversions.

23focusedchecksacrossmodel9/objective3/source9/cost2:actualtime-frequencygategradient/HermitianDFTforward-paramVJP/recordindependence/state-MAC/roleexpansion/shape/Fcap-and-targetdetach;FakeLstorethrowsifyfetchedandbothmetadatarolesPASS;GPUcapandactualCLI. SourceprivacytestinitialfixtureusedWindowswrongsplitseparator/fixedpath.name/targetedPASS/no productionchange. UniqueindependentP0P1PASS/noP0P1.

Strictmetadata-onlyF0readobservedoriginal3ff8main3view/final50-11100/232993 andoriginal392005roles;sourceG10UsCOMPLETE-VALIDATED_ONCE/noUslabels-or-secondRMS/Ld319producer. RuntimeprivateinputsnotGit;code/configidentityonecommit/onearchivewithinputs/oneSHA/remotecompile/preflight/postPID-CWD-argv-CUDA-GPU-loggrowth;noextra hashes/reviews/datarevalidation.

Privateall96timingrows/sourcepairretained;Gitonlyaggregates/report. IndependentNumPymedian/block/counter复算0NN/0truth. Scientificsourcecostnotactuallearning/Phase3AK-RU/registersuccess. Nextfromscratchsingle30fixedmatrixafterrealcost/F0binding;allactualnewGversionsmustformaltest. Fullsixgap3pp-clean-Phase3open-efficiencygoalACTIVE/defaultDURAandothersunchanged.

速度阈值只检验零更新resident步骤成本，不能替代30轮训练质量、六景逐个3pp、clean保持或实际Phase3开放指标。固定96步完整执行，技术失败保留产物；无训练后选择或目标访问。


## 实际技术失败与保留

VERIFIED / FAILED。执行7b49cf9c74162e27dbf9e6e271cf5d621928a417，controller2119396/worker2119420、GPU4准入前1个compute和23189MiB。实际进程CWD/argv/CUDA已观察，现均退出；归档SHA和远端编译通过。真实2Ls smoke在DURA生产接口返回值适配处失败：把InferenceEngine(reader=None)的(restored_IQ,)误解包为两项。引擎内部已验证并移除valid，正确公共调用为engine(x)[0]。未完成smoke、未进入96步成本循环，无优化、target或新性能结果；不宣称完整状态检查通过。未捕获live nvidia PID样本，不伪造GPU峰值或为遥测重跑。旧root/权重/数据/日志全保留。

仅修正返回值并改用独立r02 root，其他模型、512源ID、G10配对、损失、96步矩阵、权限和资源不改。唯一直接问题的针对性P0/P1复审PASS；不重审无改动部分、不热改r01。
