# 八组冻结后 L_dev 分支机制分析预登记

状态：LOCAL_VERIFIED_NOT_LAUNCHED。属于同一八组row的冻结后源解释阶段，不增加训练候选、epoch或目标查询。parent八组全部训练/七视图独立评分完成后才发布独立CPU release，当前健康release不热改。源码experiments/cvs_urban_branch50/mechanism.py与source_publish.py；既有trace B20/B21保持pending，不能把本地工具检查称为八组结果。

输入是实际inner_source_contract声明的720条L_dev，原外层L6300/U56700/V27000保持。source cache会被原读取器整体载入；L_train/V随即丢弃，不作为forward、教师、错误TX参照或任何更新输入。U不打开，query IQ/truth/目标指标不读取。核对八组真实source freeze、final50/11100、scratch来源、相同内部物理ID与checkpoint split_info；默认头原样，teacher为每组自己checkpoint内EMA。来源身份验证是既有输入权限，不增加data revalidation/签名或hash链。

固定A0 final50在同一L_dev上的clean正确集合，所有组用同一集合统计satellite保持、A0错误修复/新错、原默认头scale30归一化间隔。六种源practical使用独立固定同物理命名流和真实原引擎；正式received IQ完全不读写。actual T/F读fuse、P读pa_norm、R读projection、S读output_norm、J等于原部署joint；不混用旧raw projection观察器。

对六个实际位置逐样本计算student clean对satellite方向、ownEMA clean方向，以及相对于全部720条合法不同TX clean参照的最近错误TX cosine和正确减错误gap；不拟合任何分类器/阈值/决策规则。记录全局branch variance/norm、P/R/S真实gain和gain后贡献相对joint范数，T/F是经过非线性融合的输入，不能冒充线性增量。单看loss/cosine下降或gain降低不能宣布鲁棒性提高。

源mid/low另以eval且零更新的autograd.grad计算实际全特征195041参数（W排除，unused按零）的 .5CE 与当前固定aux梯度比例/每层夹角。T/S同时记录全局夹角与仅两者均有非零梯度的共享参数范围/夹角；后者排除独立t_proj/local_projection等，避免把完整向量夹角解释为共享层冲突。W和融合gain的aux梯度必须零，所有.grad保持None，model/teacher参数buffer逐值不变。该检查是冻结后eval源梯度，不冒充真实Dropout、clip或AdamW的即时增量；实际训练clipping/总更新另取完整native日志。

样本按RX/LOS-PARTIAL-BLOCKED/locked-degraded-unlocked/新增噪声SNR分层，另存receiver质量代理、残余CFO与核长。新增噪声SNR不是总SNR。固定完整信道比较提供方向和错误证据，不是单项物理损伤的唯一因果解释；原损伤控制诊断另有其明确total/fixed状态口径。

预算46800前向=720固定A0 clean集合+8×(720student clean+720ownEMA clean+6×720satellite)，optimizer/head-fit/EMA更新均0。第一固定A0实际clean/default-head前向打印SMOKE PASS/query=false并继续，无额外smoke token。CPU-only、2线程，独立release cvs_sigtime_branch_align_source_20261015_r01，CWD PROJECT/releases/该release；owner codex/root/branch-align-source-20261015。output是当前parent run/source_branch_mechanism，拒绝覆盖。精确命令和预期artifact见experiment.json.source_analyses。

本地4项PASS：真实T/S共享梯度到t2且排除独立投影、aux W/gain零和模型state/grad保持；gain关闭可使贡献归零而S本身不变；只选合法互斥开发物理ID并拒绝混入训练/缺失ID；固定A0集合上的repair/new-error口径。compileall和source_publish --help导入检查。一次独立限定本分析的P0/P1审查PASS，指定范围未发现P0/P1；之后保留一份Git代码commit、一次archive传输SHA/远端编译、普通账户预检和PID/CWD/argv/日志/最终source结果独立回读。父模型训练成本仍记各row E50，本阶段增量训练0。正式三seed及稀疏组合仍按原源规则，不以此单seed诊断选择最佳分支或目标得分。

P/R/S贡献比是gain后向量norm相对joint的诊断比，可能因向量抵消超过1，不是和为1的因果重要性比例。所有source结果仍待真实冻结八组；当前仅工具验证和预登记，不称已完成机制归因。
