# 源响应 EMP：在读取新正式成绩前固定的决定

证据为原运行source_decision.json（本地continuation4.json只读副本），不是目标分数。EMP已经固定E20/4440更新，ROT仍在E18；当前未读取本轮任何目标accuracy/F1。

EMP fullV clean=98.214815%，相对匹配B00=-0.018519pp；最差源RX=95.240741%。
固定检查={'clean_gain': False, 'worst_RX_health': True}；资格=FAIL。

固定规则要求source clean至少+0.5pp、最差RX损失≤1pp。决定在自身target prediction之前已由launcher写入；当前独立读回不使用任何target结果。

按预登记结束当前固定EMP方向，不发布其联合或seed扩展；不依据后续目标成绩挽救或换ROT为候选。

仍完成当前两行已注册原七视图，保留全部正负结果；不停止健康ROT或正在预测的EMP。整体clean/urban联合目标保持active。

后续先解释本轮源端失败或成功、完整预算和实际G强度；不能把冻结输入安全或理想方向覆盖当成训练收益。新候选需要新的源证据，不追加对齐损失或擅自扩大E20。
