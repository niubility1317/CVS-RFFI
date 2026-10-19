# 聚焦验证

本候选本地aggregate PASS，共68个负例：4配置、41模型hook/角色/来源/严格键、6rawadapter、15原有蒸馏、2优化器。验证保留head、双教师shadow、NoAs学生128且As_n0、As行学生256，raw/As onlinecorrect/n取已有logits，无新增身份前向。

另有source3条件36/36/−54正确数边界和10个source/panel元数据负例PASS，均0真实CP/IQ/NN/remote。代码编译与空白检查通过。

完整实际F scratch合成输入验证：原206→214，zeroB后的joint160/logits逐值相同，原206不变，B1/B2梯度非零，临时hooks全清。此为合成输入，无checkpoint和真实IQ，不当成正式性能证据。

本地验证使用已确认Wf/W2/W namespace与相应native参数入口；首次W-first旧native参数路由不兼容已归为本机调用层问题，未修改candidate或训练参数。成功调用与模块来源保存在private local_checks.py/local_checks.json。N607沿用pinned r02不可变bootstrap。

一次独立P0/P1审查已PASS（raw_iq_review，19个新包Python文件及预登记连接）；scope只限下一真实运行直接缺陷，不加P2/报告/签名门槛。源场景metadata标签已改为实际low_suburban/mid_urban/low_urban，数值轴与门槛不变。真实合法父checkpoint无query smoke在launcher首步，PASS立即继续。
