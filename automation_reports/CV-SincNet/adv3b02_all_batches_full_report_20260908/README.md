# ADV3B02完整报告与数据

先打开[离线完整报告](report.html)，或阅读[Markdown正文](report.md)。HTML包含39个内容块、18幅图和6张可排序表，桌面1440px与移动390px标准探针均通过，来源按钮检查通过。报告不依赖外部网络。

覆盖A1—A7、P1—P5/E1/R1及PAIR八配置×三seed：38行、7318个epoch，36行E200最终结果、2行失败；未启动新实验或补测。训练全字段1191列、标准loss85项。失败空值保持N/A。早期与PAIR的LEO场景分配和模拟seed不同；SAFE版本也不同，解释边界见正文。

## 快速入口

- [全部测试结果](all_test_results.csv)：38行，含2行失败空值。
- [PAIR三seed统计](combined_pair_method_summary.csv)：均值、样本SD、同seed差值。
- [完整训练数值](training_all_fields.csv.gz)：7318行×1191字段，CSV已压缩。
- [逐epoch损失](training_losses.csv)：85个标准原始/加权损失字段。
- [配置与实际命令](configurations.json)、[机制激活统计](pair_activation_evidence.csv)。
- [逐接收机/日期结果](pair_receiver_day_breakdown.csv)、[逐类别结果](pair_per_class.csv)、[完整混淆矩阵与计数](pair_test_all_counts_confusions.json)。
- [早期原始训练和测试日志](raw_early_training_test_logs.zip)：278个文件。
- [PAIR原始训练日志及实际发布核心源码](raw_pair_training_logs.tar.gz)：258个文件。
- [早期V2发布源码](early_release_sources.tar.gz)：22个文件；[A1历史参考源码](early_historical_git_reference.zip)有明确的原始release身份限制。

## 验证与复现

[validation.json](validation.json)记录数据/归档检查；[html_delivery_validation.json](html_delivery_validation.json)与[reader_probe_results.json](reader_probe_results.json)记录显示验证。[file_manifest.csv](file_manifest.csv)包含下列文件的字节数和SHA256。源码与根目录输出镜像到当前Git分支，最终完整提交OID及远端读回在交付消息中记录。

Windows此安装环境的dump-DOM虚拟时间先于阅读器加载完成，因此交付脚本保留标准打包器、标准SVG提取器和标准验证探针，只用已安装Playwright提供真实时间浏览器运行；没有替换报告阅读器或隐藏失败。SQL数据库chart_data.sqlite是图表数据的物化副本，chart_queries.json记录实际运行的读取SQL及上游文件。

不含数据集、checkpoint张量、received-IQ或预测NPZ大数组；未记录的早期逐TX/逐日期数据不补造。所有现有日志、本文用到的训练字段和评分细分已包含。复现报告数据使用tools/build_adv3b02_combined_report.py与tools/render_adv3b02_combined_report.py；生成并验证HTML使用tools/deliver_adv3b02_report.mjs。完整重训需要原环境、数据和checkpoint。

## 文件清单

|文件|MiB|
|---|---:|
|[.gitattributes](.gitattributes)|0.000|
|[all_numeric_statistics.csv](all_numeric_statistics.csv)|3.706|
|[all_test_results.csv](all_test_results.csv)|0.007|
|[analysis_summary.json](analysis_summary.json)|0.000|
|[artifact.json](artifact.json)|2.424|
|[chart_data.sqlite](chart_data.sqlite)|0.562|
|[chart_queries.json](chart_queries.json)|0.009|
|[combined_pair_method_summary.csv](combined_pair_method_summary.csv)|0.001|
|[configurations.json](configurations.json)|0.524|
|[early_historical_git_reference.zip](early_historical_git_reference.zip)|0.143|
|[early_receiver_scenario.csv](early_receiver_scenario.csv)|0.018|
|[early_release_source_inventory.json](early_release_source_inventory.json)|0.005|
|[early_release_sources.tar.gz](early_release_sources.tar.gz)|0.276|
|[html_delivery_validation.json](html_delivery_validation.json)|0.001|
|[log_scan.json](log_scan.json)|0.059|
|[loss_statistics.csv](loss_statistics.csv)|0.345|
|[pair_activation_evidence.csv](pair_activation_evidence.csv)|0.016|
|[pair_current_training_status.json](pair_current_training_status.json)|2.792|
|[pair_execution_evidence.json](pair_execution_evidence.json)|0.012|
|[pair_method_summary.csv](pair_method_summary.csv)|0.002|
|[pair_paired_deltas.csv](pair_paired_deltas.csv)|0.003|
|[pair_per_class.csv](pair_per_class.csv)|0.036|
|[pair_receiver_day_breakdown.csv](pair_receiver_day_breakdown.csv)|0.237|
|[pair_summary.csv](pair_summary.csv)|0.005|
|[pair_test_all_counts_confusions.json](pair_test_all_counts_confusions.json)|8.458|
|[pair_validation.json](pair_validation.json)|0.000|
|[prior_a1_report.md](prior_a1_report.md)|0.011|
|[prior_pair23_report.md](prior_pair23_report.md)|0.014|
|[prior_pair_recovery_report.md](prior_pair_recovery_report.md)|0.005|
|[prior_two_batch_report.md](prior_two_batch_report.md)|0.029|
|[raw_early_inventory.json](raw_early_inventory.json)|0.067|
|[raw_early_training_test_logs.zip](raw_early_training_test_logs.zip)|25.645|
|[raw_pair_inventory.json](raw_pair_inventory.json)|0.047|
|[raw_pair_training_logs.tar.gz](raw_pair_training_logs.tar.gz)|59.306|
|[reader_probe_results.json](reader_probe_results.json)|0.001|
|[report.html](report.html)|1.632|
|[report.md](report.md)|0.033|
|[structured_file_inventory.json](structured_file_inventory.json)|0.035|
|[training_all_fields.csv.gz](training_all_fields.csv.gz)|16.187|
|[training_analysis.json](training_analysis.json)|2.006|
|[training_curves.csv](training_curves.csv)|2.071|
|[training_health.csv](training_health.csv)|0.011|
|[training_losses.csv](training_losses.csv)|6.414|
|[validation.json](validation.json)|0.001|
