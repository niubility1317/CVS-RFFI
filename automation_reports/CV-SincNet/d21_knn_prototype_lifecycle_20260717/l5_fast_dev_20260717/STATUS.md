# SPLIT_MISMATCH_DIAGNOSTIC

本目录首轮36-cell结果错误地按mother NPZ原数组每类前20/后20切分，没有使用正式capsule的`somph-offline-split-v1`SHA256排序，因此所有性能数值仅保留为切分错误诊断，禁止用于候选比较、晋级或正式声明。

修正后的正式切分开发结果位于相邻目录`l5_fast_dev_formalsplit_20260717`。本目录脚本后来增加了显式`--split-policy`开关；`command.txt`未提供该开关，仍精确对应本目录的历史错误切分结果。
