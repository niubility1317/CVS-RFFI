# D92本地数值测试环境恢复记录

日期：2026-09-29。范围仅限本地合成数据的CPU数值测试。

## 环境来源与限制

原`C:/Users/lh594/.conda/envs/ssr-gpu`链接指向不可用的`D:/conda-envs/ssr-gpu`。本次未修改该链接、磁盘挂载、系统PATH或共享环境，也未使用管理员权限。

新前缀为`E:/type10-7/local_envs/ssr-gpu`。其`conda-meta/history`记录了2026-09-29 21:07:43执行的离线克隆：来源`C:/Users/lh594/.conda/envs/SSL-RFFI`，Conda 25.9.1，共19个Conda包，Python 3.10.0。此操作建立独立的最小CPU数值环境，不代表原GPU环境已恢复。

依赖固定为`numpy==2.2.6`、`scipy==1.15.3`、`pytest==8.4.2`及`threadpoolctl==3.6.0`。没有安装Torch或CUDA。仅通过原默认PyPI源下载二进制包，保留TLS证书校验。

## 中断核对与补齐

接续前通过Windows原生`tasklist`确认没有`python.exe`、`conda.exe`或`pip.exe`安装进程。包元数据显示NumPy和threadpoolctl已经安装，SciPy和pytest缺失，因此没有重建前缀或盲目重装。

第一次补齐使用`--retries 0 --timeout 30 --only-binary=:all:`，总时限180秒，在41.3 MB的SciPy wheel下载阶段超时。其安装子进程退出后，独立Conda激活读回确认SciPy和pytest仍缺失；再次通过`tasklist`确认没有安装进程。随后沿用固定版本、默认来源和TLS校验，将下载总时限调整为1200秒，保留已有缓存。执行脚本及日志保留于：

- `E:/type10-7/local_artifacts/finish_minimal_numeric_env_20260929.py`
- `E:/type10-7/local_artifacts/minimal_numeric_env_reconciled_install_20260929.log`
- `E:/type10-7/local_artifacts/readback_minimal_numeric_env_20260929.py`
- `E:/type10-7/local_artifacts/minimal_numeric_env_readback_20260929.json`
- `E:/type10-7/local_artifacts/finish_minimal_numeric_env_20260929_long.py`
- `E:/type10-7/local_artifacts/minimal_numeric_env_reconciled_install_20260929_long.log`

## 验证

状态：**VERIFIED**。延长时限的补齐完成；SciPy wheel下载耗时约4分钟，最终日志为`RETURN_CODE=0`。随后独立执行Conda激活与数值检查，实际解释器、`sys.prefix`及`CONDA_PREFIX`均指向新前缀，四个固定版本完全匹配，用户site已禁用，Torch未安装。

NumPy Cholesky重构残差为`1.5463617361248926e-15`，SciPy Cholesky求解残差为`2.9340723804077858e-15`，均低于`1e-12`。验证期间两个OpenBLAS后端均限制为2线程。完整读回证据为`E:/type10-7/local_artifacts/minimal_numeric_env_verified_20260929.json`。

数值检查脚本`E:/type10-7/local_artifacts/verify_minimal_numeric_env_20260929.py`核对实际解释器、`sys.prefix`、`CONDA_PREFIX`、四个固定依赖版本、禁用用户site以及Torch缺席，并对固定随机种子的合成矩阵执行NumPy Cholesky和SciPy Cholesky求解。不读取真实IQ、特征、checkpoint、support或query，不启动实验。

运行时使用绝对前缀，避免损坏的同名环境链接：

```text
F:/App/miniconda3/Scripts/conda.exe run -p E:/type10-7/local_envs/ssr-gpu python -s E:/type10-7/local_artifacts/verify_minimal_numeric_env_20260929.py E:/type10-7/local_artifacts/minimal_numeric_env_verified_20260929.json
```

该环境只覆盖本次NumPy/SciPy测试需求；GPU训练、Torch相关功能和旧环境的完整依赖兼容性均未验证。
