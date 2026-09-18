# ADV3B02＋DAOT＋FastTrust-RC4原LEO拼接增强

用户于2026-09-18授权启动单个实验，并明确更正为DAOT+FastTrust-RC4。预算按项目默认200epoch，seed392005，从零训练，禁止外部checkpoint、anchor或teacher继承；EMA由本轮学生产生。逐行配置与路径见experiment.json。

训练采用原LEO的clear_leo、low_elev_leo、rain_leo，信道模型为legacy_full，不是leo_residual/LEO_WEAK。有标签clean和卫星IQ沿batch维拼接为2B，只调用一次主模型，输出切回clean和satellite：clean承担原有训练目标，satellite只增加CE，lambda_sat_cls=0.68、lambda_sat_cons=0。拼接批统计从E1生效；保留原课程E1–40 p=0.30晴空、E41–90 p=0.60低仰角/降雨、E91–200 p=0.80三场景，保留卫星辅助CE从E80起点。

DAOT保留A1两视图mean版本，clean+clear_leo教师与clear_leo学生，E21起逐步启用；有标签与无标签两条路径均使用原LEO。A1不调用hard teacher，因此低仰角/降雨的训练覆盖来自有标签拼接课程。RC4保持原生H/P及质量预算0.15；独立RC4 hard-satellite路径保留关闭，不能声称所有U都有卫星伪标签CE。无ECRS、R3、响应博弈或STN附加项。

数据沿用已核实source契约：RX1/3/4/6/8、day1/2/3、TX0–5、equalized=1、2×256、L/U/V=6300/56700/27000。启动时按物理ID逐角色核对已有契约；V只读，训练不创建目标loader。六类seed及派生增强seed见登记。

固定在E100、110、120、130、140、150、160、170、180、190、200保存并测试，共11次。每次针对相同168000条opaque目标IQ分别输出clean、clear_leo、low_elev_leo、rain_leo的672000条预测，全部固定后由独立CPU scorer连接truth；训练仅读取覆盖数量，不读取准确率。重建独立eval模型并恢复Python/NumPy/Torch随机状态。最终E200另存clean及三个LEO_WEAK参考结果，保持独立目录。所有中间测试为用户授权探索性观察，不用于选模、调参或选择性重跑；最终权重固定E200。

## 本地验证

- 真实本轮随机初始化checkpoint保存与精确重建通过；三种原LEO均使用legacy_full、IQ改变且有限；主模型输入确认为[8,2,256]（clean B=4＋sat B=4），单次前向并反传通过。
- DAOT E21真实L/U目标执行并反传通过，仅调用clear_leo；详见evidence/local_smoke.json。该检查不读取query或历史权重。
- 5项聚焦测试通过：拼接及梯度隔离、原LEO日程与11个测试点、原/弱场景完整性负测、真实模型E100预测＋独立CPU评分及随机状态恢复。小型合成评分只证明链路，不是性能结果。
- 一次独立P0/P1审查及用户更正后的定点复核完成，未留阻断项。

## 当前状态

LOCAL_VERIFIED，待提交发布及N607启动独立读回。不得将此状态解释为训练已启动或已有性能结果。唯一launch owner为本任务/root；最多占用一个空闲GPU，不干预已有进程。启动失败或结果不明先核实原run，禁止盲目重复提交。
