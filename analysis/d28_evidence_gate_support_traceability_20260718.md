# D28逐样本证据门追溯表

日期：2026-07-18

状态：资源审计修正后的v2、runner、73项回归及N607 90行support-only执行完成；D28安全透传但无held增益，不晋级。

|ID|需求|预定实现|状态|验证|
|---|---|---|---|---|
|D28-01|单IQ高维拼接|唯一LEO_weak IQ的`z160+FFT96+RF32`一条288D行|verified|capsule/operator与110行/场景审计|
|D28-02|轻型适应与注册|复用D27-B 15+10步，不扩展backbone/原型|verified-local|73项相邻回归|
|D28-03|逐样本新旧分辨|单行score的E5证据+闭式ridge gate|verified-local|row-independence测试|
|D28-04|旧类与floor保护|OOF逐类/总体门，失败回退D27-B|verified-local|真实identity argmax OOF安全测试|
|D28-05|K1|无合法cross-fit时禁用gate|verified-local|K1精确透传单测|
|D28-06|无Oracle/无quota|逐样本全注册类一次argmax，无query batch统计|verified-local|API、CLI与resource审计|
|D28-07|资源上限|约2,022活动标量、25step、状态≤256KB、无dense query图|verified-local|资源与runner fold测试|
|D28-08|完整证据|训练日志、逐类/场景、资源、receipt、Git提交|verified|v2 90行、6 artifact与独立审计|

## 执行结论

- D28-B/C 15/15fold及full-K10 3/3场景均安全禁用，逐行精确透传D27-B；不晋级。
- gate OOF角色BA约81%，但平均以约12pp old换约7pp new，说明类无关共同平移过粗。
- low-elev `09f8=0%`、rain `f608=0%`，old最低场景floor40%；D29必须同时做类条件旧域保护和弱新类释放。
- v2资源审计将外部OOF trace与predictor state分离，并计入完整`D27 score+gate`延迟。
