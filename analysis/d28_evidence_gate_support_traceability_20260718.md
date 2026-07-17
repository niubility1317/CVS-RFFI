# D28逐样本证据门追溯表

日期：2026-07-18

状态：核心、runner与本地73项相邻回归完成；待N607 90行support-only执行。

|ID|需求|预定实现|状态|验证|
|---|---|---|---|---|
|D28-01|单IQ高维拼接|唯一LEO_weak IQ的`z160+FFT96+RF32`一条288D行|pending|capsule/operator审计|
|D28-02|轻型适应与注册|复用D27-B 15+10步，不扩展backbone/原型|verified-local|73项相邻回归|
|D28-03|逐样本新旧分辨|单行score的E5证据+闭式ridge gate|verified-local|row-independence测试|
|D28-04|旧类与floor保护|OOF逐类/总体门，失败回退D27-B|verified-local|真实identity argmax OOF安全测试|
|D28-05|K1|无合法cross-fit时禁用gate|verified-local|K1精确透传单测|
|D28-06|无Oracle/无quota|逐样本全注册类一次argmax，无query batch统计|verified-local|API、CLI与resource审计|
|D28-07|资源上限|约2,022活动标量、25step、状态≤256KB、无dense query图|verified-local|资源与runner fold测试|
|D28-08|完整证据|训练日志、逐类/场景、资源、receipt、Git提交|pending|N607完成后核验|
