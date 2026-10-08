"""Create a separate immutable repair package; never alter running R1 files."""
from pathlib import Path


def main():
    source=Path(__file__).resolve().parent
    output=source.parent/'cvs_receiver_residual_v2'
    output.mkdir(exist_ok=False)
    for path in source.glob('*.py'):
        if path.name=='prepare_r2_package.py': continue
        text=path.read_text(encoding='utf-8').replace('cvs_receiver_residual','cvs_receiver_residual_v2')
        text=text.replace('20261008-phase1-receiver-residual-manysig-m16-r01',
            '20261008-phase1-receiver-residual-manysig-m16-r02')
        text=text.replace('cvs_receiver_residual_v2_20261008_r01','cvs_receiver_residual_v2_20261008_r02')
        text=text.replace('codex/root/receiver-residual-20261008','codex/root/receiver-residual-20261008-r02')
        (output/path.name).write_text(text,encoding='utf-8',newline='\n')
    (output/'DESIGN.md').write_text(source.joinpath('DESIGN.md').read_text(encoding='utf-8')+
        '\n## R2初始化修复\n\nR1的CPU头初始化错误重置CUDA dropout RNG，保留为含随机性混杂的诊断，健康任务不停止、不热改。R2从零重新执行固定16行，不继承R1权重；仅修复CPU私有生成器，不改变结构、损失或预算。设计在R1任何target预测前固定。R2使用剩余合法GPU名额，最多4个worker、每GPU总任务最多2个；已验证CPU与CUDA随机数状态均保留。\n',encoding='utf-8',newline='\n')
    print(output)


if __name__=='__main__': main()
