from pathlib import Path
p=Path(__file__).parent;s=(p/'check_phase1_estimated_channel_source.py').read_text(encoding='utf8').replace('phase1_source_estimated_channel_restoration_probe.py','phase1_source_estimated_channel_fusion_probe.py').replace('VERIFIED_ESTIMATED_CHANNEL_SOURCE_SMOKE','VERIFIED_ESTIMATED_CHANNEL_FUSION_SOURCE_SMOKE')
exec(compile(s,str(p/'check_phase1_estimated_channel_source.py'),'exec'))
