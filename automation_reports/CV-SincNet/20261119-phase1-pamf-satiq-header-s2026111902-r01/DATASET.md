# SatIQ牛津和图恩完整包头

N607路径为 `/home/szu2070436088/2510044040/CV-SincNet/data/20261119-satiq-oxford-thun-full-v1`。两站各有6个NumPy波形分片，每个含5,000条记录。形状为 `[5000, 11000, 2]`，最后一维是I、Q，类型为float32，采样率为25 MS/s。

|路径|内容|
|---|---|
|`source/index.json`、`source/ids.npy`|牛津分片与行序、30,000条不透明ID|
|`source/metadata.npz`|牛津原始 `satellite_id`、`beam_id`、`timestamp_ns`、`confidence`、`original_split`、`shard`、`row`|
|`target/query/index.json`、`target/query/ids.npy`|图恩分片与行序、30,000条不透明ID|
|`target/scorer/truth.npz`|按ID连接的图恩真值，仅供冻结预测之后的独立评分|

索引中的分片顺序与各站的 `ids.npy` 一致。英国元数据按同一顺序存放。原 `split` 字段记录采集包原角色，本次实验的牛津划分由原配对实验的 `20261119-pamf-satiq-oxford-thun-v1/source` 决定，图恩全站为测试。

```python
from pathlib import Path
import json
import numpy as np

root = Path("/home/szu2070436088/2510044040/CV-SincNet/data/20261119-satiq-oxford-thun-full-v1")
uk = root / "source"
index = json.loads((uk / "index.json").read_text(encoding="utf-8"))
iq = np.load(uk / index["shards"][0]["path"], mmap_mode="r", allow_pickle=False)
ids = np.load(uk / "ids.npy", mmap_mode="r", allow_pickle=False)
with np.load(uk / "metadata.npz", allow_pickle=False) as metadata:
    first_tuple = (metadata["satellite_id"][0], metadata["beam_id"][0])
first_packet = iq[0]
```

图恩IQ用 `root / "target/query"` 读取。N607现有PyTorch环境可以通过 `torch.utils.dlpack.from_dlpack(np.array(iq[:32], copy=True))` 将只读内存映射中的批次传给PyTorch。
