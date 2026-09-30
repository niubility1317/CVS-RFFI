import json
import urllib.request
from pathlib import Path
from pypdf import PdfReader

out=Path('local_artifacts/newclass_registration_20260930/papers')
out.mkdir(parents=True,exist_ok=True)
records=[]
for name,url in [('LoRa_RFFI','https://arxiv.org/pdf/2107.02867'),
                 ('ISSL_DOI','https://doi.org/10.1109/TCCN.2025.3583118'),
                 ('LoRa_dataset','https://ieee-dataport.org/open-access/lorarffidataset')]:
    record={'name':name,'url':url}
    try:
        req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'})
        with urllib.request.urlopen(req,timeout=35) as r:
            blob=r.read(); record['resolved_url']=r.url
        if blob.startswith(b'%PDF'):
            path=out/(name+'.pdf')
            if not path.exists(): path.write_bytes(blob)
            reader=PdfReader(path)
            record.update(status='VERIFIED',pages=len(reader.pages),path=str(path))
        else:
            path=out/(name+'_landing.html')
            if not path.exists(): path.write_bytes(blob)
            record.update(status='LANDING_ONLY',path=str(path),bytes=len(blob))
    except Exception as e: record.update(status='FAILED',error=str(e))
    records.append(record)
print(json.dumps(records,ensure_ascii=False,indent=2))
(out/'download_status.json').write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
