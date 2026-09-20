"""Append epoch telemetry; rewrite CSV only when its schema expands."""
import csv
import json
from pathlib import Path


class IncrementalTelemetry:
    def __init__(self, csv_path, jsonl_path):
        self.csv_path=Path(csv_path) if csv_path else None
        self.jsonl_path=Path(jsonl_path) if jsonl_path else None
        self.fields=[]
        self.seen=set()
        self.count=0

    def write(self, rows):
        if len(rows)!=self.count+1:
            raise ValueError('Telemetry requires exactly one new epoch row')
        row=rows[-1]
        additions=[key for key in row if key not in self.seen]
        self.fields.extend(additions);self.seen.update(additions)
        # Serialize before touching files, preserving the original NaN rejection.
        encoded=json.dumps(dict(row),ensure_ascii=True,sort_keys=True,allow_nan=False)+'\n'
        if self.csv_path:
            self.csv_path.parent.mkdir(parents=True,exist_ok=True)
            rewrite=bool(additions) or self.count==0
            with self.csv_path.open('w' if rewrite else 'a',encoding='utf-8',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=self.fields,extrasaction='ignore')
                if rewrite:
                    writer.writeheader();writer.writerows(dict(item) for item in rows)
                else:
                    writer.writerow(dict(row))
        if self.jsonl_path:
            self.jsonl_path.parent.mkdir(parents=True,exist_ok=True)
            with self.jsonl_path.open('w' if self.count==0 else 'a',encoding='utf-8') as f:
                f.write(encoded)
        self.count+=1
