"""Actual fresh-source entry checks for both full-precision E600 candidates."""
import argparse
import json
from pathlib import Path
import torch
from run_a1_e600_repair import repair_matrix
from check_a1_ecrs_cross_rx import run_check


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    matrix = repair_matrix()
    results = []
    for row in matrix['rows']:
        one = dict(matrix, rows=[row])
        folder = args.output.with_suffix('')/row['id']
        result = run_check(torch.device(args.device), folder,
            matrix_factory=lambda: one,
            expect_cross_rx=float(row['options']['--a1_ecrs_cross_rx_weight']) > 0)
        assert not list(folder.rglob('first_rc4_anomaly.pt')), 'Full precision entry has a numerical anomaly'
        result['row'] = row['id']
        results.append(result)
    args.output.write_text(json.dumps({'status': 'PASS', 'rows': results}, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'status': 'PASS', 'rows': [r['row'] for r in results]}), flush=True)


if __name__ == '__main__':
    main()
