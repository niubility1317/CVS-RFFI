"""Hierarchical discovery policy, sharing the frozen registration truth pipeline."""
from __future__ import annotations
import argparse
from . import star_hier_register_run as evaluation
from . import star_signal_register_score as base

RUN_ID=evaluation.RUN_ID
SCHEMA='pamf_phase3_hier_register_scores_v1'
STATUS='HIER_REGISTER_SCORES_COMPLETE'
METHODS=base.METHODS
PRIMARY=base.PRIMARY


def score(prediction_root,U_truth,source_novel_truth,clean_truth,star_truth,output):
    return base.score(prediction_root,U_truth,source_novel_truth,clean_truth,star_truth,output,
        evaluation_override=evaluation,result_schema=SCHEMA,result_status=STATUS)


def main():
    p=argparse.ArgumentParser();p.add_argument('--prediction-root',required=True);p.add_argument('--U-truth',required=True)
    p.add_argument('--source-novel-truth',required=True);p.add_argument('--clean-truth',required=True);p.add_argument('--star-truth',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();score(a.prediction_root,a.U_truth,a.source_novel_truth,a.clean_truth,a.star_truth,a.output)


if __name__=='__main__':main()
