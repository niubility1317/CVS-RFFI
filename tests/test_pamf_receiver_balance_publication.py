import json,tempfile,unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
from experiments.pamf_channel_pl import publish_receiver_balance_confirmation as p
from tests.test_pamf_receiver_balance_evaluation import completed_source


class ReceiverBalancePublicationTests(unittest.TestCase):
    def test_source_prerelease_gate_matches_predictor_choice(self):
        e=p.c.evaluation
        for raw,h in [(26568,62109),(26569,62109),(26568,62110)]:
            scores={m:dict(raw_correct=26568,H3_correct=62109) for m in e.source.METHODS};scores[e.source.METHODS[1]]=dict(raw_correct=raw,H3_correct=h);summary=completed_source(scores)
            with tempfile.TemporaryDirectory() as td:
                file=Path(td)/'complete.json';file.write_text(json.dumps(summary));package=dict(source_check=dict(path=str(file),run_id=e.source.RUN,methods=list(e.source.METHODS)))
                winner=p.prerelease_source_check(package)
            with patch.object(e,'read',return_value=summary):self.assertEqual(winner,e.selected_method(dict(selection_source='/project/runs/'+e.source.RUN+'/complete.json',purpose='source_selection')))

    def test_no_source_completion_no_pass_or_wrong_winner_denied(self):
        e=p.c.evaluation;good=completed_source();bad=completed_source({m:dict(raw_correct=26567,H3_correct=62109) for m in e.source.METHODS})
        for s in [dict(good,status='RUNNING'),dict(good,source_scores={}),dict(good,target_reads=1),dict(good,source_selected=e.source.METHODS[1]),bad]:
            with tempfile.TemporaryDirectory() as td:
                file=Path(td)/'complete.json';file.write_text(json.dumps(s));package=dict(source_check=dict(path=str(file),run_id=e.source.RUN,methods=list(e.source.METHODS)))
                with self.assertRaises(ValueError):p.prerelease_source_check(package)

    def test_prerelease_rule_checks_integer_physical_counts(self):
        e=p.c.evaluation
        for value in (True,26568.,27001,-1):
            s=deepcopy(completed_source());s['source_scores'][e.source.METHODS[0]]['raw_correct']=value
            with tempfile.TemporaryDirectory() as td:
                file=Path(td)/'complete.json';file.write_text(json.dumps(s));package=dict(source_check=dict(path=str(file),run_id=e.source.RUN,methods=list(e.source.METHODS)))
                with self.assertRaises(ValueError):p.prerelease_source_check(package)


if __name__=='__main__':unittest.main()
