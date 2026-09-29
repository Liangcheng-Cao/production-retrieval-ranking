"""Synthetic checks of final-evaluation boundaries; no real final data access."""
import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('phase9',Path(__file__).resolve().parents[1]/'scripts/run_phase9.py')
p9=importlib.util.module_from_spec(spec)
spec.loader.exec_module(p9)


def test_evidence_is_exclusive_and_checksummed(tmp_path):
    path=tmp_path/'evidence.json'
    sha=p9.write_once(path,{'value':1})
    assert sha==p9.sha(path)
    with pytest.raises(FileExistsError):p9.write_once(path,{'value':2})
    assert p9.read(path)=={'value':1}


def test_label_guard_requires_authorization_and_audits_first(tmp_path):
    guard=p9.LabelGuard(tmp_path)
    path=tmp_path/'data/processed/judgments.test.jsonl'
    with pytest.raises(PermissionError):guard.check('open',(str(path),'r',0))
    events=[]
    guard.allowed=True;guard.callback=lambda:events.append('audit-before-open')
    guard.check('open',(str(path),'r',0))
    events.append('simulated-OS-open')
    assert events==['audit-before-open','simulated-OS-open'] and guard.opens==1
    with pytest.raises(PermissionError):guard.check('open',(str(path),'r',0))


@pytest.mark.parametrize('name',['data/raw/label.csv','data/processed/judgment_conflicts.jsonl'])
def test_raw_and_conflicts_remain_forbidden(tmp_path,name):
    guard=p9.LabelGuard(tmp_path);guard.allowed=True
    with pytest.raises(PermissionError):guard.check('open',(str(tmp_path/name),'rb',0))


def test_empty_tracked_directory_marker_is_not_raw_data(tmp_path):
    guard=p9.LabelGuard(tmp_path)
    guard.check('open',(str(tmp_path/'data/raw/.gitkeep'),'rb',0))
    assert not guard.blocked


def test_label_loader_membership_and_duplicate_rejection():
    line=b'{"query_id":1,"product_id":2,"label":"Exact","relevance":2}\n'
    assert p9.labels_from_bytes(line,{1,3})==({1:{2:2},3:{}},1)
    with pytest.raises(AssertionError):p9.labels_from_bytes(line,{3})
    with pytest.raises(ValueError):p9.labels_from_bytes(line*2,{1})


def test_summary_ce_depth_exclusion_pairing_and_aggregation():
    rankings={mode:{1:[2,3],4:[2,3]} for mode in p9.PIPELINES}
    rankings['Hybrid + CE-20'][1]=[3,2]
    labels={1:{2:1,3:2},4:{2:0}}
    result=p9.summarize(rankings,labels,{'bootstrap':{'seed':42,'samples':10000}})
    quality=result['final_quality.json']['pipelines']
    for mode in p9.PIPELINES:
        assert quality[mode]['excluded_queries']['Recall@20']==1
        assert quality[mode]['excluded_queries']['NDCG@10']==1
    assert quality['Hybrid + CE-20']['metrics']['Recall@100'] is None
    assert quality['Hybrid + CE-20']['metrics']['Recall@50'] is None
    assert quality['Hybrid']['metrics']['Recall@20']==quality['Hybrid + CE-20']['metrics']['Recall@20']==1
    assert result['paired_final.json']['NDCG@10']['eligible_queries']==1
    assert result['paired_final.json']['NDCG@10']['improved']==1
    assert result['ranking_diagnostics_final.json']['ordering_changed_fraction']==.5
    assert result==p9.summarize(rankings,labels,{'bootstrap':{'seed':42,'samples':10000}})


def test_deterministic_comparison_does_not_hide_differences():
    assert p9.max_difference({'x':[1,None]},{'x':[1.00001,None]})>0
    with pytest.raises(AssertionError):p9.max_difference({'x':1},{'y':1})


def test_seal_detects_modified_evidence(tmp_path):
    evidence=tmp_path/'result.json';p9.write_once(evidence,{'a':1})
    seal=tmp_path/'seal.json';p9.write_once(seal,{'files':{'result.json':p9.sha(evidence)}})
    p9.verify_seal(seal)
    evidence.write_text('{}')
    with pytest.raises(AssertionError):p9.verify_seal(seal)
