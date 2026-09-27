import json
import math
from dataclasses import replace
import numpy as np
import pytest
from product_search.data.schema import Product
from product_search.retrieval.common import Hit, product_text
from product_search.retrieval.lexical import BM25Retriever
from product_search.retrieval.dense import DenseRetriever, normalize
from product_search.retrieval.hybrid import HybridRetriever, rrf
from product_search.evaluation import query_metrics, evaluate, complementarity
from product_search.development_data import BoundaryGuard, load_development

@pytest.fixture
def products():
    return [Product(i, text, "", "", "", "", None, None, None)
            for i,text in [(8,"red chair"),(2,"red chair"),(15,"blue table")]]

def test_bm25_mapping_ties_determinism_and_reload(products,tmp_path):
    model=BM25Retriever.build(products)
    assert [h.product_id for h in model.search("RED chair")] == [2,8]
    assert model.search("red chair")==BM25Retriever.build(products[::-1]).search("red chair")
    assert model.search("unseenword")==[]
    config={"representation":"A","k1":1.5,"b":0.75}
    model.save(tmp_path,config)
    loaded=BM25Retriever.load(tmp_path,config)
    assert model.search("red")==loaded.search("red")
    with pytest.raises(ValueError,match="compatibility"):
        BM25Retriever.load(tmp_path,{**config,"b":0.5})

@pytest.mark.parametrize("bad",[0,-1,True,1.5,"10"])
def test_top_k_validation(products,bad):
    models=[BM25Retriever.build(products),DenseRetriever([1],[[1,0]])]
    for model in models:
        with pytest.raises(ValueError): model.search("",bad)

def test_empty_query(products):
    lexical=BM25Retriever.build(products)
    dense=DenseRetriever([2,8,15],[[1,0],[1,0],[0,1]])
    hybrid=HybridRetriever(lexical,dense)
    for model in (lexical,dense,hybrid): assert model.search(" \n")==[]
    with pytest.raises(ValueError): hybrid.search("x",101)

def test_dense_validation_ranking_and_artifacts(tmp_path):
    vectors=normalize([[3,4],[3,4],[4,-3]])
    assert np.allclose(np.linalg.norm(vectors,axis=1),1)
    model=DenseRetriever([8,2,10],vectors)
    assert [h.product_id for h in model.search_vector([3,4],2)]==[2,8]
    with pytest.raises(ValueError,match="dimension"): model.search_vector([1,2,3])
    with pytest.raises(ValueError): DenseRetriever([1,1],[[1,0],[0,1]])
    with pytest.raises(ValueError): DenseRetriever([1],[[3,4]])
    with pytest.raises(ValueError): normalize([[0,0]])
    with pytest.raises(ValueError): normalize([[float("nan"),0]])
    cfg={"dimension":2,"revision":"fixture"}
    model.save(tmp_path,cfg)
    assert DenseRetriever.load(tmp_path,cfg).search_vector([3,4])==model.search_vector([3,4])
    (tmp_path/'embeddings.npy').write_bytes(b'bad')
    with pytest.raises(ValueError,match="checksum"): DenseRetriever.load(tmp_path,cfg)

def test_rrf_known_scores_and_stable_ties():
    a=[Hit(8,9,1,"a"),Hit(2,1,2,"a")]
    b=[Hit(2,8,1,"b"),Hit(8,0,2,"b")]
    hits=rrf([a,b],constant=60)
    assert [h.product_id for h in hits]==[2,8]
    assert hits[0].score==pytest.approx(1/61+1/62)
    assert hits==rrf([b,a],constant=60)
    with pytest.raises(ValueError): rrf([a+a])

def test_metrics_hand_calculation_and_unknowns():
    m=query_metrics([20,999,10],{10:2,20:1,30:0})
    assert m['Recall@10']==1
    assert m['NDCG@10']==pytest.approx((1+3/math.log2(4))/(3+1/math.log2(3)))
    assert m['judged_fraction@100']==pytest.approx(2/3)
    assert query_metrics([10],{10:0})['Recall@10'] is None
    assert query_metrics([10],{10:0})['NDCG@20'] is None
    result=evaluate({1:[10],2:[20]},{1:{10:2},2:{20:0}})
    assert result['metrics']['Recall@20']==1
    assert result['excluded_queries']['Recall@20']==1
    assert result['excluded_queries']['NDCG@10']==1
    with pytest.raises(ValueError): query_metrics([10,10],{10:2})

def test_complementarity_accounting():
    c=complementarity({1:[1,2]},{1:[2,3]},{1:[2,3]},{1:{1:1,2:2,3:1}})['20']
    assert c['overlap_count']['sum']==1
    assert c['hybrid_recovered_beyond_bm25']['sum']==1
    assert c['hybrid_lost_vs_bm25']['sum']==1

def test_boundary_rejects_test_before_io(tmp_path):
    with pytest.raises(ValueError,match="Only train/validation"):
        load_development(tmp_path,'test')
    guard=BoundaryGuard(tmp_path)
    for name in ('data/processed/judgments.test.jsonl','data/raw/label.csv','data/processed/judgment_conflicts.jsonl'):
        with pytest.raises(PermissionError): guard.check('open',(str(tmp_path/name),'r',0))
    guard.check('open',(str(tmp_path/'data/processed/judgments.train.jsonl'),'r',0))

def test_representations_preserve_identity(products):
    p=replace(products[0],product_class='Chair',category_hierarchy='Furniture/Chair',product_description='A description',product_features='color:red')
    assert product_text(p,'A')=='red chair'
    assert 'description' not in product_text(p,'B')
    assert 'A description' in product_text(p,'C')

def test_scoped_loader_skips_test_query_decode_and_never_opens_test_labels(tmp_path,products,monkeypatch):
    from dataclasses import asdict
    from pathlib import Path
    from product_search.data.io import file_hash
    directory=tmp_path/'processed'; directory.mkdir()
    payloads={
        'products.jsonl': '\n'.join(json.dumps(asdict(p)) for p in products)+'\n',
        'queries.jsonl': json.dumps({'query_id':0,'query':'chair','query_class':'','normalized_query':'chair'},separators=(',',':'))+'\n'+json.dumps({'query_id':1,'intentionally_not_a_query_schema':True},separators=(',',':'))+'\n',
        'judgments.train.jsonl': json.dumps({'query_id':0,'product_id':2,'label':'Exact','relevance':2})+'\n'}
    for name,body in payloads.items(): (directory/name).write_text(body,encoding='utf-8')
    manifest={'query_ids':{'train':[0],'validation':[],'test':[1]},'artifacts':{name:{'sha256':file_hash(directory/name)} for name in payloads}}
    (directory/'data_manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
    opened=[]; original=Path.open
    def observed(path,*args,**kwargs):
        opened.append(path.name)
        assert path.name not in ('judgments.test.jsonl','judgment_conflicts.jsonl')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',observed)
    data=load_development(directory,'train')
    assert [q.query_id for q in data.queries]==[0]
    assert set(opened)==set(payloads)|{'data_manifest.json'}

def test_recall_cutoffs_and_empty_ranking():
    judgments={i:1 for i in range(12)}
    metrics=query_metrics(list(range(12)),judgments)
    assert metrics['Recall@10']==pytest.approx(10/12)
    assert metrics['Recall@20']==1
    assert query_metrics([],judgments)['Recall@10']==0
    assert query_metrics([],judgments)['NDCG@10']==0
