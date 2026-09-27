import math
import numpy as np
import pytest
from product_search.data.schema import Product
from product_search.retrieval.common import Hit
from product_search.reranking import CrossEncoderReranker, TransformerPairScorer, validate_reranker_config
from product_search.evaluation import query_metrics
from product_search.paired import paired_comparison

class FakeScorer:
    def score_pairs(self,pairs,batch_size,max_length):
        # Independent of batch padding; emulates stable per-pair logits.
        return [float(d.split()[0]) for q,d in pairs],{'batch_sizes':[len(pairs[i:i+batch_size]) for i in range(0,len(pairs),batch_size)]}

@pytest.fixture
def fixture():
    products=[Product(i,str(i)+' title','','','','',None,None,None) for i in range(6)]
    candidates=[Hit(i,1/(i+1),i+1,'hybrid_rrf') for i in range(6)]
    return products,candidates

def test_membership_scores_provenance_and_batch_invariance(fixture):
    products,candidates=fixture
    a=CrossEncoderReranker(FakeScorer(),products,batch_size=2).rerank('query',candidates,6)
    b=CrossEncoderReranker(FakeScorer(),products,batch_size=32).rerank('query',candidates,6)
    assert a.hits==b.hits
    assert [h.product_id for h in a.hits]==[5,4,3,2,1,0]
    assert a.timings['batch_sizes']==[2,2,2]
    assert a.hits[0].retrieval_rank==6 and a.hits[0].final_rank==1
    assert a.hits[0].retrieval_score==candidates[5].score
    before=query_metrics([h.product_id for h in candidates],{i:i%3 for i in range(6)})
    after=query_metrics([h.product_id for h in a.hits],{i:i%3 for i in range(6)})
    assert before['Recall@10']==after['Recall@10']
    assert before['NDCG@10']!=after['NDCG@10']

def test_stable_ties(fixture):
    class Equal:
        def score_pairs(self,pairs,*args): return [1]*len(pairs),{}
    products,candidates=fixture
    result=CrossEncoderReranker(Equal(),products).rerank('query',candidates,6)
    assert [h.product_id for h in result.hits]==[h.product_id for h in candidates]

@pytest.mark.parametrize('output',[[1], [1,2,float('nan'),4,5,6], [[1]]*6, [float('inf')]*6])
def test_bad_model_output_falls_back(fixture,output):
    class Bad:
        def score_pairs(self,*args): return output,{}
    products,candidates=fixture
    reranker=CrossEncoderReranker(Bad(),products)
    result=reranker.rerank('query',candidates,6)
    assert result.fallback_used and result.error_type=='ValueError'
    assert [h.product_id for h in result.hits]==[h.product_id for h in candidates]
    assert all(h.reranker_score is None for h in result.hits)
    with pytest.raises(ValueError): reranker.rerank('query',candidates,6,fallback=False)

def test_exception_fallback_and_depth_validation(fixture):
    class Broken:
        def score_pairs(self,*args): raise RuntimeError('synthetic inference failure')
    products,candidates=fixture
    reranker=CrossEncoderReranker(Broken(),products,max_candidates=6)
    assert reranker.rerank('q',candidates,3).fallback_used
    for k in (0,-1,True,7):
        with pytest.raises(ValueError): reranker.rerank('q',candidates,k)
    with pytest.raises(ValueError): reranker.rerank('q',candidates+candidates,3)
    assert reranker.rerank('q',[],3).hits==()

def test_paired_bootstrap():
    result=paired_comparison({1:.5,2:.4,3:.3},{1:.7,2:.4,3:.2},samples=1000)
    assert result['mean_delta']==pytest.approx(1/30)
    assert (result['improved'],result['unchanged'],result['worsened'])==(1,1,1)
    assert result==paired_comparison({1:.5,2:.4,3:.3},{1:.7,2:.4,3:.2},samples=1000)
    with pytest.raises(ValueError): paired_comparison({1:0},{2:0})

def test_manifest_config_mismatch():
    validate_reranker_config({'revision':'one'},{'revision':'one'})
    with pytest.raises(ValueError): validate_reranker_config({'revision':'one'},{'revision':'two'})

def test_truncation_policy_is_fixed_without_model_download():
    class Tokenizer:
        def __call__(self,queries,docs,**kwargs):
            assert kwargs=={'padding':False,'truncation':False}
            return {'input_ids':[list(range(len(q.split())+len(d.split())+3)) for q,d in zip(queries,docs)]}
    scorer=object.__new__(TransformerPairScorer); scorer.tokenizer=Tokenizer()
    pairs=[('one','two three'),('one','two three four five')]
    assert scorer.truncation_stats(pairs,6)==scorer.truncation_stats(pairs,6)
    assert scorer.truncation_stats(pairs,6)['pairs_over_max_length']==1

@pytest.mark.parametrize('depth',[20,50,100])
def test_recall_at_candidate_depth_is_invariant(depth):
    products=[Product(i,str(i)+' title','','','','',None,None,None) for i in range(100)]
    hits=[Hit(i,1/(i+1),i+1,'hybrid') for i in range(depth)]
    reranker=CrossEncoderReranker(FakeScorer(),products)
    result=reranker.rerank('q',hits,depth)
    labels={i:i%3 for i in range(100)}
    before=query_metrics([h.product_id for h in hits],labels)
    after=query_metrics([h.product_id for h in result.hits],labels)
    assert before[f'Recall@{depth}']==after[f'Recall@{depth}']
    assert {h.product_id for h in result.hits}=={h.product_id for h in hits}
