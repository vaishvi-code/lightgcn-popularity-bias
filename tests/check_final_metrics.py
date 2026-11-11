"""Small independent checks, including held-out-positive masking at test time."""
import math,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from final_metrics import RankingMetrics, exclusion_sets
fit={0:{0},1:{1}};val={0:{2},1:{3}}
assert exclusion_sets(fit,val,'validation')[0]=={0}
assert exclusion_sets(fit,val,'test')[0]=={0,2}
assert fit=={0:{0},1:{1}}, 'Exclusions must not mutate the fitting graph input'
m=RankingMetrics(6,{3,4,5},k=2)
m.add([2,3],{3,4},{0,1})
m.add([2,4],{2},{0,1})
r=m.finish()
assert math.isclose(r['recall_at_20'],.75)
assert math.isclose(r['ndcg_at_20'],((1/math.log2(3))/(1+1/math.log2(3))+1)/2)
assert r['tail_recall_at_20']==.5 and r['tail_evaluation_users']==1
assert r['tail_share_at_20']==.5 and r['coverage_at_20']==.5
# Gini independently from all pairwise absolute differences, including zeros.
counts=[0,0,2,1,1,0]
gini=sum(abs(a-b) for a in counts for b in counts)/(2*6*sum(counts))
assert math.isclose(r['exposure_gini'],gini)
for items,relevant,seen in [([2,3],{3},{2}),([2,2],{3},{0}),([2,3],{0},{0})]:
 try: RankingMetrics(6,{3},2).add(items,relevant,seen)
 except ValueError: pass
 else: raise AssertionError('Invalid ranking or leakage accepted')
print('PASS: recall, NDCG, tail-user denominator, full-catalog Gini, coverage, and validation/test exclusions')
