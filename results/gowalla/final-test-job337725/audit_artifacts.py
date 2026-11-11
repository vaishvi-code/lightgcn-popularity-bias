"""Reproduce the artifact audit for the frozen job-337725 results (standard library only)."""
from pathlib import Path
import csv, hashlib, json, math, statistics, struct
from collections import defaultdict
out=Path(__file__).resolve().parent
root=out.parents[2]
plan_path=root/'provenance/final-test-plan.json'
plan=json.loads(plan_path.read_text()); plan_hash=hashlib.sha256(plan_path.read_bytes()).hexdigest()
gate=json.loads((out/'validation-verification.json').read_text()); completion=json.loads((out/'test-completion.json').read_text())
assert plan_hash=='5f459b7479142af789861eb0179fde02a0ac0d76b3b7627dae3a189f3b1fa709'
assert gate['all_passed'] and completion['all_passed']
assert gate['plan_sha256']==completion['plan_sha256']==plan_hash
assert gate['source_sha256']==completion['source_sha256']
assert gate['test_file_sha256'] is None and completion['test_file_sha256']
for name,h in gate['source_sha256'].items(): assert hashlib.sha256((root/'scripts'/name).read_bytes()).hexdigest()==h
expected={e['run_id']:e for e in plan['evaluations']}
verified={r['run_id']:r for r in gate['evaluations']}; tested={r['run_id']:r for r in completion['evaluations']}
assert len(expected)==len(verified)==len(tested)==23 and expected.keys()==verified.keys()==tested.keys()
release=json.loads((root/'provenance/seed2020-release-manifest.json').read_text())
for c in release['checkpoints']: assert verified[c['run_id']]['checkpoint_sha256']==c['sha256']
pop=list(csv.DictReader((root/'results/gowalla/item-popularity.csv').open()))
tail={int(r['item_id']) for r in pop if r['group']=='tail'}; n=len(pop)
metrics=('recall_at_20','ndcg_at_20','tail_share_at_20','coverage_at_20','exposure_gini','tail_recall_at_20')
counts_by_phase={}; groups=defaultdict(list)
for phase,report in [('verify',gate),('test',completion)]:
    population=set()
    for rid,entry in expected.items():
        m=json.loads((out/phase/rid/'metrics.json').read_text())
        assert m['plan_sha256']==plan_hash and m['source_sha256']==gate['source_sha256']
        assert m['checkpoint_sha256']==verified[rid]['checkpoint_sha256']==tested[rid]['checkpoint_sha256']
        assert m['test_file_sha256']==report['test_file_sha256']
        assert m['evaluation_split']==('test' if phase=='test' else 'validation')
        assert m['excluded_interactions']==('fitting + validation' if phase=='test' else 'fitting')
        assert m['graph_interactions']=='fitting only' and m['top_k']==20 and m['catalog_size']==n
        for k in ('fitting_file_sha256','validation_file_sha256','popularity_file_sha256'): assert m[k]==plan[k]
        assert (m['model'],m['training_seed'],m['checkpoint_epoch'],m['comparison_group'])==tuple(entry.get(k) for k in ('model','training_seed','checkpoint_epoch','comparison_group'))
        exposure=list(csv.DictReader((out/phase/rid/'exposure.csv').open()))
        assert [int(r['item_id']) for r in exposure]==list(range(n))
        c=[int(r['recommendation_count']) for r in exposure]
        assert all(v>=0 for v in c)
        slots=sum(c); distinct=sum(v>0 for v in c)
        assert slots==m['evaluation_users']*20==m['recommendation_slots'] and distinct==m['distinct_recommended_items']
        ordered=sorted(c)
        # Independent cumulative-area form of the full-catalog Gini.
        cumulative=0; area=0
        for value in ordered:
            area+=2*cumulative+value; cumulative+=value
        recomputed={'tail_share_at_20':sum(c[i] for i in tail)/slots,'coverage_at_20':distinct/n,'exposure_gini':1-area/(n*slots)}
        for key,value in recomputed.items(): assert math.isclose(value,m[key],rel_tol=0,abs_tol=1e-12),(rid,key)
        if phase=='verify':
            h=hashlib.sha256(struct.pack('<'+'q'*n,*c)).hexdigest()
            assert h==entry['validation_exposure_sha256']
            for key in metrics: assert math.isclose(m[key],entry['validation_metrics'][key],rel_tol=0,abs_tol=1e-6)
        else: groups[(m['comparison_group'],m['model'])].append(m)
        population.add((m['evaluation_users'],m['tail_evaluation_users']))
    assert len(population)==1
    counts_by_phase[phase]=list(population.pop())
summary=list(csv.DictReader((out/'test-summary.csv').open()))
for row in summary:
    runs=groups[(row['comparison_group'],row['model'])]; values=[m[row['metric']] for m in runs]
    assert math.isclose(float(row['mean']),statistics.mean(values),rel_tol=0,abs_tol=1e-14)
    if len(runs)==3:
        assert sorted(m['training_seed'] for m in runs)==[2020,2021,2022]
        assert math.isclose(float(row['sample_std']),statistics.stdev(values),rel_tol=0,abs_tol=1e-14)
    else: assert len(runs)==1 and row['sample_std']==''
audit={'arc_job_id':337725,'verified_on':'2026-11-11','all_passed':True,'validation_entries':23,'test_entries':23,'population_counts_evaluation_then_tail':counts_by_phase,'plan_sha256':plan_hash,'test_file_sha256':completion['test_file_sha256'],'checks':['Unchanged frozen plan, evaluator sources and checkpoint identities across validation and test','All four release checkpoint hashes match the published manifest','All validation metrics and exact per-item exposures match the frozen references','All 46 exposure tables independently reproduce tail share, coverage, full-catalog Gini and slot totals','Consistent split/popularity hashes, exclusion rules, catalog and user denominators','All grouped test means and sample standard deviations reproduced; seeds and exploratory groups kept separate'],'limitation':'Recall, NDCG and tail recall are evaluator measurements; this artifact audit does not independently rerank test users.'}
(out/'artifact-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps(audit,indent=2))
for key,runs in groups.items():
    print(key,len(runs),{metric:statistics.mean(m[metric] for m in runs) for metric in metrics})
