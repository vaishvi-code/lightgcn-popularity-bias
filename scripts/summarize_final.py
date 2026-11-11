"""Summarize a complete frozen final-test evaluation without selecting models/seeds."""
import argparse,csv,hashlib,json,statistics
from collections import defaultdict
from pathlib import Path
from final_metrics import METRICS

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--plan',type=Path,required=True)
args=parser.parse_args();plan=json.loads(args.plan.read_text());plan_hash=hashlib.sha256(args.plan.read_bytes()).hexdigest()
report=json.loads((args.output/'test-completion.json').read_text())
assert report['all_passed'] and report['plan_sha256']==plan_hash
assert {r['run_id'] for r in report['evaluations']}=={e['run_id'] for e in plan['evaluations']}
rows=[];groups=defaultdict(list);shared=None
for entry in plan['evaluations']:
 m=json.loads((args.output/'test'/entry['run_id']/'metrics.json').read_text())
 assert m['evaluation_split']=='test' and m['plan_sha256']==plan_hash
 assert m['source_sha256']==report['source_sha256']
 assert m['test_file_sha256']==report['test_file_sha256']
 audit=tuple(m[k] for k in ['fitting_file_sha256','validation_file_sha256','test_file_sha256','popularity_file_sha256','evaluation_users','tail_evaluation_users','catalog_size','top_k'])
 if shared is None:shared=audit
 assert audit==shared
 assert m['model']==entry['model'] and m['training_seed']==entry['training_seed']
 rows.append({k:m[k] for k in ['model','run_id','comparison_group','training_seed','checkpoint_epoch','evaluation_split','evaluation_users','tail_evaluation_users',*METRICS]})
 groups[(m['comparison_group'],m['model'])].append(m)
summary=[]
for (group,model),entries in groups.items():
 if group=='main_three_seed':assert sorted(e['training_seed'] for e in entries)==[2020,2021,2022]
 else:assert len(entries)==1
 for metric in METRICS:
  values=[e[metric] for e in entries];assert all(v is not None for v in values)
  summary.append({'model':model,'comparison_group':group,'metric':metric,'evaluation_split':'test','run_count':len(values),'mean':statistics.mean(values),'sample_std':statistics.stdev(values) if len(values)>1 else '', 'run_ids':';'.join(e['run_id'] for e in entries)})
for name,content in [('test-per-run.csv',rows),('test-summary.csv',summary)]:
 with (args.output/name).open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(content[0]));writer.writeheader();writer.writerows(content)
print('Wrote all 23 per-run results and grouped summaries; single-seed variants remain separate.')
