"""Check grouped reporting without using test data or actual model outcomes."""
import hashlib,json,subprocess,sys,tempfile,csv
from pathlib import Path
base=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(base/'scripts'))
from final_metrics import METRICS
with tempfile.TemporaryDirectory() as tmp:
 root=Path(tmp);plan={'evaluations':[]}
 for model,group,seeds in [('Model A','main_three_seed',[2020,2021,2022]),('Variant B','exploratory_single_seed',[2020]),('MostPopular','deterministic_baseline',[None])]:
  for seed in seeds:plan['evaluations'].append({'model':model,'run_id':f'{model}-{seed}','comparison_group':group,'training_seed':seed})
 p=root/'plan.json';p.write_text(json.dumps(plan));sha=hashlib.sha256(p.read_bytes()).hexdigest();records=[]
 for e in plan['evaluations']:
  d=root/'test'/e['run_id'];d.mkdir(parents=True)
  value={2020:1.,2021:2.,2022:3.,None:0.}[e['training_seed']]
  m={**e,'evaluation_split':'test','checkpoint_epoch':1,'plan_sha256':sha,'source_sha256':{'code':'frozen'},'fitting_file_sha256':'fit','validation_file_sha256':'val','test_file_sha256':'test','popularity_file_sha256':'pop','evaluation_users':2,'tail_evaluation_users':1,'catalog_size':30,'top_k':20,**{k:value for k in METRICS}}
  (d/'metrics.json').write_text(json.dumps(m));records.append({'run_id':e['run_id'],'status':'passed'})
 report={'all_passed':True,'plan_sha256':sha,'source_sha256':{'code':'frozen'},'test_file_sha256':'test','evaluations':records}
 (root/'test-completion.json').write_text(json.dumps(report))
 command=[sys.executable,str(base/'scripts/summarize_final.py'),'--plan',str(p),'--output',str(root)]
 subprocess.run(command,check=True,capture_output=True)
 rows=list(csv.DictReader((root/'test-summary.csv').open()))
 for row in rows:
  if row['model']=='Model A':assert float(row['mean'])==2 and float(row['sample_std'])==1 and row['run_count']=='3'
  else:assert row['sample_std']=='' and row['run_count']=='1'
 report['all_passed']=False;(root/'test-completion.json').write_text(json.dumps(report))
 assert subprocess.run(command,capture_output=True).returncode!=0
print('PASS: sample standard deviation, separate single-seed/baseline reporting, incomplete-evaluation rejection')
