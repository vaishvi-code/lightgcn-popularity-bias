"""Frozen validation-selected checkpoint evaluation; test requires a complete verification gate."""
import argparse,csv,gc,hashlib,json,math,os,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
from final_metrics import METRICS,RankingMetrics,load_interactions,exclusion_sets


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def exposure_digest(counts):
    return hashlib.sha256(np.asarray(counts,dtype='<i8').tobytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['verify','test'],required=True)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--project-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--checkpoint-store',type=Path,required=True)
    args=parser.parse_args();plan=json.loads(args.plan.read_text());base=args.project_root.resolve()
    plan_hash=digest(args.plan)
    source_hashes={name:digest(Path(__file__).parent/name) for name in
                   ['evaluate_final.py','final_metrics.py','normalized_graph.py','weighted_lightgcn.py']}
    assert plan['top_k']==20 and plan['training_policy']=='frozen_best_validation_checkpoints_no_retraining'
    upstream=base/'LightGCN-PyTorch'
    assert subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'],text=True).strip()==plan['upstream_commit']
    data=base/'prepared/gowalla'
    hashes={name:digest(data/file) for name,file in [('fitting_file_sha256','train.txt'),('validation_file_sha256','val.txt')]}
    assert all(hashes[k]==plan[k] for k in hashes)
    pop_path=base/'results/gowalla/item-popularity.csv';assert digest(pop_path)==plan['popularity_file_sha256']
    rows=list(csv.DictReader(pop_path.open()));n=len(rows)
    degrees={int(r['item_id']):int(r['fitting_degree']) for r in rows};tail={int(r['item_id']) for r in rows if r['group']=='tail'}
    assert set(degrees)==set(range(n)) and n==plan['catalog_size']
    fit=load_interactions(data/'train.txt');val=load_interactions(data/'val.txt')
    assert all(fit.get(u,set()).isdisjoint(items) for u,items in val.items())
    measured_degrees=np.zeros(n,dtype=np.int64)
    for items in fit.values():
        for i in items:measured_degrees[i]+=1
    assert all(measured_degrees[i]==d for i,d in degrees.items())
    head=set(sorted(degrees,key=lambda i:(-degrees[i],i))[:math.ceil(.2*n)])
    assert tail==set(range(n))-head
    known=exclusion_sets(fit,val,'test')
    original=upstream/'data/gowalla/train.txt'
    assert digest(original)==plan['original_train_sha256']
    assert {u:v for u,v in known.items() if v}=={u:v for u,v in load_interactions(original).items() if v}
    checkpoint_paths={}
    for entry in plan['evaluations']:
        if entry['kind']=='mostpopular':continue
        local=args.checkpoint_store/entry['run_id']/'best.pt'
        fallback=Path('/mnt/beegfs')/os.environ['USER']/entry['checkpoint_project']/'results/gowalla'/entry['run_id']/'best.pt' if entry['checkpoint_project'] else local
        checkpoint_paths[entry['run_id']]=local if local.exists() else fallback
    gate_path=args.output/'validation-verification.json'
    args.output.mkdir(parents=True,exist_ok=True)
    if args.phase=='test':
        gate=json.loads(gate_path.read_text())
        assert gate['all_passed'] and gate['plan_sha256']==plan_hash and gate['source_sha256']==source_hashes, 'Complete matching validation verification is required'
        assert len(gate['evaluations'])==len(plan['evaluations'])
        for entry in gate['evaluations']:
            if entry['checkpoint_sha256']:
                assert digest(checkpoint_paths[entry['run_id']])==entry['checkpoint_sha256'], 'Checkpoint changed after verification'
        # Test labels are first opened only after the complete validation gate passes.
        test_file=upstream/'data/gowalla/test.txt'
        actual=subprocess.check_output(['git','-C',str(upstream),'hash-object',str(test_file)],text=True).strip()
        expected=subprocess.check_output(['git','-C',str(upstream),'rev-parse',"HEAD:data/gowalla/test.txt"],text=True).strip()
        assert actual==expected, 'Authors test file differs from pinned upstream'
        relevant=load_interactions(test_file);test_hash=digest(test_file)
    else:
        relevant=val;test_hash=None
    users=sorted(u for u,items in relevant.items() if items)
    seen=exclusion_sets(fit,val,'test' if args.phase=='test' else 'validation')
    for u in users:
        assert relevant[u].isdisjoint(seen.get(u,set())), 'Held-out positives overlap excluded observations'
        assert all(0<=i<n for i in relevant[u]) and n-len(seen.get(u,set()))>=20
    sys.argv=[sys.argv[0]]
    sys.path.insert(0,str(upstream/'code'))
    import world
    from normalized_graph import build_dataset,normalization_spec
    from weighted_lightgcn import build_model
    assert torch.cuda.is_available(), 'Use an ARC GPU allocation'
    torch.set_num_threads(4);world.tensorboard=0
    loader=base/'prepared'/('gowalla-validation-'+hashes['fitting_file_sha256'][:12]+'-'+hashes['validation_file_sha256'][:12])
    assert digest(loader/'train.txt')==hashes['fitting_file_sha256'] and digest(loader/'test.txt')==hashes['validation_file_sha256']
    records=[];datasets={}
    for entry in plan['evaluations']:
        rid=entry['run_id'];checkpoint_hash=None;network=None;checkpoint=None;start=time.perf_counter()
        print(f"BEGIN {args.phase} {rid}",flush=True)
        if entry['kind']=='checkpoint':
            path=checkpoint_paths[rid]
            if not path.is_file():
                records.append({'run_id':rid,'status':'missing_checkpoint','checkpoint_sha256':None,'path':str(path)})
                print(f'MISSING_CHECKPOINT {rid}: {path}',flush=True);continue
            checkpoint_hash=digest(path);checkpoint=torch.load(path,map_location='cpu',weights_only=False);meta=checkpoint['metadata']
            assert meta['config']==entry['config'] and meta['training_seed']==entry['training_seed']
            assert all(meta[k]==hashes[k] for k in hashes)
            assert meta.get('appnp_beta')==entry['appnp_beta']
            norm=meta.get('normalization',normalization_spec())
            assert norm['a']==entry['normalization_a'] and norm['b']==entry['normalization_b']
            assert checkpoint['epoch']==entry['checkpoint_epoch']
            for key in ['recall_at_20','ndcg_at_20']:
                assert math.isclose(checkpoint['validation_'+key],entry['validation_metrics'][key],abs_tol=1e-6)
            world.config.update(meta['config'])
            ds_key=(norm['a'],norm['b'])
            if ds_key not in datasets:datasets[ds_key]=build_dataset(world.config,loader,*ds_key)
            dataset=datasets[ds_key]
            assert dataset.m_items==n and all(0<=u<dataset.n_users for u in users)
            network=build_model(world.config,dataset,meta.get('appnp_beta')).to(world.device)
            network.load_state_dict(checkpoint['model_state_dict']);network.eval()
        accumulator=RankingMetrics(n,tail)
        if network is None:
            ranking=sorted(degrees,key=lambda i:(-degrees[i],i))
            for u in users:
                excluded=seen.get(u,set());recommendations=[]
                for i in ranking:
                    if i not in excluded:recommendations.append(i)
                    if len(recommendations)==20:break
                accumulator.add(recommendations,relevant[u],excluded)
        else:
            with torch.no_grad():
                for start_user in range(0,len(users),256):
                    batch=users[start_user:start_user+256]
                    scores=network.getUsersRating(torch.tensor(batch,device=world.device))
                    assert torch.isfinite(scores).all()
                    for row,u in enumerate(batch):scores[row,list(seen.get(u,set()))]=-torch.inf
                    top=torch.topk(scores,k=20,dim=1).indices.cpu().numpy()
                    for u,items in zip(batch,top):accumulator.add(items,relevant[u],seen.get(u,set()))
        metrics=accumulator.finish()
        if args.phase=='verify':
            for key in METRICS:
                expected=entry['validation_metrics'][key]
                assert metrics[key]==expected if expected is None else math.isclose(metrics[key],expected,rel_tol=0,abs_tol=1e-6), (rid,key,metrics[key],expected)
            for key in ['evaluation_users','tail_evaluation_users']:assert metrics[key]==entry['validation_metrics'][key]
            assert exposure_digest(accumulator.exposure)==entry['validation_exposure_sha256'], 'Validation item-exposure counts changed'
        result={'model':entry['model'],'run_id':rid,'comparison_group':entry['comparison_group'],
                'evaluation_split':'test' if args.phase=='test' else 'validation','training_seed':entry['training_seed'],
                'checkpoint_epoch':entry.get('checkpoint_epoch'),'checkpoint_sha256':checkpoint_hash,
                'top_k':20,'plan_sha256':plan_hash,'source_sha256':source_hashes,**hashes,
                'test_file_sha256':test_hash,'popularity_file_sha256':plan['popularity_file_sha256'],
                'graph_interactions':'fitting only','excluded_interactions':'fitting + validation' if args.phase=='test' else 'fitting',
                'elapsed_seconds':time.perf_counter()-start,**metrics}
        dest=args.output/args.phase/rid;dest.mkdir(parents=True,exist_ok=False)
        (dest/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
        with (dest/'exposure.csv').open('w',newline='') as f:
            writer=csv.writer(f);writer.writerow(['item_id','recommendation_count']);writer.writerows(enumerate(accumulator.exposure.tolist()))
        records.append({'run_id':rid,'status':'passed','checkpoint_sha256':checkpoint_hash})
        print(f"PASS {args.phase} {rid} recall={metrics['recall_at_20']:.8f}",flush=True)
        del network,checkpoint;gc.collect();torch.cuda.empty_cache()
    report={'phase':args.phase,'plan_sha256':plan_hash,'source_sha256':source_hashes,
            'all_passed':len(records)==len(plan['evaluations']) and all(r['status']=='passed' for r in records),
            'evaluations':records,'test_file_sha256':test_hash}
    report_path=gate_path if args.phase=='verify' else args.output/'test-completion.json'
    report_path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    if not report['all_passed']:return 2
    return 0


if __name__=='__main__':raise SystemExit(main())
