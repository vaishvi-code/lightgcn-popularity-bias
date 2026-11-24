"""Validation-only sensitivity of frozen rankings to removal of item-vector magnitude."""
import argparse
import csv
import gc
import gzip
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch
from final_metrics import METRICS, RankingMetrics, load_interactions
from magnitude_scoring import equal_length_items, score
from supplemental_test_gate import assert_reconciles, plan_digest, require_gate, source_digest
from user_groups import GROUPS, assign_group


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer=csv.DictWriter(stream, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def ranking_change(reference, changed):
    intersection=np.array([len(set(a) & set(b)) for a,b in zip(reference,changed)], dtype=float)
    return dict(mean_top20_overlap=float(np.mean(intersection/20)), mean_jaccard=float(np.mean(intersection/(40-intersection))),
                fraction_top1_changed=float(np.mean(reference[:,0]!=changed[:,0])),
                fraction_sets_changed=float(np.mean(intersection<20)),
                fraction_ordered_lists_changed=float(np.mean(np.any(reference!=changed,axis=1))),
                mean_positional_agreement=float(np.mean(reference==changed)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase",choices=["verify","analyze"],required=True)
    parser.add_argument("--repository",type=Path,required=True)
    parser.add_argument("--project-root",type=Path,required=True)
    parser.add_argument("--checkpoint-root",type=Path,action="append",required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();repo,base=args.repository,args.project_root
    plan=json.loads((repo/"provenance/magnitude-plan.json").read_text());identity=plan_digest(plan)
    assert plan["evaluation_split"]=="validation" and not plan["test_evaluation_authorized"]
    assert len(plan["evaluations"])==9 and plan["scoring_modes"]==["legacy_sigmoid","raw_dot","unit_item"]
    sources={n:source_digest(repo/"scripts"/n) for n in plan["source_sha256"]};assert sources==plan["source_sha256"]
    paths,hashes={},{}
    for entry in plan["evaluations"]:
        run=entry["run_id"];path=next((r/run/"best.pt" for r in args.checkpoint_root if (r/run/"best.pt").is_file()),None)
        if path is None:raise FileNotFoundError(run)
        paths[run]=path;hashes[run]=digest(path);assert hashes[run]==entry["checkpoint_sha256"]
    gate_path=args.output/"validation-verification.json"
    if args.phase=="analyze":require_gate(json.loads(gate_path.read_text()),identity,sources,hashes)
    upstream=base/"LightGCN-PyTorch"
    assert subprocess.check_output(["git","-C",str(upstream),"rev-parse","HEAD"],text=True).strip()==plan["upstream_commit"]
    subprocess.run(["git","-C",str(upstream),"diff","--exit-code","HEAD","--","code/model.py","code/dataloader.py"],check=True)
    data=base/"prepared/gowalla"
    for filename,key in [("train.txt","fitting_file_sha256"),("val.txt","validation_file_sha256")]:assert digest(data/filename)==plan[key]
    fit,val=(load_interactions(data/f) for f in ["train.txt","val.txt"])
    popularity=repo/"results/gowalla/item-popularity.csv";assert digest(popularity)==plan["popularity_file_sha256"]
    with popularity.open() as stream:items=list(csv.DictReader(stream))
    n=plan["catalog_size"];assert [int(r["item_id"]) for r in items]==list(range(n))
    degree=np.array([int(r["fitting_degree"]) for r in items]);tail={int(r["item_id"]) for r in items if r["group"]=="tail"}
    assert tail==set(range(n))-set(sorted(range(n),key=lambda i:(-degree[i],i))[:math.ceil(.2*n)])
    users=sorted(u for u,v in val.items() if v);assert len(users)==29858 and all(val[u].isdisjoint(fit[u]) for u in users)
    groups={u:assign_group(len(fit[u]&tail),len(fit[u])) for u in users}
    args.output.mkdir(parents=True,exist_ok=True)
    write_csv(args.output/"group-profiles.csv",[dict(group=g,evaluation_users=sum(groups[u]==g for u in users),tail_evaluation_users=sum(groups[u]==g and bool(val[u]&tail) for u in users)) for g in GROUPS])
    sys.argv=[sys.argv[0]];sys.path.insert(0,str(upstream/"code"))
    import world
    from dataloader import Loader
    from model import LightGCN
    from single_lightgcn import SingleLayerOutputLightGCN
    assert torch.cuda.is_available();world.device=torch.device("cuda");world.tensorboard=0;torch.set_num_threads(4)
    loader=base/"prepared"/("gowalla-validation-"+plan["fitting_file_sha256"][:12]+"-"+plan["validation_file_sha256"][:12])
    assert digest(loader/"train.txt")==plan["fitting_file_sha256"] and digest(loader/"test.txt")==plan["validation_file_sha256"]
    dataset=Loader(config=plan["evaluations"][0]["config"],path=str(loader))
    assert np.array_equal(np.asarray(dataset.UserItemNet.sum(axis=0)).ravel(),degree)
    records,rows,changes=[],[],[]
    for entry in plan["evaluations"]:
        run=entry["run_id"];checkpoint=torch.load(paths[run],map_location="cpu",weights_only=False);meta=checkpoint["metadata"]
        assert meta["config"]==entry["config"] and meta["training_seed"]==entry["training_seed"] and checkpoint["epoch"]==entry["checkpoint_epoch"]
        assert meta.get("appnp_beta") is None
        normalization=meta.get("normalization",{"a":.5,"b":.5});assert normalization["a"]==normalization["b"]==.5
        assert meta.get("layer_combination","uniform")==("single" if entry["family"]=="single" else "uniform")
        for key in ["fitting_file_sha256","validation_file_sha256"]:assert meta[key]==plan[key]
        world.config.update(entry["config"])
        model=(SingleLayerOutputLightGCN if entry["family"]=="single" else LightGCN)(world.config,dataset).to(world.device)
        model.load_state_dict(checkpoint["model_state_dict"]);model.eval()
        with torch.no_grad():
            u_vectors,i_vectors=model.computer();unit,norms=equal_length_items(i_vectors,plan["epsilon"])
            assert torch.allclose(unit*norms.clamp_min(plan["epsilon"])[:,None],i_vectors,rtol=1e-6,atol=1e-7)
            norm_np=norms.cpu().numpy();user_norm=torch.linalg.vector_norm(u_vectors,dim=1).cpu().numpy()
            diagnostics=dict(zero_item_vectors=int(np.sum(norm_np==0)),sub_epsilon_item_vectors=int(np.sum((norm_np>0)&(norm_np<plan["epsilon"]))),
                zero_user_vectors=int(np.sum(user_norm==0)),mean_original_item_norm=float(norm_np.mean()),max_original_item_norm=float(norm_np.max()))
            modes=["legacy_sigmoid"] if args.phase=="verify" else ["raw_dot","unit_item","legacy_sigmoid"]
            base_recommendations=None;base_result=None
            for mode in modes:
                accumulator=RankingMetrics(n,tail);grouped_acc={g:RankingMetrics(n,tail) for g in GROUPS};recommendations=[]
                vectors=unit if mode=="unit_item" else i_vectors
                for offset in range(0,len(users),256):
                    batch=users[offset:offset+256];scores=score(u_vectors[torch.tensor(batch,device=world.device)],vectors,mode)
                    assert torch.isfinite(scores).all()
                    for row,user in enumerate(batch):scores[row,list(fit[user])]=-torch.inf
                    top=torch.topk(scores,k=20,dim=1).indices.cpu().numpy();recommendations.append(top)
                    for user,recs in zip(batch,top):accumulator.add(recs,val[user],fit[user]);grouped_acc[groups[user]].add(recs,val[user],fit[user])
                recommended=np.concatenate(recommendations);metrics=accumulator.finish();grouped={g:a.finish() for g,a in grouped_acc.items()}
                assert_reconciles(metrics,grouped);assert np.array_equal(sum(a.exposure for a in grouped_acc.values()),accumulator.exposure)
                deltas={};exposure_l1=None
                if mode=="legacy_sigmoid":
                    deltas={k:metrics[k]-entry["validation_metrics"][k] for k in METRICS}
                    assert all(abs(v)<=plan["metric_absolute_tolerance"] for v in deltas.values()),(run,deltas)
                    with (repo/entry["validation_exposure"]).open() as stream:old=list(csv.DictReader(stream))
                    assert [int(r["item_id"]) for r in old]==list(range(n))
                    exposure_l1=int(np.abs(accumulator.exposure-np.array([int(r["recommendation_count"]) for r in old])).sum())
                    assert exposure_l1<=plan["exposure_l1_tolerance"],(run,exposure_l1)
                result=dict(model=entry["model"],run_id=run,training_seed=entry["training_seed"],mode=mode,evaluation_split="validation",
                    checkpoint_sha256=hashes[run],checkpoint_epoch=entry["checkpoint_epoch"],plan_sha256=identity,source_sha256=sources,
                    test_labels_opened=False,original_test_already_viewed=True,grouped=grouped,**diagnostics,
                    reference_metric_deltas=deltas,reference_exposure_l1=exposure_l1,
                    mean_recommended_original_norm=float(np.dot(norm_np.astype(float),accumulator.exposure)/metrics["recommendation_slots"]),
                    mean_recommended_fitting_degree=float(np.dot(degree.astype(float),accumulator.exposure)/metrics["recommendation_slots"]),**metrics)
                dest=args.output/args.phase/run/mode;dest.mkdir(parents=True,exist_ok=False)
                (dest/"metrics.json").write_text(json.dumps(result,indent=2)+"\n")
                with (dest/"exposure.csv").open("w",newline="") as stream:
                    writer=csv.writer(stream);writer.writerow(["item_id","recommendation_count"]);writer.writerows(enumerate(accumulator.exposure.tolist()))
                if args.phase=="analyze":
                    if mode=="raw_dot":base_recommendations=recommended.copy();base_result=result
                    else:
                        change=dict(model=entry["model"],run_id=run,training_seed=entry["training_seed"],comparison=mode+"_vs_raw_dot",
                            metric_changes={k:metrics[k]-base_result[k] for k in METRICS},**ranking_change(base_recommendations,recommended))
                        change["groups"]={g:ranking_change(base_recommendations[[groups[u]==g for u in users]],recommended[[groups[u]==g for u in users]]) for g in GROUPS}
                        changes.append(change)
                rows.append({k:result[k] for k in ["model","run_id","training_seed","mode",*METRICS,"mean_recommended_original_norm","mean_recommended_fitting_degree"]})
                print("PASS "+args.phase+" "+mode+" "+run,flush=True)
            if args.phase=="analyze":
                with gzip.open(args.output/(run+"-item-norms.csv.gz"),"wt",newline="") as stream:
                    writer=csv.writer(stream);writer.writerow(["item_id","fitting_degree","original_final_norm"]);writer.writerows(zip(range(n),degree,norm_np))
            records.append(dict(run_id=run,checkpoint_sha256=hashes[run],status="passed"))
        del model,checkpoint,u_vectors,i_vectors,unit;gc.collect();torch.cuda.empty_cache()
    write_csv(args.output/(args.phase+"-per-run.csv"),rows)
    if args.phase=="analyze":(args.output/"ranking-changes.json").write_text(json.dumps(changes,indent=2)+"\n")
    report=dict(phase="verify" if args.phase=="verify" else "analyze",all_passed=len(records)==9,plan_sha256=identity,source_sha256=sources,
        evaluations=records,evaluation_split="validation",test_labels_opened=False,torch_version=torch.__version__,gpu=torch.cuda.get_device_name(0))
    (gate_path if args.phase=="verify" else args.output/"analysis-completion.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":main()
