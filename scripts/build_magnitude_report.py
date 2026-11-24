"""Audit validation scoring arms and summarize paired ranking sensitivity."""
import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import statistics
import tarfile
from collections import defaultdict
from pathlib import Path
from build_extended_test_report import METRICS, exposure_check, read_csv, write_csv
from supplemental_test_gate import assert_reconciles, plan_digest, require_gate, source_digest

MODELS=["MF","LightGCN K=3","LightGCN-single K=3"]
RANKING=["mean_top20_overlap","mean_jaccard","fraction_top1_changed","fraction_sets_changed","fraction_ordered_lists_changed","mean_positional_agreement"]


def aggregate(rows,keys,metrics):
    buckets=defaultdict(list)
    for row in rows:buckets[tuple(row[k] for k in keys)].append(row)
    result=[]
    for identity,members in buckets.items():
        assert sorted(int(r["training_seed"]) for r in members)==[2020,2021,2022]
        for metric in metrics:
            values=[float(r[metric]) for r in members]
            result.append(dict(zip(keys,identity),metric=metric,seed_count=3,mean=statistics.mean(values),sample_sd=statistics.stdev(values)))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results",type=Path);parser.add_argument("--repository",type=Path,default=Path(__file__).resolve().parents[1]);parser.add_argument("--check",action="store_true")
    args=parser.parse_args();root,repo=args.results,args.repository
    plan=json.loads((repo/"provenance/magnitude-plan.json").read_text());identity=plan_digest(plan)
    gate=json.loads((root/"validation-verification.json").read_text());completion=json.loads((root/"analysis-completion.json").read_text())
    hashes={e["run_id"]:e["checkpoint_sha256"] for e in plan["evaluations"]}
    require_gate(gate,identity,plan["source_sha256"],hashes)
    assert completion["phase"]=="analyze" and completion["evaluation_split"]=="validation" and not completion["test_labels_opened"]
    require_gate(dict(completion,phase="verify"),identity,plan["source_sha256"],hashes)
    profiles={r["group"]:r for r in read_csv(root/"group-profiles.csv")};assert len(profiles)==3
    assert sum(int(r["evaluation_users"]) for r in profiles.values())==29858
    assert sum(int(r["tail_evaluation_users"]) for r in profiles.values())==18987
    popularity=read_csv(repo/"results/gowalla/item-popularity.csv");tail={int(r["item_id"]) for r in popularity if r["group"]=="tail"};n=plan["catalog_size"]
    per_run,group_rows,records=[],[],{}
    references=[]
    for entry in plan["evaluations"]:
        run=entry["run_id"]
        with gzip.open(root/(run+"-item-norms.csv.gz"),"rt") as stream:norm_rows=list(csv.DictReader(stream))
        assert [int(r["item_id"]) for r in norm_rows]==list(range(n))
        assert [r["fitting_degree"] for r in norm_rows]==[r["fitting_degree"] for r in popularity]
        norms=[float(r["original_final_norm"]) for r in norm_rows]
        assert all(math.isfinite(v) and v>=0 for v in norms)
        for phase,modes in [("verify",["legacy_sigmoid"]),("analyze",plan["scoring_modes"])]:
            for mode in modes:
                folder=root/phase/run/mode;measured=json.loads((folder/"metrics.json").read_text())
                assert measured["plan_sha256"]==identity and measured["source_sha256"]==plan["source_sha256"]
                assert measured["checkpoint_sha256"]==entry["checkpoint_sha256"] and measured["checkpoint_epoch"]==entry["checkpoint_epoch"]
                assert measured["evaluation_split"]=="validation" and not measured["test_labels_opened"]
                assert measured["training_seed"]==entry["training_seed"] and measured["model"]==entry["model"] and measured["mode"]==mode
                assert_reconciles(measured,measured["grouped"])
                for group,value in measured["grouped"].items():
                    assert value["evaluation_users"]==int(profiles[group]["evaluation_users"])
                    assert value["tail_evaluation_users"]==int(profiles[group]["tail_evaluation_users"])
                counts=exposure_check(folder/"exposure.csv",measured,tail,n)
                if mode=="legacy_sigmoid":
                    old=read_csv(repo/entry["validation_exposure"])
                    assert len(old)==n
                    difference=sum(abs(c-int(r["recommendation_count"])) for c,r in zip(counts,old))
                    assert difference==measured["reference_exposure_l1"]<=plan["exposure_l1_tolerance"]
                    delta={k:measured[k]-entry["validation_metrics"][k] for k in METRICS}
                    assert all(abs(v)<=plan["metric_absolute_tolerance"] for v in delta.values())
                    references.append(dict(run_id=run,phase=phase,exposure_l1=difference,metric_deltas=delta))
                if phase=="analyze":
                    assert measured["zero_item_vectors"]==sum(v==0 for v in norms)
                    assert measured["sub_epsilon_item_vectors"]==sum(0<v<plan["epsilon"] for v in norms)
                    assert math.isclose(measured["mean_recommended_original_norm"],sum(v*c for v,c in zip(norms,counts))/sum(counts),rel_tol=1e-6,abs_tol=1e-6)
                    assert math.isclose(measured["mean_recommended_fitting_degree"],sum(int(p["fitting_degree"])*c for p,c in zip(popularity,counts))/sum(counts),abs_tol=1e-8)
                    records[run,mode]=measured
                    per_run.append({k:measured[k] for k in ["model","run_id","training_seed","mode",*METRICS,"mean_recommended_original_norm","mean_recommended_fitting_degree","zero_item_vectors","sub_epsilon_item_vectors","zero_user_vectors"]})
                    group_rows.extend(dict(model=entry["model"],run_id=run,training_seed=entry["training_seed"],mode=mode,group=g,**v) for g,v in measured["grouped"].items())
    changes=json.loads((root/"ranking-changes.json").read_text());assert len(changes)==18
    assert {(r["run_id"],r["comparison"]) for r in changes}=={(e["run_id"],m+"_vs_raw_dot") for e in plan["evaluations"] for m in ["unit_item","legacy_sigmoid"]}
    effects=[]
    for change in changes:
        run,comparison=change["run_id"],change["comparison"];mode=comparison.removesuffix("_vs_raw_dot")
        original,altered=records[run,"raw_dot"],records[run,mode]
        assert change["model"]==original["model"] and change["training_seed"]==original["training_seed"]
        for metric in METRICS:assert abs(change["metric_changes"][metric]-(altered[metric]-original[metric]))<=1e-12
        assert change["fraction_sets_changed"]<=change["fraction_ordered_lists_changed"]+1e-12
        assert change["fraction_top1_changed"]<=change["fraction_ordered_lists_changed"]+1e-12
        for metric in RANKING:
            assert 0<=change[metric]<=1
            pooled=sum(change["groups"][g][metric]*int(p["evaluation_users"]) for g,p in profiles.items())/29858
            assert abs(pooled-change[metric])<=1e-12
        assert abs(change["fraction_sets_changed"]*29858-round(change["fraction_sets_changed"]*29858))<1e-6
        effects.append(dict(model=change["model"],run_id=run,training_seed=change["training_seed"],comparison=comparison,
                            **{k:change[k] for k in RANKING},**change["metric_changes"]))
    arms=aggregate(per_run,["model","mode"],METRICS+["mean_recommended_original_norm","mean_recommended_fitting_degree"])
    paired=aggregate(effects,["model","comparison"],METRICS+RANKING)
    groups=aggregate(group_rows,["model","mode","group"],METRICS)
    datasets={"comparison-per-run.csv":per_run,"comparison-summary.csv":arms,"paired-effects.csv":effects,"paired-summary.csv":paired,"user-groups-per-run.csv":group_rows,"user-groups-summary.csv":groups}
    if args.check:
        for name,expected in datasets.items():
            actual=read_csv(root/name);assert len(actual)==len(expected)
            for observed,reference in zip(actual,expected):
                assert set(observed)==set(reference)
                for key,value in reference.items():
                    if isinstance(value,(float,int)):assert math.isclose(float(observed[key]),value,abs_tol=1e-12)
                    else:assert observed[key]==value
        with tarfile.open(root/"inference-sources.tar.gz") as archive:
            for name,wanted in plan["source_sha256"].items():assert hashlib.sha256(archive.extractfile("scripts/"+name).read().replace(b"\r\n",b"\n")).hexdigest()==wanted
            assert plan_digest(json.loads(archive.extractfile("provenance/magnitude-plan.json").read()))==identity
        print("PASS: nine paired models, all exposure/group/seed summaries and source snapshot verified")
        return
    for name,values in datasets.items():write_csv(root/name,values)
    idx={(r["model"],r["mode"],r["metric"]):r for r in arms}
    lines=["# Item magnitude intervention: validation results","","Three training seeds, fixed existing checkpoints. Primary comparison: raw dot product versus unit-length item vectors with the same user vectors. Sigmoid is a separate numerical control, not the intervention.","",
           "| Model | Scoring | Recall@20 | NDCG@20 | Tail Recall@20 | Tail share (%) | Coverage (%) | Gini |",
           "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for model in MODELS:
        for mode,label in [("raw_dot","Raw dot baseline"),("unit_item","Equal-length items"),("legacy_sigmoid","Legacy sigmoid check")]:
            cells=[]
            for metric in ["recall_at_20","ndcg_at_20","tail_recall_at_20","tail_share_at_20","coverage_at_20","exposure_gini"]:
                row=idx[model,mode,metric];scale=100 if metric in ["tail_share_at_20","coverage_at_20"] else 1
                cells.append(f"{scale*row['mean']:.5f} +/- {scale*row['sample_sd']:.5f}")
            lines.append("| "+model+" | "+label+" | "+" | ".join(cells)+" |")
    (root/"comparison.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    eidx={(r["model"],r["comparison"],r["metric"]):r for r in paired}
    lines=["# Paired changes from removing item magnitude","","Mean +/- sample SD of within-checkpoint changes across three training seeds. Recall, tail recall and tail-share changes are percentage points. Top-20 overlap is the percentage of original recommended items retained, averaged over validation users. Group-level ranking changes are in ranking-changes.json.","",
           "| Model | Recall change | Tail recall change | Tail-share change | Top-20 retained (%) | Top-1 changed (%) |",
           "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for model in MODELS:
        cells=[]
        for metric in ["recall_at_20","tail_recall_at_20","tail_share_at_20","mean_top20_overlap","fraction_top1_changed"]:
            row=eidx[model,"unit_item_vs_raw_dot",metric];cells.append(f"{100*row['mean']:.3f} +/- {100*row['sample_sd']:.3f}")
        lines.append("| "+model+" | "+" | ".join(cells)+" |")
    (root/"paired-changes.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    with tarfile.open(root/"inference-sources.tar.gz","w:gz") as archive:
        for name in ["scripts/"+n for n in plan["source_sha256"]]+["provenance/magnitude-plan.json"]:
            if name.startswith("scripts/"):assert source_digest(repo/name)==plan["source_sha256"][Path(name).name]
            raw=(repo/name).read_bytes().replace(b"\r\n",b"\n");info=tarfile.TarInfo(name);info.size=len(raw);archive.addfile(info,io.BytesIO(raw))
    (root/"artifact-audit.json").write_text(json.dumps(dict(all_passed=True,checkpoints=9,original_verification_evaluations=9,analysis_scoring_evaluations=27,
        primary_pairs=9,grouped_rows=81,plan_sha256=identity,numerical_reference_checks=references,
        limits="Relevance and ranking-overlap measurements come from inference. Aggregate audits verify provenance, exposed item metrics, group pooling and seed summaries; individual user rankings are not published."),indent=2)+"\n")
    print("PASS: nine legacy gates, 27 scoring evaluations, nine primary pairs, exposures and grouped seed summaries")


if __name__=="__main__":main()
