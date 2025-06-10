import os
import json
import numpy as np
import pandas as pd
from intertext_graph import SpanNode
from tqdm import tqdm

from ..utils import struc_answer_to_text


def eval_io(input_paths, pred_paths, metrics, output_path=None, binary=False, verbose=True):
    # validate
    assert output_path is None or os.path.exists(output_path), "the provided output path does not exist."

    if type(pred_paths) == str:
        pred_paths = [pred_paths]

    for pp in pred_paths:
        assert os.path.exists(pp), f"the provided prediction file {pp} does not exist."

    # load
    workflow, papers, answers = load_itemwise_dataset(input_paths)
    predictions = [load_predictions(p) for p in pred_paths]

    aggregated = {}
    samplewise = []
    paperwise = []
    itemwise = []

    eval_name = "_".join([m[0] for m in metrics])

    for ix, pred in tqdm(enumerate(predictions), desc="iterating over predictions"):
        samplewise += [{}]
        paperwise += [{}]
        itemwise += [{}]

        gold = {uid: a["output"] for uid, a in answers.items() if uid in pred}
        if binary:
            gold = {uid: a["answer"] == "Yes" for uid, a in gold.items()}
        else:
            gold = {uid: struc_answer_to_text(a) for uid, a in gold.items()}

        if binary:
            pred = {uid: a is not None and type(a) != str and a["answer"] == "Yes" for uid, a in pred.items()}  # parsing maybe failed; take care of None values
        else:
            pred = {uid: (struc_answer_to_text(a) if a is not None and a != "" else "None") for uid, a in pred.items()}

        instructions = {wi: f'{workflow[wi]["prompt"]} This means: {workflow[wi]["description"]}' for wi in workflow}

        for metric in metrics:
            mn, m  = metric[0], metric[1]
            if len(metric) == 3 and metric[2]:  # flag for loading instructions is set to True
                iids = [uid.split(".")[1] for uid in gold]
                uid_instructions = {uid:instructions[iid] for uid, iid in zip(gold.keys(), iids)}

                scored = m(gold, pred, instructions=uid_instructions)
            else:
                scored = m(gold, pred)

            if type(scored) == dict and all(g in scored for g in gold):
                samplewise[ix][mn] = scored

                per_paper = {pid: [v for k, v in scored.items() if k.split(".")[0] == pid] for pid in
                             [uid.split(".")[0] for uid in scored]}
                paperwise[ix][mn] = {pid: (np.mean(scores), np.std(scores)) for pid, scores in per_paper.items()}

                per_item = {iid: [v for k, v in scored.items() if k.split(".")[1] == iid] for iid in
                            [uid.split(".")[1] for uid in scored]}
                itemwise[ix][mn] = {iid: (np.mean(scores), np.std(scores)) for iid, scores in per_item.items()}

                aggregated["mean_" + mn] = aggregated.get("mean_" + mn, []) + [np.mean(list(scored.values()))]
                aggregated["std_" + mn] = aggregated.get("std_" + mn, []) + [np.std(list(scored.values()))]
            else:
                aggregated[mn] = aggregated.get(mn, []) + [scored]

    if len(predictions) > 1:
        added = {}
        for mn in aggregated:
            added["mean_" + mn] = [np.mean(aggregated[mn])]
            added["std_" + mn] = [np.std(aggregated[mn])]

        aggregated.update(added)

    if verbose:
        print(f"EVALUATION REPORT on {pred_paths}")
        print(f"METRICS: {[m[0] for m in metrics]}")

        for key, value in aggregated.items():
            print(f"** {key}:")
            if type(value) == list and len(value) > 1:
                print("\n".join([str(v) for v in value]))
            elif type(value == list):
                print(value[0])
            else:
                print(value)
            print("-" * 20)

    if output_path:
        if len(samplewise) > 0:
            # save samplewise output
            records = []
            metrics = list(samplewise[0].keys())
            if len(metrics) > 0:
                sample_ids = set(iid for ix in range(len(samplewise)) for iid in samplewise[ix][metrics[0]].keys())
                for uid in sample_ids:
                    records += [dict(uid=uid, **{
                        f"run_{ix + 1}_{m}": (samplewise[ix][m][uid] if uid in samplewise[ix][m] else np.nan) for ix in
                        range(len(samplewise)) for m in metrics})]

                pd.DataFrame.from_records(records).to_csv(
                    output_path + f"/evaluation_per_sample_{eval_name}{'_binary' if binary else ''}.csv", sep=";",
                    index=False)

        if len(paperwise) > 0:
            records = []
            metrics = list(paperwise[0].keys())
            if len(metrics) > 0:
                pids = set(iid for ix in range(len(paperwise)) for iid in paperwise[ix][metrics[0]].keys())
                for pid in pids:
                    records += [dict(uid=pid, **{
                        f"run_{ix + 1}_{m}": (paperwise[ix][m][pid] if pid in paperwise[ix][m] else np.nan) for ix in
                        range(len(paperwise)) for m in metrics})]

                pd.DataFrame.from_records(records).to_csv(
                    output_path + f"/evaluation_per_paper_{eval_name}{'_binary' if binary else ''}.csv", sep=";",
                    index=False)

        if len(itemwise) > 0:
            records = []
            metrics = list(itemwise[0].keys())
            if len(metrics) > 0:
                iids = set(iid for ix in range(len(itemwise)) for iid in itemwise[ix][metrics[0]].keys())
                for iid in iids:
                    records += [dict(uid=iid, **{
                        f"run_{ix + 1}_{m}": (itemwise[ix][m][iid] if iid in itemwise[ix][m] else np.nan) for ix in
                        range(len(itemwise)) for m in metrics})]

                pd.DataFrame.from_records(records).to_csv(
                    output_path + f"/evaluation_per_item_{eval_name}{'_binary' if binary else ''}.csv", sep=";",
                    index=False)

        pd.DataFrame.from_dict(
            {"metric": [k for k in aggregated], "value": [v[0] for k, v in aggregated.items()]}).to_csv(
            output_path + f"/evaluation_aggregated_{eval_name}{'_binary' if binary else ''}.csv", sep=";", index=False)

    return aggregated, samplewise


def eval_extract_against_text(input_paths, pred_paths, item_ids, metrics, output_path=None, binary=False, verbose=True):
    # validate
    assert output_path is None or os.path.exists(output_path), "the provided output path does not exist."

    if type(pred_paths) == str:
        pred_paths = [pred_paths]

    for pp in pred_paths:
        assert os.path.exists(pp), f"the provided prediction file {pp} does not exist."

    # load
    workflow, papers, answers = load_itemwise_dataset(input_paths)
    predictions = [load_predictions(p) for p in pred_paths]

    aggregated = {}
    samplewise = []
    paperwise = []
    itemwise = []

    eval_name = "_".join([m[0] for m in metrics])

    def get_paper_text(uid, answer):
        pid = uid.split(".")[0]
        paper = papers[pid]

        if "spans" not in answer or answer["spans"] is None or len(answer["spans"]) == 0:
            spans = [n.ix for n in paper.nodes if len(n.content) > 300 and not isinstance(n, SpanNode)]# compress paper to main passages
        else:
            spans = answer["spans"]

        res = []
        for s in spans:
            res += [paper.get_node_by_ix(s).content]

        return "\n".join(res)

    for ix, pred in tqdm(enumerate(predictions), desc="iterating over predictions"):
        samplewise += [{}]
        paperwise += [{}]
        itemwise += [{}]

        gold = {uid: a["output"] for uid, a in answers.items() if uid in pred}
        gold = {k:v for k,v in gold.items() if k.split(".")[1] in item_ids}
        gold = {uid: get_paper_text(uid, a) for uid, a in gold.items()}

        pred = {uid: (struc_answer_to_text(a) if a is not None and a != "" else "None") for uid, a in pred.items() if uid in gold.keys()}
        instructions = {wi: f'{workflow[wi]["prompt"]} This means: {workflow[wi]["description"]}' for wi in workflow}

        for metric in metrics:
            mn, m  = metric[0], metric[1]
            mn = mn + "_vs_text"

            if len(metric) == 3 and metric[2]:  # flag for loading instructions is set to True
                iids = [uid.split(".")[1] for uid in gold]
                uid_instructions = {uid:instructions[iid] for uid, iid in zip(gold.keys(), iids)}

                scored = m(gold, pred, instructions=uid_instructions)
            else:
                scored = m(gold, pred)

            if type(scored) == dict and all(g in scored for g in gold):
                samplewise[ix][mn] = scored

                per_paper = {pid: [v for k, v in scored.items() if k.split(".")[0] == pid] for pid in
                             [uid.split(".")[0] for uid in scored]}
                paperwise[ix][mn] = {pid: (np.mean(scores), np.std(scores)) for pid, scores in per_paper.items()}

                per_item = {iid: [v for k, v in scored.items() if k.split(".")[1] == iid] for iid in
                            [uid.split(".")[1] for uid in scored]}
                itemwise[ix][mn] = {iid: (np.mean(scores), np.std(scores)) for iid, scores in per_item.items()}

                aggregated["mean_" + mn] = aggregated.get("mean_" + mn, []) + [np.mean(list(scored.values()))]
                aggregated["std_" + mn] = aggregated.get("std_" + mn, []) + [np.std(list(scored.values()))]
            else:
                aggregated[mn] = aggregated.get(mn, []) + [scored]

    if len(predictions) > 1:
        added = {}
        for mn in aggregated:
            added["mean_" + mn] = [np.mean(aggregated[mn])]
            added["std_" + mn] = [np.std(aggregated[mn])]

        aggregated.update(added)

    if verbose:
        print(f"EVALUATION REPORT on {pred_paths}")
        print(f"METRICS: {[m[0] for m in metrics]}")

        for key, value in aggregated.items():
            print(f"** {key}:")
            if type(value) == list and len(value) > 1:
                print("\n".join([str(v) for v in value]))
            elif type(value == list):
                print(value[0])
            else:
                print(value)
            print("-" * 20)

    if output_path:
        if len(samplewise) > 0:
            # save samplewise output
            records = []
            metrics = list(samplewise[0].keys())
            if len(metrics) > 0:
                sample_ids = set(iid for ix in range(len(samplewise)) for iid in samplewise[ix][metrics[0]].keys())
                for uid in sample_ids:
                    records += [dict(uid=uid, **{
                        f"run_{ix + 1}_{m}": (samplewise[ix][m][uid] if uid in samplewise[ix][m] else np.nan) for ix in
                        range(len(samplewise)) for m in metrics})]

                pd.DataFrame.from_records(records).to_csv(
                    output_path + f"/evaluation_per_sample_{eval_name}{'_binary' if binary else ''}.csv", sep=";",
                    index=False)

        if len(paperwise) > 0:
            records = []
            metrics = list(paperwise[0].keys())
            if len(metrics) > 0:
                pids = set(iid for ix in range(len(paperwise)) for iid in paperwise[ix][metrics[0]].keys())
                for pid in pids:
                    records += [dict(uid=pid, **{
                        f"run_{ix + 1}_{m}": (paperwise[ix][m][pid] if pid in paperwise[ix][m] else np.nan) for ix in
                        range(len(paperwise)) for m in metrics})]

                pd.DataFrame.from_records(records).to_csv(
                    output_path + f"/evaluation_per_paper_{eval_name}{'_binary' if binary else ''}.csv", sep=";",
                    index=False)

        if len(itemwise) > 0:
            records = []
            metrics = list(itemwise[0].keys())
            if len(metrics) > 0:
                iids = set(iid for ix in range(len(itemwise)) for iid in itemwise[ix][metrics[0]].keys())
                for iid in iids:
                    records += [dict(uid=iid, **{
                        f"run_{ix + 1}_{m}": (itemwise[ix][m][iid] if iid in itemwise[ix][m] else np.nan) for ix in
                        range(len(itemwise)) for m in metrics})]

                pd.DataFrame.from_records(records).to_csv(
                    output_path + f"/evaluation_per_item_{eval_name}{'_binary' if binary else ''}.csv", sep=";",
                    index=False)

        pd.DataFrame.from_dict(
            {"metric": [k for k in aggregated], "value": [v[0] for k, v in aggregated.items()]}).to_csv(
            output_path + f"/evaluation_aggregated_{eval_name}{'_binary' if binary else ''}.csv", sep=";", index=False)

    return aggregated, samplewise