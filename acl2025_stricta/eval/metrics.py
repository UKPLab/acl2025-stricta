import gc
import uuid

import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score


def f1(gold, predicted, sample_weights=None):
    # validate
    assert set(gold.keys()) == set(predicted.keys())

    return f1_score([gold[uid] for uid in gold], [predicted[uid] for uid in gold], pos_label=True,
                    sample_weight=[sample_weights[uid] for uid in gold] if sample_weights else None)


def accuracy(gold, predicted, sample_weights=None):
    # validate
    assert set(gold.keys()) == set(predicted.keys())

    return accuracy_score([gold[uid] for uid in gold], [predicted[uid] for uid in gold],
                          sample_weight=[sample_weights[uid] for uid in gold] if sample_weights else None)


def recall(gold, predicted, sample_weights=None):
    # validate
    assert set(gold.keys()) == set(predicted.keys())

    return recall_score([gold[uid] for uid in gold], [predicted[uid] for uid in gold],
                          sample_weight=[sample_weights[uid] for uid in gold] if sample_weights else None)


def precision(gold, predicted, sample_weights=None):
    # validate
    assert set(gold.keys()) == set(predicted.keys())

    return precision_score([gold[uid] for uid in gold], [predicted[uid] for uid in gold],
                          sample_weight=[sample_weights[uid] for uid in gold] if sample_weights else None)


def _paper_weights(gold):
    assert all(g for g in gold if
               type(g) == str and len(g.split(".")) == 3), "not all ids are given in format pid.stepid.annotator_id"

    # determine weights
    samples_per_paper = {}
    for k in gold:
        pid, sid, aid = tuple(k.split("."))
        samples_per_paper[pid] = samples_per_paper.get(pid, 0) + 1
    weights = {g: 1.0 / samples_per_paper[g.split(".")[0]] for g in gold}

    return weights


def f1_paper_weighted(gold, predicted):
    # validate
    assert set(gold.keys()) == set(predicted.keys()), "true and predicted samples should have same indices"

    return f1(gold, predicted, _paper_weights(gold))


def accuracy_paper_weighted(gold, predicted):
    # validate
    assert set(gold.keys()) == set(predicted.keys()), "true and predicted samples should have same indices"

    return accuracy(gold, predicted, _paper_weights(gold))


def recall_paper_weighted(gold, predicted):
    # validate
    assert set(gold.keys()) == set(predicted.keys()), "true and predicted samples should have same indices"

    return recall(gold, predicted, _paper_weights(gold))


def precision_paper_weighted(gold, predicted):
    # validate
    assert set(gold.keys()) == set(predicted.keys()), "true and predicted samples should have same indices"

    return precision(gold, predicted, _paper_weights(gold))


def bert_score_f1(gold, predicted, model_type="distilbert-base-uncased"):
    from evaluate import load

    # validate
    assert set(gold.keys()) == set(predicted.keys())

    # load
    scorer = load("bertscore", experiment_id=str(uuid.uuid4())) # to fix caching issues

    # compute
    scores = scorer.compute(predictions=[predicted[uid] for uid in gold],
                            references=[gold[uid] for uid in gold],
                            lang="en",
                            model_type=model_type)

    return dict(zip(gold.keys(), scores["f1"]))


def meteor_score(gold, predicted):
    from evaluate import load

    # validate
    assert set(gold.keys()) == set(predicted.keys())

    # load
    scorer = load("meteor", experiment_id=str(uuid.uuid4())) # to fix caching issues

    # compute
    scores = scorer.compute(predictions=[predicted[uid] for uid in gold],
                            references=[gold[uid] for uid in gold])

    return scores["meteor"]


def rouge2_score(gold, predicted):
    from evaluate import load

    # validate
    assert set(gold.keys()) == set(predicted.keys())

    # load
    scorer = load("rouge", experiment_id=str(uuid.uuid4())) # to fix caching issues

    # compute
    scores = scorer.compute(predictions=[predicted[uid] for uid in gold],
                            references=[gold[uid] for uid in gold])

    return scores["rouge2"]


def rouge1_score(gold, predicted):
    from evaluate import load

    # validate
    assert set(gold.keys()) == set(predicted.keys())

    # load
    scorer = load("rouge", experiment_id=str(uuid.uuid4())) # to fix caching issues

    # compute
    scores = scorer.compute(predictions=[predicted[uid] for uid in gold],
                            references=[gold[uid] for uid in gold])

    return scores["rouge1"]


def summa_c(gold, predicted):
    from summac.model_summac import SummaCZS

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_zs = SummaCZS(granularity="sentence",
                        model_name="vitc",
                        device=device)

    res = model_zs.score([gold[uid] for uid in gold], [predicted[uid] for uid in gold])

    return dict(zip([uid for uid in gold], [float(f) for f in res["scores"]]))


def true_score(gold, predicted):
    from transformers import pipeline

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pipe = pipeline("text2text-generation", model="google/t5_xxl_true_nli_mixture", device=device)
    to_assess = [f"premise: {g} hypothesis: {p}" for g,p in zip([gold[uid] for uid in gold], [predicted[uid] for uid in gold])]

    res = pipe(to_assess)
    scores = []
    for r in res:
        try:
            s = min(float(r["generated_text"]),1) # ensure max 1 value
            scores += [s]
        except ValueError:
            print(f"WARNING: Failed to parse TRUE score output as float. Setting to 0. Output = ", r["generated_text"])
            scores += [0]

    return dict(zip([uid for uid in gold], scores))