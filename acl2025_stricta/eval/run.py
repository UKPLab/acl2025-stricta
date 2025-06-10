from ..models.AutoWorkflow import AutoWorkflow

from tqdm import tqdm
import os
import pandas as pd


def store_predictions(predictions, fpath):
    df = pd.DataFrame.from_records({"uid": k, "prediction": v} for k,v in predictions.items())
    df.to_csv(fpath, sep=";", index=False)


def load_predictions(fpath):
    df = pd.read_csv(fpath, sep=";", keep_default_na=False)

    predictions = dict((r["uid"], r["prediction"]) for r in df.to_records())

    res = {}
    for uid, pred in predictions.items():
        try:
            p = eval(pred)
        except SyntaxError:
            print("Error during parsing of prediction:", pred, "for item ", uid)
            p = None
        except ValueError:
            print("Decoding error during parsing of prediction", pred, "for item", uid)
            p = None

        res[uid] = p

    return res


def predict_itemwise(workflow, papers, itemwise_answers, model: AutoWorkflow, item_id_filter=None, paper_id_filter=None, binary=False, figures_only=False, _cache_dir=None, cached_answers=None):
    wf = workflow.to_binary() if binary else workflow
    answers = {uid: answers for uid, answers in itemwise_answers.items() if uid.split(".")[1] in wf}

    def _get_inputs(raw):
        if raw["input"] is not None:
            return raw["input"]
        else:
            return []

    def _filter_item(item):
        iid = item.split(".")[1]

        if workflow[iid]["action"] == "read":  # always skip read nodes for prediction, as there is nothing to predict
            return False

        if item_id_filter:
            return item_id_filter(iid)
        else:
            return True

    assert _cache_dir is None or os.path.exists(_cache_dir), "provided cache dir does not exist: " + _cache_dir

    predictions = {}
    selected_papers = list(filter(paper_id_filter, papers.keys())) if paper_id_filter else papers

    if cached_answers:
        print(f"LOADING {len(cached_answers)} CACHED ANSWERS AS PREDICTIONS")
        predictions.update(cached_answers)

    for pid in tqdm(selected_papers, desc=f"Predicting over papers"):
        print("Gathering answers")
        paper_answers = [uid for uid, a in answers.items() if uid.split(".")[0] == pid]
        rel_answers = [uid for uid in paper_answers if _filter_item(uid)]

        if figures_only:
            rel_answers = [uid for uid in paper_answers if workflow[uid.split(".")[1]]["multimodal"]]

        print(f"Identified relevant nodes = {len(rel_answers)}")

        # filter if already given by cached answers
        rel_answers = [uid for uid in rel_answers if uid not in predictions]

        print(f"Identified relevant nodes (after excluding already predicted ones = {len(rel_answers)}")

        if len(rel_answers) == 0:
            print(f"No steps to predict for paper {pid}! Skipping it")
            continue

        # extract nodes should be evaluated only once based on the respective paper text to save compute
        extract_answers = [iid for iid in rel_answers if workflow[iid.split(".")[1]]["action"] == "extract"]
        extract_representatives = {uid.split(".")[1]:uid for uid in extract_answers}  # one example answer per item
        extract_inputs = [[(pa["id"], pa) for pa in _get_inputs(answers[extract_representatives[uid]])] for uid in extract_representatives]
        extract_inputs = [model.preprocess_inputs(papers[pid], [x[0] for x in i], [x[1] for x in i]) for i in extract_inputs]

        # inference nodes need to be computed individually for each annotator and item
        infer_answers = [iid for iid in rel_answers if iid not in extract_answers]
        infer_inputs = [[(pa["id"], pa) for pa in _get_inputs(answers[uid])] for uid in infer_answers]
        infer_inputs = [model.preprocess_inputs(papers[pid], [x[0] for x in i], [x[1] for x in i]) for i in infer_inputs]

        # extract nodes
        predex = model.batch_generate_responses(papers[pid], list(extract_representatives.keys()), extract_inputs)
        predex = model.postprocess_outputs(papers[pid], list(extract_representatives.keys()), predex)
        predex_per_iid = dict(zip(list(extract_representatives.keys()), predex))
        predex_per_answer_id = {ea: predex_per_iid[ea.split(".")[1]] for ea in extract_answers}

        # infer nodes
        predinf = model.batch_generate_responses(papers[pid], [iid.split(".")[1] for iid in infer_answers], infer_inputs)
        predinf = model.postprocess_outputs(papers[pid], [iid.split(".")[1] for iid in infer_answers], predinf)

        predictions.update(dict(zip(infer_answers, predinf)))
        predictions.update(predex_per_answer_id)

        if _cache_dir:
            with open(_cache_dir + "/_predict_cache.txt", "w+") as f:
                f.write(str(predictions))

    return predictions


def pred_io(input_paths, output_path, model: AutoWorkflow, binary=False, paper_filter=None, item_filter=None, out_prefix=None, cached_answers=None):
    # validate
    assert os.path.exists(output_path), "the provided output path does not exist."

    # load
    workflow, papers, answers = load_itemwise_dataset(input_paths)

    predictions = predict_itemwise(workflow, papers, answers, model=model, binary=binary, paper_id_filter=paper_filter, item_id_filter=item_filter, cached_answers=cached_answers, _cache_dir=output_path)
    ending = f"/{(out_prefix if out_prefix else '')}predictions"
    if binary:
        ending += "_binary"
    ending += ".csv"

    store_predictions(predictions, output_path + ending)

    with open(output_path + "DONE.txt", "w+") as f:
        f.write("Completed run")

    return output_path + ending

