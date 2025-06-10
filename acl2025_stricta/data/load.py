from ..base.workflow import Workflow
from intertext_graph import IntertextDocument
from ..base.WorkflowExecution import WorkflowExecution


def load_itemwise_dataset(paths:str|list[str], paper_ids:list[str]|None=None, annotator_ids:list[str]|None=None, join:bool=True):
    if type(paths) == str:
        paths = [paths]

    dataset = []
    for dir in paths:
        wf = Workflow.from_json(dir + "/workflow.json")

        dataset_prefix = os.path.basename(dir[:-len("/dataset")])

        papers = {}
        annotations = {}
        for dn, dfp in iterate_dirs(dir, prefix="paper_"):
            if paper_ids is not None and dn not in paper_ids:
                continue

            with open(dfp + "/paper.itg.json", "r") as f:
                papers[dn] = IntertextDocument.load_json(f)

            with open(dfp + "/meta.json", "r") as f:
                papers[dn].meta.update(json.load(f))

            papers[dn].meta["pid"] = dataset_prefix + "_" + dn

            with open(dfp + "/annotations_in_out.json", "r") as f:
                in_out = json.load(f)

            annotations.update({
                dataset_prefix + "_" + i["global_id"] + "." + str(aid) : i["answers"][aid] for i in in_out for aid in i["answers"]
            })

        if annotator_ids is not None:
            def _surpress_item(item_id):
                pid, step_id, aid = tuple(item_id.split("."))
                return aid not in annotator_ids

            to_remove = [iid for iid in annotations if _surpress_item(iid)]
            for iid in to_remove:
                del annotations[iid]

        dataset += [(wf, papers, annotations, dataset_prefix)]

    if join:
        # assuming the workflows are compatible of the multiple datasets
        papers = {d[-1] + "_" + k: v for d in dataset for k,v in d[1].items()}
        workflow = dataset[0][0]
        annotations = {a[0]:a[1] for d in dataset for a in d[2].items()}

        dataset = workflow, papers, annotations

    return dataset


def load(dir):
    wf = Workflow.from_json(dir + "/workflow.json")

    papers = {}
    annotations = {}
    for dn, dfp in iterate_dirs(dir, prefix="paper_"):
        with open(dfp + "/paper.itg.json", "r") as f:
            papers[dn] = IntertextDocument.load_json(f)

        with open(dfp + "/meta.json", "r") as f:
            papers[dn].meta.update(json.load(f))

        annotations[dn] = WorkflowExecution.from_json(dfp + "/annotations_in_out.json", wf, papers[dn])

    return wf, papers, annotations
