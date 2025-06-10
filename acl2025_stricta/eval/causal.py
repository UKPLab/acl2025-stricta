from ..data.load import load
from ..base.WorkflowSCM import WorkflowSCM


def fit_boolean_scm(dataset_paths):
    assert len(dataset_paths) == 2, "expected student and main study dataset paths"

    workflow, papers, annotations = load(dataset_paths[0])
    workflow2, papers2, annotations2 = load(dataset_paths[1])
    annotations.update(annotations2)
    papers.update(papers2)

    scm = WorkflowSCM(workflow, annotations)
    scm.fit()

    return scm


def ate(scm, intervention:dict[str, Callable], not_intervention:dict[str, Callable]):
    for k in intervention.keys():
        assert k in scm.actual_workflow, f"intervention key {k} not in workflow"

    for k in not_intervention.keys():
        assert k in scm.actual_workflow, f"not_intervention key {k} not in workflow"

    res = scm.average_causal_effect("step47_x", intervention, not_intervention)

    return res


def counterfactual(scm, sample:dict, intervention:dict[str, Callable], target_node:str= "step47_x"):
    for k in intervention.keys():
        assert k in scm.actual_workflow, f"intervention key {k} not in workflow"

    ans = {wi: sample[wi]["answer"] == "Yes" for wi in scm.actual_workflow}
    res = scm.counterfactual(dict(intervention), ans)

    return res[target_node]