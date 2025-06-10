from nltk import ngrams
import numpy as np
import pandas as pd
from dowhy.gcm import InvertibleStructuralCausalModel
from dowhy.gcm.causal_mechanisms import DiscreteAdditiveNoiseModel
from dowhy import gcm

from .Workflow import Workflow


def _avg_ngram_overlap(text, reference_texts, n=2):
    if len(text) == 0:
        return 0

    ngrams_t = set(ngrams(text, n))
    if len(ngrams_t) == 0:
        return 0

    overlap = []
    for i, tr in enumerate(reference_texts):
        ngrams_r = set(ngrams(tr, n))

        if len(tr) == 0 or len(ngrams_r) == 0:
            overlap += [0]
        else:
            overlap += [len(ngrams_t.intersection(ngrams_r)) / (len(ngrams_t.union(ngrams_r)))]

    return np.mean(overlap)


class BooleanDecisionSCM:
    def __init__(self, workflow, executions):
        self.workflow = workflow
        self.executions = executions

        self.actual_workflow = None
        self.scm = None

        self._setup()

    @staticmethod
    def _reduce_workflow_to_binary_nodes(workflow):
        bin_vars = [wi for wi in workflow if "format" in workflow[wi] and workflow[wi]["format"] == "truth"]

        current_graph = workflow.to_graph()
        non_bin_vars = [wi for wi in current_graph if wi not in bin_vars]
        for nb in non_bin_vars:
            parents, children = list(current_graph.predecessors(nb)), list(current_graph.successors(nb))

            if len(parents) > 0 and len(children) > 0:
                # replace node by an "edge", i.e. connecting all children and parents
                for p in parents:
                    for c in children:
                        current_graph.add_edge(p, c)

            current_graph.remove_node(nb)

        return Workflow._from_graph(workflow.name, current_graph)

    def _setup(self):
        self.actual_workflow = self.workflow.to_binary()
        graph = self.workflow.to_binary().to_graph()

        self.scm = InvertibleStructuralCausalModel(graph=graph)
        for wi in graph:
            if len(list(graph.predecessors(wi))) == 0:  ## root node
                mechanism = gcm.EmpiricalDistribution()
            else:
                mechanism = DiscreteAdditiveNoiseModel(
                    gcm.ml.create_gaussian_process_classifier()
                )
            self.scm.set_causal_mechanism(target_node=wi, mechanism=mechanism)

    def fit(self):
        items = [n for n in self.scm.graph]

        data = []
        for pid, aid_to_ex in self.executions.items():
            for aid, we in aid_to_ex.items():
                data += [{
                    i: we[i]["answer"].lower() == "yes"
                    for i in items
                }]

        gcm.fit(self.scm, pd.DataFrame.from_records(data))

    def evaluate_fitted_model(self, data=None):
        items = [n for n in self.scm.graph]

        data_df = []
        for pid, aid_to_ex in data.items():
            for aid, we in aid_to_ex.items():
                data_df += [{
                    i: we[i]["answer"] == "Yes"
                    for i in items
                }]

        return gcm.evaluate_causal_model(self.scm, pd.DataFrame.from_records(data_df))

    def arrow_strength(self, item_id):
        return gcm.arrow_strength(self.scm, item_id)

    def average_causal_effect(self, item_id, intervention, intervention_reference):
        return gcm.average_causal_effect(
            self.scm,
            target_node=item_id,
            interventions_alternative=intervention,
            interventions_reference=intervention_reference,
            num_samples_to_draw=200
        )

    def counterfactual(self, intervention, context):
        return gcm.counterfactual_samples(
            self.scm,
            interventions=intervention,
            observed_data=pd.DataFrame(data={k: [v] for k, v in context.items()})
        )
