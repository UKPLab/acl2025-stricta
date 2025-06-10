from nltk import ngrams
import numpy as np
import pandas as pd
import spacy
from dowhy.gcm import StructuralCausalModel, InvertibleStructuralCausalModel
from dowhy.gcm.causal_mechanisms import FunctionalCausalModel, DiscreteAdditiveNoiseModel, StochasticModel, \
    ConditionalStochasticModel, InvertibleFunctionalCausalModel
from dowhy import gcm
from sentence_transformers import SentenceTransformer, util

from ccrw.causal.WorkflowLM import WorkflowLM
from ccrw.workflow.Workflow import Workflow


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


class WorkflowRootModel(StochasticModel):
    def fit(self, X: np.ndarray) -> None:
        """Fits the model according to the data."""
        pass

    def draw_samples(self, num_samples: int) -> np.ndarray:
        """Draws samples for the fitted model."""
        pass

    def clone(self):
        return WorkflowRootModel()


class WorkflowItemStructuralEquation(InvertibleFunctionalCausalModel, ConditionalStochasticModel):
    def __init__(self, workflow, item_id, lm, semantic_encoder=None, nlp=None):
        self.workflow = workflow
        self.item_id = item_id

        self.item = self.workflow[self.item_id]
        self.parents = workflow.parents(item_id)

        self.lm = lm
        if semantic_encoder is None:
            self.sem = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
        else:
            self.sem = semantic_encoder

        if nlp is None:
            self.nlp = spacy.load('en_core_sci_lg')
        else:
            self.nlp = nlp

        self.sem_centroids = None
        self.syn_knns = None
        self.lex_knns = None

        self.sem_noise_estimator = None
        self.syn_noise_estimator = None
        self.lex_noise_estimator = None

    def fit(self, X, Y):
        # ASSUMPTION: X (k, 1) is provided as: a dict including the paper id and the text in out
        #             Y (k, ) is provided as: a dict including the text in out

        # computing theta and epsilon estimators from data
        processed = [self.nlp(r["out"]) for r in Y]
        paper_ids = [r["id"].split(".")[0] for r in Y]

        groups = {}
        for i, pid in enumerate(paper_ids):
            groups[pid] = groups.get(pid, []) + [i]

        Y_t = np.array([y["out"] for y in Y])

        # 1) determine the centroid of meaning vectors of answers
        Y_sem = np.apply_along_axis(lambda x: self.sem.encode(x), 0, Y_t).transpose()
        self.sem_centroids = {
            pid: np.mean(Y_sem[groups[pid]], axis=0) for pid in groups
        }

        # 2) define noise estimator (i.e. penalty function) for generation proximity to meaning centroid
        def sem_estimator(samples):
            samples_t = np.array(([x["out"] for x in samples]))
            papers = list(set([r["id"].split(".")[0] for r in Y]))

            assert len(papers) == 1, f"expected samples from exactly one paper; found {papers}"

            sem_rep = np.apply_along_axis(lambda x: self.sem.encode(x["out"]), 0, samples_t).transpose()

            return np.apply_along_axis(lambda x: float(util.cos_sim(x, self.sem_centroids[papers[0]])),
                                       0,
                                       sem_rep)[np.newaxis].transpose()

        self.sem_noise_estimator = sem_estimator

        # 3) determine syntactic tau + kNN distance
        self.syn_knns = {
            pid: [[t.pos for t in processed[i]] for i in groups[pid]]
            for pid in groups
        }

        def syn_estimator(samples):
            samples_t = np.array(([x["out"] for x in samples]))
            papers = list(set([r["id"].split(".")[0] for r in Y]))

            assert len(papers) == 1, f"expected samples from exactly one paper; found {papers}"

            processed_samples = [self.nlp(r) for r in samples_t]
            return [_avg_ngram_overlap([t.pos for t in processed_samples[i]], self.syn_knns[papers[0]]) for i in
                    range(len(processed_samples))]

        self.syn_noise_estimator = syn_estimator

        # 4) determine lexical tau + kNN penalty function for lexical alignment
        self.lex_knns = {
            pid: [[t.text.lower()  for t in processed[i]] for i in groups[pid]]
            for pid in groups
        }

        def lex_estimator(samples):
            samples_t = np.array(([x["out"] for x in samples]))
            papers = list(set([r["id"].split(".")[0] for r in Y]))

            assert len(papers) == 1, f"expected samples from exactly one paper; found {papers}"

            processed_samples = [self.nlp(r) for r in samples_t]
            return [_avg_ngram_overlap([t.text.lower() for t in processed_samples[i]], self.lex_knns[papers[0]]) for i in
                    range(len(processed_samples))]

        self.lex_noise_estimator = lex_estimator

        # 5) configure lm through controlled text generation
        self.lm.add_penalties(semantic=self.sem_noise_estimator,
                              syntactic=self.syn_noise_estimator,
                              lexical=self.lex_noise_estimator)
        self.lm.add_examples(X, Y)

    def estimate_noise(self, target_samples: np.ndarray, parent_samples: np.ndarray) -> np.ndarray:
        def _estimate_epsilon(x):
            pass

        def _estimate_theta(x):
            pass

        np.apply_along_axis()

    def draw_noise_samples(self, num_samples: int) -> np.ndarray:
        raise NotImplementedError  # todo

    def evaluate(self, parent_samples, noise_samples):
        raise NotImplementedError  # todo

    def clone(self):
        return WorkflowItemStructuralEquation(self.workflow, self.item_id, self.lm)

    @staticmethod
    def to_json(out_path):
        pass  # todo

    @staticmethod
    def from_json(in_path):
        pass  # todo


class WorkflowSCM:
    def __init__(self, workflow, executions, binary_only=False):
        self.workflow = workflow
        self.executions = executions

        self.binary_only = binary_only
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
        if self.binary_only:
            self.actual_workflow = self.workflow.to_binary()
            graph = self.workflow.to_binary().to_graph()
        else:
            self.actual_workflow = self.workflow
            graph = self.workflow.to_graph()

        if self.binary_only:
            self.scm = InvertibleStructuralCausalModel(graph=graph)
        else:
            self.scm = StructuralCausalModel(graph=graph)

        if self.binary_only:
            for wi in graph:
                if len(list(graph.predecessors(wi))) == 0:  ## root node
                    mechanism = gcm.EmpiricalDistribution()
                else:
                    mechanism = DiscreteAdditiveNoiseModel(
                        gcm.ml.create_gaussian_process_classifier()
                    )
                self.scm.set_causal_mechanism(target_node=wi, mechanism=mechanism)

        else:
            nlp = spacy.load('en_core_sci_lg')
            sem_encoder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
            lm = WorkflowLM()

            for wi in graph:
                if len(list(graph.predecessors(wi))) == 0:
                    mechanism = WorkflowRootModel()
                else:
                    mechanism = WorkflowItemStructuralEquation(workflow,
                                                               wi,
                                                               lm,
                                                               semantic_encoder=sem_encoder,
                                                               nlp=nlp)
                self.scm.set_causal_mechanism(node=wi, mechanism=mechanism)

    def _fit_binary_only(self):
        items = [n for n in self.scm.graph]

        data = []
        for pid, aid_to_ex in self.executions.items():
            for aid, we in aid_to_ex.items():
                data += [{
                    i: we[i]["answer"].lower() == "yes"
                    for i in items
                }]

        gcm.fit(self.scm, pd.DataFrame.from_records(data))

    def _fit_text(self):
        items = [n for n in self.scm.graph]

        data = []
        for pid, aid_to_ex in self.executions.items():
            for aid, we in aid_to_ex.items():
                inp = {
                    i: we.get_text(i)
                    for i in items
                }
                data += [inp]

        gcm.fit(self.scm, pd.DataFrame.from_records(data))

    def fit(self):
        if self.binary_only:
            return self._fit_binary_only()
        else:
            return self._fit_text()

    def _eval_binary_only(self, data=None):
        items = [n for n in self.scm.graph]

        data_df = []
        for pid, aid_to_ex in data.items():
            for aid, we in aid_to_ex.items():
                data_df += [{
                    i: we[i]["answer"] == "Yes"
                    for i in items
                }]

        return gcm.evaluate_causal_model(self.scm, pd.DataFrame.from_records(data_df))

    def evaluate_fitted_model(self, data=None):
        if data is None:
            data = self.executions

        if self.binary_only:
            return self._eval_binary_only(data)

    def arrow_strength(self, item_id):
        # currently assuming only boolean scm
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
