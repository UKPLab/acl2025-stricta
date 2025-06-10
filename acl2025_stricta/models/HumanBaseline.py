import random

from intertext_graph import IntertextDocument

from .AutoWorkflow import AutoWorkflow


class HumanBaseline(AutoWorkflow):
    def __init__(self, workflow, workflow_executions, binary_only=False):
        super().__init__(workflow, "human_baseline", binary_only)

        self.wes = workflow_executions
        self.pid_to_annotators = None
        self.annotator_per_paper = None

        self._setup()

    def _setup(self):
        self.pid_to_annotators = {}

        for pid, aid_to_exec in self.wes.items():
            self.pid_to_annotators[pid] = list(aid_to_exec.keys())

    def select_annotators(self, annotators_per_paper:dict):
        self.annotator_per_paper = annotators_per_paper

    def get_annotator_folds(self, seed=0):
        random.seed(seed)

        num_folds = min([len(aids) for pid, aids in self.pid_to_annotators.items()])
        folds = [{} for i in range(num_folds)]
        for pid, aids in self.pid_to_annotators.items():
            random.shuffle(aids)

            for i, f in enumerate(folds):
                folds[i][pid] = aids[i]

        return folds

    def generate_response(self, paper: IntertextDocument, item_id: str, inputs: list[str]):
        pid = paper.meta["pid"]
        assert pid in self.annotator_per_paper, "provided paper is not part of this human baseline"

        aid = self.annotator_per_paper[pid]

        if self.binary_only:
            return self.wes[pid][aid][item_id]["answer"] == "Yes"
        else:
            return self.wes[pid][aid][item_id]

    def batch_generate_responses(self, paper: IntertextDocument, item_ids: list[str], inputs: list[list]) -> list:
        res = []
        for ii in item_ids:
            res += [self.generate_response(paper, ii, [])]

        return res

    def preprocess_inputs(self, paper: IntertextDocument, item_ids: list[str], inputs: list):
        processed = []

        for ix, item_id in enumerate(item_ids):
            # if binary mode, turn inputs to binary, otherwise just take object as it is
            if self.binary_only:
                processed += [inputs[ix]["answer"] == "Yes"]
            else:
                processed += [inputs[ix]]

        return processed