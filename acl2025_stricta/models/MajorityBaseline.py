import numpy as np
from intertext_graph import IntertextDocument

from .AutoWorkflow import AutoWorkflow


class MajorityBaseline(AutoWorkflow):
    def __init__(self, workflow, workflow_executions):
        super().__init__(workflow, "majority_baseline", True)

        self.wes = workflow_executions
        self.iid_to_majority_label = None

        self._setup()

    def _setup(self):
        self.iid_to_majority_label = {}

        for wi in self.workflow:
            for aid_to_exec in self.wes.values():
                for exec in aid_to_exec.values():
                    self.iid_to_majority_label[wi] = self.iid_to_majority_label.get(wi, []) + [exec[wi]["answer"] == "Yes"]

        for wi in self.iid_to_majority_label:
            true_count = np.count_nonzero(np.array(self.iid_to_majority_label[wi]))

            self.iid_to_majority_label[wi] = true_count >= (len(self.iid_to_majority_label[wi]) - true_count)

    def generate_response(self, paper: IntertextDocument, item_id: str, inputs: list[str]):
        return self.iid_to_majority_label[item_id]

    def batch_generate_responses(self, paper: IntertextDocument, item_ids: list[str], inputs: list[list]) -> list:
        res = []
        for ii in item_ids:
            res += [self.generate_response(paper, ii, [])]

        return res

    def preprocess_inputs(self, paper: IntertextDocument, item_ids: list, inputs: list):
        processed = []

        for ix, item_id in enumerate(item_ids):
            processed += [inputs[ix]["answer"] == "Yes"]

        return processed