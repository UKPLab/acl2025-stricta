import json

from intertext_graph import IntertextDocument, Etype

from .Workflow import Workflow
from ..utils import struc_answer_to_text


class WorkflowExecution:
    def __init__(self, annotator_id, answers, paper=None, workflow=None):
        self.workflow = workflow
        self.annotator_id = annotator_id
        self.paper = paper
        self.answers = answers

        self.text = {}
        self._setup()

    def __getitem__(self, item):
        return self.answers[item]

    def parents(self, item):
        return {i: self[i] for i in self.workflow.parents(item)}

    def children(self, item):
        return {i: self[i] for i in self.workflow.children(item)}

    def get_text(self, item):
        return self.text[item]

    def __len__(self):
        return len(self.answers)

    def __iter__(self):
        for i in self.workflow:
            yield self.answers[i]

    def __contains__(self, item):
        return item in self.answers

    def _setup(self):
        self.text = {}
        for wi in self.workflow:
            self.text[wi] = self._to_text(wi)

    def _relevant_paras(self, item, answer):
        spans = answer["spans"]

        for ci, c in self.children(item).items():
            if c["spans"]:
                spans += c["spans"]

        spans = list(set(spans))

        to_del = []
        for s in range(len(spans)):
            above = [n.ix for n in self.paper.breadcrumbs(self.paper.get_node_by_ix(spans[s]), Etype.PARENT)]

            for s2 in range(s+1, len(spans)):
                if spans[s2] in above:
                    to_del += [s2]

        to_del = [spans[s] for s in set(to_del)]
        spans = set(spans)

        for s in to_del:
            spans.remove(s)

        return list(spans) if len(spans) > 0 else [self.paper.root.ix]

    def _to_text(self, item):
        step  = self.workflow[item]
        answer = self.answers[item]

        def _span_answer_to_text(spans):
            unrolled = [n for n in self.paper.unroll_graph() if n.ix in spans or any(m for m in self.paper.breadcrumbs(n, Etype.PARENT) if m.ix in spans)]
            return "\n\n".join([n.content for n in unrolled])

        res = {"id": f"{self.paper.root.ix}_{item}_{self.annotator_id}"}
        if step["action"] == "read":
            res["out"] = _span_answer_to_text(self._relevant_paras(item, answer))
        elif step["action"] in ["extract", "infer", "infer_knowledge"]:
            res["out"] = struc_answer_to_text(answer)

        return res

    @staticmethod
    def from_json(fpath, workflow:Workflow, paper:IntertextDocument):
        with open(fpath, "r") as f:
            annotations = json.load(f)

        answers = {}
        for step in annotations:
            for annotator_id, answer in step["answers"].items():
                answers[annotator_id] = answers.get(annotator_id, {})
                answers[annotator_id][step["id"]] = answer["output"]

        assert len([1 for a in annotations if a["paper_id"] == paper.root.ix.split(".")[0]]) == len(annotations), \
            "Annotations do not match the paper"

        return {aid: WorkflowExecution(aid, answers[aid], paper, workflow) for aid in answers}
