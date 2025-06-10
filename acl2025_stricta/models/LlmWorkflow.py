import json
from json import JSONDecodeError

from intertext_graph import IntertextDocument
from .AutoWorkflow import AutoWorkflow
from ..utils import item_answer_to_text


class LlmWorkflow(AutoWorkflow):
    def __init__(self,
                 workflow,
                 model_name,
                 gen_config=None,
                 model_path=None,
                 model_subpath=None,
                 instruct_prompt=None,
                 binary_only=False,
                 examples:dict=None,
                 compress_paper=False,
                 multimodal=False,
                 do_not_initialize=False):
        super().__init__(workflow, model_name, binary_only)

        self.model_name = model_name
        self.model_path = model_path
        self.model_subpath = model_subpath
        self.truncate_at = 2048

        self.gen_config = gen_config
        self.multimodal = multimodal

        self.llm = None
        self.base_prompt = None
        self.ex_prompt = None
        self.inf_prompt = None
        self.compress_paper = compress_paper

        self.do_not_initialize = do_not_initialize

        self.examples = examples

        if instruct_prompt:
            self.set_instructions(instruct_prompt)

    def set_instructions(self, prompt_template, node_type=None, vars=None):
        raise NotImplementedError("set_instructions is not implemented")

    def postprocess_outputs(self, paper: IntertextDocument, item_ids: list, outputs: list) -> list:
        return [self._parse_response_by_format(iid ,out) for iid, out in zip(item_ids, outputs)]

    def preprocess_inputs(self, paper: IntertextDocument, item_ids: list, inputs: list) -> list:
        processed = []

        for ix, item_id in enumerate(item_ids):
            in_prompt = self.workflow[item_id]["prompt"]
            to_text = item_answer_to_text(item_id, inputs[ix], self.workflow, paper, compress_paper=self.compress_paper)

            is_read = self.workflow[item_id]["action"] == "read"

            if is_read:
                tba = to_text
            else:
                tba = f"""Question {ix + 1}: {in_prompt}
                        Answer: {to_text}
                        """

            processed += [tba]

        return processed

    def _item_id_to_response_format(self, item_id):
        item = self.workflow[item_id]

        if "format" in item and item["format"] == "truth":
            return """
            {
                "decision": "yes/no",
                "explanation": "text"
            }
            """
        elif "format" in item and item["format"] == "list":
            return """
                {
                    "listing": [
                        "item 1 text",
                        "item 2 text"
                    ]
                }
                """
        else:
            return """
                {
                    "answer": "some string"
                }
                """

    def _parse_response_by_format(self, item_id, out):
        item = self.workflow[item_id]

        try:
            parsed = json.loads(out)
        except JSONDecodeError:
            parsed = None

        if parsed is None:  # heuristically clean output
            first_curly = -1
            last_curly = -1

            for ix, c in enumerate(out):
                if c == "{":
                    first_curly = ix

            if first_curly == -1:
                print(f"NOTE: Failed to parse model output; couldn't find first curly bracket: {out}")
                return None

            for ix, c in enumerate(reversed(out)):
                if c == "}":
                    last_curly = ix

            if last_curly == -1:
                print(f"NOTE: Failed to parse model output; couldn't find last curly bracket: {out}")
                return None

            if last_curly > 0:
                to_parse = out[first_curly:-last_curly]
            else:
                to_parse = out[first_curly:]

            try:
                parsed = json.loads(to_parse)
            except JSONDecodeError:
                parsed = None

            # remove backslashes inside strings where necessary
            if parsed is None:
                to_parse = to_parse.replace("\\", "\\\\")

            try:
                parsed = json.loads(to_parse)
            except JSONDecodeError:
                parsed = None

            if parsed is None:
                print(f"NOTE: Failed to parse model output after cleaning: {out}")
                return None

        if "format" in item and item["format"] == "truth":
            try:
                return {
                    "answer": parsed["decision"].capitalize(),
                    "text": parsed["explanation"]
                }
            except Exception:
                print(f"NOTE: Parsed as JSON, but failed to interpret parsed model output: {out}")
                return None
        elif "format" in item and item["format"] == "list":
            try:
                return {
                    "answer": parsed["listing"],
                    "text": None
                }
            except Exception:
                print(f"NOTE: Parsed as JSON, but failed to interpret parsed model output: {out}")
                return None
        else:
            try:
                return {
                    "answer": None,
                    "text": parsed["answer"]
                }
            except Exception:
                print(f"NOTE: Parsed as JSON, but failed to interpret parsed model output: {out}")
                return None

    def generate_response(self, paper: IntertextDocument, item_id: str, inputs: list[str]):
        if self.workflow[item_id]["action"] == "extract" and self.ex_prompt:
            prompt = self.ex_prompt
        elif self.workflow[item_id]["action"].startswith("infer") and self.inf_prompt:
            prompt = self.inf_prompt
        else:
            prompt = self.base_prompt

        chain = prompt | self.llm
        inp = dict(
            task=self.workflow[item_id]["prompt"],
            description=self.workflow[item_id]["description"],
            example=self.workflow[item_id]["example"],
            parents="\nAND\n".join([p for p in inputs])
        )

        return chain.invoke(inp).replace(prompt.format(**inp), "")

    def single_batch(self, chain, prompt, inputs):
        return chain.batch(inputs, config=dict(max_concurrency=5))

    def batch_generate_responses(self, paper: IntertextDocument, item_ids: list[str], inputs: list[list]) -> list:
        def _infer(chain, items):
            return chain.batch([{
                "task": self.workflow[item_id]["prompt"],
                "description": self.workflow[item_id]["description"],
                "example": self.workflow[item_id]["example"],
                "parents": "\nAND\n".join([p for p in ins])
            } for item_id, ins in items], config=dict(max_concurrency=5))

        if self.ex_prompt and self.inf_prompt:
            exs = [(ix, iid, inputs) for (ix, (iid, inputs)) in enumerate(zip(item_ids, inputs)) if self.workflow[iid]["action"] == "extract"]
            infs = [(ix, iid, inputs) for (ix, (iid, inputs)) in enumerate(zip(item_ids, inputs)) if self.workflow[iid]["action"].startswith("infer")]

            assert len(exs) + len(infs) == len(item_ids), "not all items are covered. are there read nodes?"

            res1 = _infer(self.ex_prompt | self.llm, [(iid, ins) for ix, iid, ins in exs])
            res2 = _infer(self.inf_prompt | self.llm, [(iid, ins) for ix, iid, ins in infs])

            res = list(zip([ix for ix, iid, inp in exs], res1)) + list(zip([ix for ix, iid, inp in infs], res2))
            return [next(r for r in res if r[0] == ix)[1] for ix, ii in enumerate(item_ids)]
        else:
           return _infer(self.base_prompt | self.llm, zip(item_ids, inputs))