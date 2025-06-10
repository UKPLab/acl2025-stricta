import json
import os.path
import time
from datetime import datetime

import openai
from intertext_graph import IntertextDocument
from langchain_core.messages import HumanMessage
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate

from langchain_community.callbacks import get_openai_callback
from langchain_openai import AzureOpenAI, AzureChatOpenAI
from openai import RateLimitError

from .LlmWorkflow import LlmWorkflow
from ..utils import item_answer_to_text

import tiktoken


class OpenAiWorkflow(LlmWorkflow):
    def __init__(self,
                 workflow,
                 model_name,
                 gen_config=None,
                 instruct_prompt=None,
                 binary_only=False,
                 examples: dict = None,
                 compress_paper=True,
                 multimodal=False,
                 context_window_size=15000,
                 cost_cache_dir=None):
        super().__init__(workflow, model_name, binary_only=binary_only, do_not_initialize=False)

        self.model_name = model_name

        self.gen_config = gen_config
        self.multimodal = multimodal

        self.llm = None
        self.base_prompt = None
        self.ex_prompt = None
        self.inf_prompt = None
        self.compress_paper = compress_paper
        self.context_window_size = context_window_size

        self.examples = examples

        assert cost_cache_dir is not None, "the costs for OpenAI models need to be tracked and stored on disk. You did not provide a directory for storing this information"
        assert os.path.exists(cost_cache_dir), f"the provided cache dir does not exist: {cost_cache_dir}"

        self._cost_cache_dir = cost_cache_dir
        self._cost_cache_file = self._cost_cache_dir + "/" + self.model_name.replace("/",
                                                                                     "_") + "_" + datetime.now().strftime(
            "%Y-%m-%d-%S-%f")[:-3] + ".json"
        self.costs = {
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_cost_usd": 0
        }

        self._setup()

        if instruct_prompt:
            self.set_instructions(instruct_prompt)

        self.debug = False # "DEBUG_MODE" in os.environ and os.environ["DEBUG_MODE"]
        if self.debug:
            print("RUNNING IN DEBUG MODE!")

    def _setup(self):
        print("Connecting to AZURE in chat mode with:")
        print("DEPLOYMENT = ", self.model_name)
        print("ENDPOINT = ", os.environ["AZURE_OPENAI_ENDPOINT"])
        print("API VERSION = ", os.environ["AZURE_OPENAI_API_VERSION"])

        self.llm = AzureChatOpenAI(
            openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"].replace('"', ''),
            azure_deployment=self.model_name,
            api_key=os.environ["AZURE_OPENAI_API_KEY"].replace('"', ''),
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"].replace('"', '')
        )

    def _log_costs(self, cb_result):
        self.costs["total_tokens"] += cb_result.total_tokens
        self.costs["prompt_tokens"] += cb_result.prompt_tokens
        self.costs["completion_tokens"] += cb_result.completion_tokens
        self.costs["total_cost_usd"] += cb_result.total_cost

        # print
        print("COSTS:" + str(self.costs["total_cost_usd"]))
        print("PROMPTED:" + str(self.costs["prompt_tokens"]))
        print("COMPLETED:" + str(self.costs["completion_tokens"]))

        # cache
        with open(self._cost_cache_file, "w+") as f:
            json.dump(self.costs, f)

    def set_instructions(self, prompt_template, node_type=None, vars=None):
        if vars is None:
            vars = ["task", "description", "parents", "example"]

        assert type(prompt_template) != str, "expected chat prompt templates for OpenAI models"

        # validate
        assert all("{" + v + "}" in "\n".join(str(t) for t in prompt_template) for v in
                   vars), f"WARNING: provided prompt is missing some parameters; expected: {vars}"

        # set template
        if node_type is None:
            self.base_prompt = ChatPromptTemplate.from_messages(prompt_template)
        elif node_type == "extract":
            self.ex_prompt = ChatPromptTemplate.from_messages(prompt_template)
        elif node_type == "infer":
            self.inf_prompt = ChatPromptTemplate.from_messages(prompt_template)

    def postprocess_outputs(self, paper: IntertextDocument, item_ids: list, outputs: list) -> list:
        res = [self._parse_response_by_format(iid, out) for iid, out in zip(item_ids, outputs)]

        # store raw output in addition to parsed ones
        for r, o in zip(res, outputs):
            if r is None:
                r = {"answer": None}

            r["raw"] = o

        return res

    def preprocess_inputs(self, paper: IntertextDocument, item_ids: list, inputs: list) -> list:
        processed = []

        tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")  # use gpt 3.5 tokenizer as default

        for ix, item_id in enumerate(item_ids):
            in_prompt = self.workflow[item_id]["prompt"]

            if self.multimodal and type(inputs[ix]) == dict and "imgs" in inputs[ix]:  # input is given in the format {text: "", imgs: {fig_name: ""}}
                to_text = inputs[ix]["text"]
            else:
                to_text = item_answer_to_text(item_id, inputs[ix], self.workflow, paper,
                                              compress_paper=self.compress_paper)

            # fix size if needed otherwise keep as is (this only works if read nodes are only added once and there is a margin in the content size included!)
            to_text_tokens = tokenizer.encode(to_text)
            if len(to_text_tokens) > self.context_window_size:
                print("WARN: Truncating answer to max content window size")

                to_text_tokens = to_text_tokens[:self.context_window_size]
                to_text = tokenizer.decode(to_text_tokens)

            is_read = self.workflow[item_id]["action"] == "read"

            if is_read:
                tba = to_text
            else:
                tba = f"""Question {ix + 1}: {in_prompt}
                        Answer: {to_text}
                        """

            if self.multimodal and type(inputs[ix]) == dict and "imgs" in inputs[ix]:
                tba = {"text": tba, "imgs": inputs[ix]["imgs"]}

            processed += [tba]

        return processed

    def _multimodal_inputs(self, inputs):
        return [i for i in inputs if type(i) == dict and "imgs" in i]

    def generate_response(self, paper: IntertextDocument, item_id: str, inputs: list[str]):
        n_inputs = inputs[:]

        if self.workflow[item_id]["action"] == "extract" and self.ex_prompt:
            prompt = self.ex_prompt
        elif self.workflow[item_id]["action"].startswith("infer") and self.inf_prompt:
            prompt = self.inf_prompt
        else:
            prompt = self.base_prompt

        item_multimodal = "multimodal" in self.workflow[item_id] and self.workflow[item_id]["multimodal"]
        if self.multimodal and item_multimodal:  # multimodal inputs
            msgs = []
            for mi in self._multimodal_inputs(n_inputs):
                for img_name, img in mi["imgs"].items():
                    img_msg = HumanMessage(
                        content=[
                            {"type": "text", "text": img_name},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": img["base64"]
                                }
                            },
                        ]
                    )
                    msgs += [img_msg]

            prompt = prompt + msgs

        if self.multimodal:  # reformat multimodal inputs to texts if applicable
            for ix, i in enumerate(n_inputs):
                if item_multimodal and type(i) == dict and "imgs" in i and len(i["imgs"]) > 0:  # item expects only figures (for now); skip the textual inputs
                    n_inputs[ix] = "See figures in below messages."
                    continue

                if type(i) == dict and "text" in i:
                    n_inputs[ix] = i["text"]  # reduce item to text when presented in multimodal format

        chain = prompt | self.llm
        inp = dict(
            task=self.workflow[item_id]["prompt"],
            description=self.workflow[item_id]["description"],
            example=self.workflow[item_id]["example"],
            parents="\nAND\n".join([p for p in n_inputs]),
            format=self._item_id_to_response_format(item_id)
        )

        with get_openai_callback() as cb:
            res = self._call_openai([inp], chain)
            self._log_costs(cb)

        return res[0]

    def _fake_openai_call(self, full_input, chain):
        result = []

        for fi in full_input:
            result += [fi["format"]]

        return result

    def _call_openai(self, full_input, chain):
        if self.debug:
            return self._fake_openai_call(full_input, chain)

        # https://medium.com/@hey_16878/efficient-batch-processing-with-langchain-and-openai-overcoming-ratelimiterror-daa9de4bbd8b
        results = []

        max_retries = 6
        delay_increment = 60

        if len(full_input) == 0:
            return []

        # Optimized batch size calculation
        batch_size = min(20, len(full_input))

        for i in range(0, len(full_input), batch_size):
            batch = full_input[i: i + batch_size]

            retries = 0
            while retries <= max_retries:
                try:
                    result = chain.batch(batch)
                    results.extend(map(lambda x: x.content, result))
                    break  # Exit the retry loop once successful
                except (RateLimitError, openai.APIConnectionError) as connection_error:
                    delay = (retries + 1) * delay_increment
                    print(f"OPENAI WARNING: {connection_error}. Retrying in {delay} seconds...")
                    time.sleep(delay)
                    retries += 1

                    if retries > max_retries:
                        print(
                            f"ERROR: Max retries reached for batch starting at index {i}. Raising Error!")
                        raise connection_error

        return results

    def single_batch(self, chain, prompt, inputs):
        time.sleep(2)  # force waiting time

        return self._call_openai(inputs, chain)

    def multimodal_batch_generate_responses(self, paper, item_ids, inputs):
        assert self.ex_prompt is not None and self.inf_prompt is not None, "multimodal inference expects separate prompts for inference and extraction"

        exs = [(ix, iid, inputs) for (ix, (iid, inputs)) in enumerate(zip(item_ids, inputs)) if
               self.workflow[iid]["action"] == "extract"]
        infs = [(ix, iid, inputs) for (ix, (iid, inputs)) in enumerate(zip(item_ids, inputs)) if
                self.workflow[iid]["action"].startswith("infer")]

        assert len(exs) + len(infs) == len(item_ids), "not all items are covered. are there read nodes?"

        def _infer(llm, prompt, items, is_extract):
            time.sleep(3)  # force waiting time

            return [self.generate_response(paper, item_id, ins) for item_id, ins in items]  # no batching, itemwise calling

        res1 = _infer(self.llm, self.ex_prompt, [(iid, ins) for ix, iid, ins in exs], True)
        res2 = _infer(self.llm, self.inf_prompt, [(iid, ins) for ix, iid, ins in infs], False)

        res = list(zip([ix for ix, iid, inp in exs], res1)) + list(zip([ix for ix, iid, inp in infs], res2))
        return [next(r for r in res if r[0] == ix)[1] for ix, ii in enumerate(item_ids)]


    def batch_generate_responses(self, paper: IntertextDocument, item_ids: list[str], inputs: list[list]) -> list:
        assert len(item_ids) == len(inputs), "the number of inputs does not match the number of items to predict"

        if self.multimodal:
            return self.multimodal_batch_generate_responses(paper, item_ids, inputs)

        def _infer(chain, items):
            time.sleep(3)  # force waiting time

            return self._call_openai([{
                "task": self.workflow[item_id]["prompt"],
                "description": self.workflow[item_id]["description"],
                "example": self.workflow[item_id]["example"],
                "parents": "\nAND\n".join([p for p in ins]),
                "format": self._item_id_to_response_format(item_id)
            } for item_id, ins in items], chain)

        if self.ex_prompt and self.inf_prompt:
            exs = [(ix, iid, inputs) for (ix, (iid, inputs)) in enumerate(zip(item_ids, inputs)) if
                   self.workflow[iid]["action"] == "extract"]
            infs = [(ix, iid, inputs) for (ix, (iid, inputs)) in enumerate(zip(item_ids, inputs)) if
                    self.workflow[iid]["action"].startswith("infer")]

            assert len(exs) + len(infs) == len(item_ids), f"not all items are covered. are there read nodes? Missing: {set(item_ids).difference(set([e[1] for e in exs]).union(set([i[1] for i in infs])))}"

            with get_openai_callback() as cb:
                res1 = _infer(self.ex_prompt | self.llm, [(iid, ins) for ix, iid, ins in exs])
                res2 = _infer(self.inf_prompt | self.llm, [(iid, ins) for ix, iid, ins in infs])
                self._log_costs(cb)

            res = list(zip([ix for ix, iid, inp in exs], res1)) + list(zip([ix for ix, iid, inp in infs], res2))
            return [next(r for r in res if r[0] == ix)[1] for ix, ii in enumerate(item_ids)]
        else:
            with get_openai_callback() as cb:
                res = _infer(self.base_prompt | self.llm, zip(item_ids, inputs))
                self._log_costs(cb)

            return res
