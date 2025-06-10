import torch
from intertext_graph import IntertextDocument
from langchain_core.prompts import PromptTemplate

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline
from langchain_huggingface import HuggingFacePipeline

from .LlmWorkflow import LlmWorkflow
from ..utils import item_answer_to_text


class MistralWorkflow(LlmWorkflow):
    def __init__(self,
                 workflow,
                 model_name,
                 gen_config=None,
                 model_path=None,
                 model_subpath=None,
                 instruct_prompt=None,
                 binary_only=False,
                 examples: dict = None,
                 compress_paper=False,
                 multimodal=False,
                 do_not_initialize=False):
        super().__init__(workflow, model_name, gen_config, model_path, model_subpath, instruct_prompt, binary_only,
                         examples, compress_paper, multimodal, do_not_initialize)

        assert not multimodal, "Mistral family models cannot be run in multimodal mode! Figures are not supported."

        self.truncate_at = 2048

        if not do_not_initialize:
            self._setup()

        if instruct_prompt:
            self.set_instructions(instruct_prompt)

    def get_name(self):
        return f"{self.model_name}"

    def _setup(self):
        # construct arguments for loading
        to_load = ""
        if self.model_path:
            to_load += self.model_path

        to_load += self.model_name

        if self.model_subpath:
            to_load += "/" + self.model_subpath

        tokenizer = AutoTokenizer.from_pretrained(to_load,
                                                  use_fast=True,
                                                  truncation=self.truncate_at is not None,
                                                  max_length=self.truncate_at,
                                                  truncation_side="left",
                                                  padding_side="left")

        model = AutoModelForCausalLM.from_pretrained(
            to_load,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="auto",  # for multi GPU!
            local_files_only=True,
            load_in_8bit=True
        )

        if self.gen_config is None:
            self.gen_config = GenerationConfig.from_pretrained(to_load)
            self.gen_config.max_new_tokens = 2048
            self.gen_config.temperature = 0.0001
            self.gen_config.top_p = 0.95
            self.gen_config.do_sample = True
            self.gen_config.repetition_penalty = 1.15

        tokenizer.add_special_tokens({"pad_token": "<pad>"})
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id
        self.gen_config.pad_token_id = tokenizer.pad_token_id

        terminators = [
            tokenizer.convert_tokens_to_ids("</s>")
        ]

        self.llm = HuggingFacePipeline(pipeline=pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            batch_size=4,
            generation_config=self.gen_config,
            eos_token_id=terminators
        ))

    def set_instructions(self, prompt_template, node_type=None, vars=None):
        if vars is None:
            vars = ["task", "description", "parents", "example", "format"]

        if not all("{" + v + "}" in prompt_template for v in vars):
            print(f"WARNING: provided prompt is missing some parameters; expected: {vars}")

        if node_type is None:
            self.base_prompt = PromptTemplate(
                input_variables=vars,
                template=prompt_template,
            )
        elif node_type == "extract":
            self.ex_prompt = PromptTemplate(
                input_variables=vars,
                template=prompt_template,
            )
        elif node_type == "infer":
            self.inf_prompt = PromptTemplate(
                input_variables=vars,
                template=prompt_template,
            )

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
            parents="\nAND\n".join([p for p in inputs]),
            format=self._item_id_to_response_format(item_id)
        )

        return chain.invoke(inp).replace(prompt.format(**inp), "")

    def single_batch(self, chain, prompt, inputs):
        results = chain.batch(inputs)
        return [r.replace(prompt.format(**i), "") for i, r in zip(inputs, results)]

    def batch_generate_responses(self, paper: IntertextDocument, item_ids: list[str], inputs: list[list]) -> list:
        def _infer(chain, prompt, items):
            inputs = [{
                "task": self.workflow[item_id]["prompt"],
                "description": self.workflow[item_id]["description"],
                "example": self.workflow[item_id]["example"],
                "parents": "\nAND\n".join([p for p in ins]),
                "format": self._item_id_to_response_format(item_id)
            } for item_id, ins in items]

            results = chain.batch(inputs)
            return [r.replace(prompt.format(**i), "") for i, r in zip(inputs, results)]

        if self.ex_prompt and self.inf_prompt:
            exs = [(ix, iid, inputs) for (ix, (iid, inputs)) in enumerate(zip(item_ids, inputs)) if
                   self.workflow[iid]["action"] == "extract"]
            infs = [(ix, iid, inputs) for (ix, (iid, inputs)) in enumerate(zip(item_ids, inputs)) if
                    self.workflow[iid]["action"].startswith("infer")]

            assert len(exs) + len(infs) == len(item_ids), "not all items are covered. are there read nodes?"

            res1 = _infer(self.ex_prompt | self.llm, self.ex_prompt, [(iid, ins) for ix, iid, ins in exs])
            res2 = _infer(self.inf_prompt | self.llm, self.inf_prompt, [(iid, ins) for ix, iid, ins in infs])

            res = list(zip([ix for ix, iid, inp in exs], res1)) + list(zip([ix for ix, iid, inp in infs], res2))
            return [next(r for r in res if r[0] == ix)[1] for ix, ii in enumerate(item_ids)]
        else:
            return _infer(self.base_prompt | self.llm, self.base_prompt, zip(item_ids, inputs))
