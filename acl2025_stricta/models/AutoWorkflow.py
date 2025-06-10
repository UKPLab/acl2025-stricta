from abc import ABC, abstractmethod

import networkx as nx
from intertext_graph import IntertextDocument

from ..base.Workflow import Workflow


class AutoWorkflow(ABC):
    def __init__(self, workflow, name, binary_only=False):
        self.workflow = workflow
        self.binary_only = binary_only
        self.name = name

        if binary_only:
            self.workflow = self._reduce_workflow_to_binary_nodes(workflow)

    @abstractmethod
    def generate_response(self, paper: IntertextDocument, item_id: str, inputs: list[str]):
        raise NotImplementedError("generate_response not implemented; abstract method.")

    @abstractmethod
    def batch_generate_responses(self, paper: IntertextDocument, item_ids: list[str], inputs: list[list]) -> list:
        raise NotImplementedError("batch_generate_response not implemented; abstract method.")

    def postprocess_outputs(self, paper: IntertextDocument, item_ids: list, outputs: list) -> list:
        return outputs  # do nothing

    def preprocess_inputs(self, paper: IntertextDocument, item_ids: list, inputs: list) -> list:
        processed = []

        for ix, item_id in enumerate(item_ids):
            in_prompt = self.workflow[item_id]["prompt"]

            processed += [f"""Task: {in_prompt}
            Solution: {inputs[ix]}
            """]

        return processed

    def get_name(self):
        return self.name

    def minimal_path_blanked(self, path):
        return [p for n in path for p in self.workflow.parents(n) if p not in path]

    def generate_path(self, inputs: dict, paper: IntertextDocument, path: list[str],
                      ignore_missing_inputs=False) -> dict:
        res = {}
        res.update(inputs)

        assert self.workflow.is_path(path), f"The provided path is no valid connection of nodes {path[0]} to {path[-1]}"

        minimal_ancestors = self.minimal_path_blanked(path)

        assert not ignore_missing_inputs and any(
            [a for a in minimal_ancestors if a not in inputs]), f"Inputs are missing" \
                                                                f" for the given path ending in {path[-1]}. For this path, at least the following inputs are mandatory: {minimal_ancestors} "

        relevant_subworkflow = self.workflow.sub([n for n in self.workflow if n in minimal_ancestors or n in path])

        for n in path:
            pa = relevant_subworkflow.parents(n)
            inputs = self.preprocess_inputs(paper, pa, [res[p] for p in pa])
            out = self.generate_response(paper, n, [inputs[p] for p in pa])

            out = self.postprocess_outputs(paper, [n], [out])
            res[n] = out

        return res

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

        return Workflow._from_graph(workflow.name + "_binary", current_graph)

    def generate_program(self, inputs: dict, paper: IntertextDocument, final_nodes: list[str], node_filter=None):
        assert len(set(final_nodes).intersection(set(inputs.keys()))) == 0, \
            "You provided inputs for the final nodes of the graph. Not allowed -- this makes the workflow program trivial."

        # result incl. inputs
        res = {}
        res.update(inputs)

        # get relevant subgraph for computation
        aux_subgraph = self.workflow.to_graph()

        # inputs replace structural links to parents, so the edges get discarded
        for n in inputs:
            pa = list(aux_subgraph.predecessors(n))
            for p in pa:
                aux_subgraph.remove_edge(p, n)

        # discard all nodes that follow the final nodes
        for f in final_nodes:
            ds = set(nx.descendants(aux_subgraph, f))
            for d in ds:
                aux_subgraph.remove_node(d)

        uaux_subgraph = aux_subgraph.to_undirected()

        # relevant components -> relevant sub workflows
        relevant_components = [nx.node_connected_component(uaux_subgraph, f) for f in final_nodes]
        for i, f in enumerate(final_nodes):
            c = relevant_components[i]

            ans = set(nx.ancestors(aux_subgraph, f)) | {f}
            no_ans = set(n for n in c if n not in ans)
            for n in no_ans:
                c.remove(n)

        relevant_subworkflows = [self.workflow.sub(c) for c in relevant_components]

        for sw in relevant_subworkflows:
            assert all(r in res for r in
                       sw.roots()), f"The provided set of inputs is insufficient to run a program. These nodes need to be specified: {sw.roots()}"

        # iterate over relevant components for computation; even if there is overlap between components in the end all relevant nodes should have consistent values generated
        for relevant_subworkflow in relevant_subworkflows:
            # for each component write results into the res object (distinct answers)
            for wis in relevant_subworkflow.parallel_topological_nodes(node_filter):
                batch_input = []
                batch_ids = []
                for wi in wis:
                    if wi in res:  # skip nodes that are already given by inputs
                        continue

                    pa = relevant_subworkflow.parents(wi)
                    batch_input += [self.preprocess_inputs(paper, pa, [res[p] for p in pa])]
                    batch_ids += [wi]

                # if all nodes of the layer are already provided, there is no need for generation
                if len(batch_input) > 0:
                    outputs = self.batch_generate_responses(
                        paper,
                        batch_ids,
                        batch_input
                    )

                    outputs = self.postprocess_outputs(paper, wis, outputs)

                    res.update(dict(zip(batch_ids, outputs)))

        return res
