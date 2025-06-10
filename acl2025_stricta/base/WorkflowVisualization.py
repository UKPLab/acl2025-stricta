from pyvis.network import Network
import graphviz


class WorkflowVisualization:
    def __init__(self, workflow, executions, out_path):
        self.workflow = workflow
        self.executions = executions
        self.out_path = out_path

        self.color_map = None
        self.size_map = None
        self.arrow_texts = None

        self.config = dict(
            full_item_spec=False,
            show_all_answers=False,
            show_ids=True
        )

    def _responses_for_item(self, item_id):
        if self.executions is None:
            return []

        for pid, aid_exec in self.executions.items():
            for aid, exec in aid_exec.items():
                yield exec[item_id]

    def show_graphviz(self, name, node_filter=None):
        def type_to_shape(node_type):
            if node_type == "read":
                return "ellipse"
            elif node_type == "extract":
                return "octagon"
            elif node_type == "infer":
                return "doubleoctagon"
            else:
                return "tripleoctagon"

        dot = graphviz.Digraph(comment='Flow Chart', format='png', engine='dot')

        G = self.workflow.to_graph(filter_fun=node_filter)
        for n in G.nodes:
            responses = list(self._responses_for_item(n))

            dot.node(
                n,
                label=f"{self.workflow[n]['name']}({n})",
                shape=type_to_shape(self.workflow[n]["action"]),
                style="filled",
                fillcolor="grey" if self.color_map is None else self.color_map(n, responses),
                width="1in" if self.size_map is None else self.size_map(n, responses)
            )

        for e in G.edges:
            dot.edge(e[0],e[1],color="black", label=self.arrow_texts[e[0],e[1]] if self.arrow_texts is not None else None)

        dot.render(name, directory=self.out_path, format='png', cleanup=True)

    def show_pyviz(self, name, node_filter=None):
        net = Network(height='750px', width='100%', directed=True)

        def type_to_shape(node_type):
            if node_type == "read":
                return "diamond"
            elif node_type == "extract":
                return "dot"
            elif node_type == "infer":
                return "star"
            else:
                return "triangle"

        G = self.workflow.to_graph(filter_fun=node_filter)
        for n in G.nodes:
            responses = list(self._responses_for_item(n))

            net.add_node(
                f"{self.workflow[n]['name']}({n})",
                shape=type_to_shape(self.workflow[n]["action"]),
                title=self.workflow[n]["id"] + ": " + self.workflow[n]["prompt"],
                color= "#97c2fc" if self.color_map is None else self.color_map(n, responses),
                size= 5 if self.size_map is None else self.size_map(n, responses)
            )
        for e in G.edges:
            net.add_edge(f"{self.workflow[e[0]]['name']}({e[0]})", f"{self.workflow[e[1]]['name']}({e[1]})", color="black")

        # Set layout explicitly
        net.show_buttons(filter_=["physics"])

        net.show(name + ".html", notebook=False)

    def set_config(self, full_item_spec=False, show_all_answers=False, show_ids=True):
        self.config = dict(
            full_item_spec=full_item_spec,
            show_all_answers=show_all_answers,
            show_ids=show_ids
        )

    def set_color_map(self, color_map):
        self.color_map = color_map

    def remove_color_map(self):
        self.color_map = None

    def set_normalized_color_map(self, color_map, base_color="#e60000"):
        def to_color(node, answers):
            v = color_map(node, answers)
            assert 0 <= v <= 1, f"invalid color map value {v}. should be normalized to 0, 1."

            return base_color + f"{int((1 - v) * 255):02X}" # if v close to 1, the alpha becomes 0, transparency is minimal

        self.color_map = to_color

    def set_size_map(self, size_map):
        self.size_map = size_map

    def remove_size_map(self):
        self.size_map = None

    def set_arrow_texts(self, arrow_texts):
        self.arrow_texts = arrow_texts

    def remove_arrow_texts(self, arrow_texts):
        self.arrow_texts = None