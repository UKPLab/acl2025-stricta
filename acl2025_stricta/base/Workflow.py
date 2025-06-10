import json
import networkx as nx


class Workflow:
    def __init__(self, name, config=None):
        self.name = name
        self.graph = None

        if config:
            self._setup(config)

    def _setup(self, config):
        self.graph = nx.DiGraph()

        for n in config["steps"]:
            if n["id"] == "FINISHED":
                continue

            node = {
                k: v for k, v in n.items() if k in ["id", "name", "action", "prompt", "example", "description", "format", "judgement", "multimodal"]
            }

            self.graph.add_node(node["id"], **node)
            self.graph.add_edges_from([(ap[1:], node["id"]) for ap in n["action_params"].values()])

    def __iter__(self):
        return self.graph.__iter__()

    def __getitem__(self, item):
        return self.graph.nodes[item]

    def __contains__(self, item):
        return item in self.graph

    def parents(self, item):
        return list(self.graph.predecessors(item))

    def children(self, item):
        return list(self.graph.successors(item))

    def roots(self):
        return [n for n in self.graph if len(self.parents(n)) == 0]

    def to_graph(self, filter_fun=None):
        return self.graph.subgraph([n for n, d in self.graph.nodes(data=True) if filter_fun is None or filter_fun(n, d)]).copy()

    def topological_nodes(self, filter_fun=None):
        return nx.topological_sort(self.to_graph(filter_fun))

    def parallel_topological_nodes(self, filter_fun=None):
        # assume DAG
        roots = self.roots()

        levels = [set(roots)]
        levelled_nodes = set(roots)
        nodes = set(n for n in self if n not in roots)
        while len(nodes) > 0:
            n_level = []

            for n in nodes:
                if set(self.parents(n)).issubset(levelled_nodes):
                    n_level += [n]

            for n in n_level:
                nodes.remove(n)

            levelled_nodes.update(n_level)
            levels += [set(n_level)]

        if filter_fun:
            res = [set(n for n in l if filter_fun(n)) for l in levels]
            res = [l for l in res if len(l) > 0]
        else:
            res = levels

        return res

    def to_binary(self):
        bin_vars = [wi for wi in self if "format" in self[wi] and self[wi]["format"] == "truth"]

        current_graph = self.to_graph()
        non_bin_vars = [wi for wi in current_graph if wi not in bin_vars]
        for nb in non_bin_vars:
            parents, children = list(current_graph.predecessors(nb)), list(current_graph.successors(nb))

            if len(parents) > 0 and len(children) > 0:
                # replace node by an "edge", i.e. connecting all children and parents
                for p in parents:
                    for c in children:
                        current_graph.add_edge(p, c)

            current_graph.remove_node(nb)

        return Workflow._from_graph(self.name + "_binary", current_graph)

    def get_paths(self, start_node, end_node):
        return nx.all_shortest_paths(self.graph, start_node, end_node)

    def get_ancestors(self, node):
        return nx.ancestors(self.graph, node)

    def get_ancestor_graph(self, node, filter_fun=None):
        ancestors = self.get_ancestors(node)
        ancestors |= {node}

        return self.sub([a for a in ancestors if filter_fun is None or filter_fun(a)])

    def is_path(self, path:list[str]) -> bool:
        rpath = list(reversed(path))

        for i, n in enumerate(rpath):
            if i < len(rpath)-1 and not rpath[i+1] in self.parents(n):
                return False

        return True

    def to_adjacency_matrix(self, nodes=None):
        return nx.adjacency_matrix(self.graph, nodelist=nodes)

    def sub(self, node_ids):
        return Workflow._from_graph("sub_" + self.name, self.graph.subgraph(node_ids))

    def filter(self, filter_fun):
        return Workflow._from_graph("sub_" + self.name, self.graph.subgraph(
            [n for n, d in self.graph.nodes(data=True) if filter_fun(n, d)]))

    @staticmethod
    def _from_graph(name, g):
        w = Workflow(name)
        w.graph = g

        return w

    @staticmethod
    def from_json(fpath):
        with open(fpath, "r") as f:
            wf = json.load(f)

        return Workflow(fpath, wf)
