from intertext_graph import Etype


def struc_answer_to_text(response):
    res = ""

    if response is None:
        return "None"
    elif response["answer"] is not None:
        if type(response["answer"]) == str:  # truth value
            res += response["answer"]
            res += f", because: {response['text']}"
        else:
            res += "\n".join(["* " + e for e in response["answer"]]) if len(response["answer"]) > 0 else "None"
    else:
        res += response["text"]

    return res


def item_answer_to_text(item, answer, workflow, paper, compress_paper=False):
    step = workflow[item]

    def _span_answer_to_text(spans):
        unrolled = [n for n in paper.unroll_graph() if
                    n.ix in spans or any(m for m in paper.breadcrumbs(n, Etype.PARENT) if m.ix in spans)]

        if compress_paper:  # discard references
            start_of_references = [ix for ix, n in enumerate(unrolled) if n.content.lower().strip() == "references"]
            if len(start_of_references) == 1:
                unrolled = unrolled[:start_of_references[0]]

        return "\n\n".join([n.content for n in unrolled])


    if step["action"] == "read":
        return _span_answer_to_text(answer["spans"])  # simplified
    elif step["action"] in ["extract", "infer", "infer_knowledge"]:
        return struc_answer_to_text(answer)