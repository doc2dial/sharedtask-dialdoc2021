import json
import os
import argparse
from collections import defaultdict

from datasets import load_dataset

DOC_DOMAIN_SPLIT = "train"
YOUR_DATASETS_SOURCE_DIR = ""  # the root folder of your local `datasets` source code.


def text2line(text):
    return text.replace("\n", "\t").replace("\r", "\t").strip()


def btag(tag, text):  # tag the content
    return "<{}>\t{}".format(tag, text2line(text))


def load_doc2dial_seq2seq(args):
    doc_dataset = load_dataset(
        "../datasets/doc2dial",
        name="document_domain",
        split=DOC_DOMAIN_SPLIT,
        cache_dir=args.cache_dir,
    )
    dial_dataset = load_dataset(
        "../datasets/doc2dial",  # path to your datasets source code
        name="dialogue_domain",
        split=args.split,
        cache_dir=args.cache_dir,
        ignore_verifications=True,
    )
    d_doc = defaultdict(dict)
    for ex in doc_dataset:
        d_doc[ex["doc_id"]]["doc_text"] = ex["doc_text"]
        for d_span in ex["spans"]:
            d_doc[ex["doc_id"]][d_span["id_sp"]] = d_span
    source = []
    target = []
    for ex in dial_dataset:
        doc_id = ex["doc_id"]
        d_doc_spans = d_doc[doc_id]
        dial_context = []
        contexts = None
        for i, turn in enumerate(ex["turns"]):
            if not turn[
                "references"
            ]:  # this task only uses instances and evalutes on the grounded turns.
                continue
            utterance = text2line(turn["utterance"])
            utterance_context = btag(turn["role"], utterance)
            if turn["role"] in args.role:  # if current turn is to predict
                contexts = [
                    btag("last_turn", dial_context[-1].split("\t", 1)[-1])
                ]  # add previous utterance as tagged query context
                contexts.extend(
                    dial_context[::-1]
                )  # add dialog history in reverse order as tagged dialogue context
                if args.full_doc:
                    # add entire document as tagged document context
                    contexts += [
                        btag("title", ex["doc_id"]),
                        btag("doc_context", d_doc[doc_id]["doc_text"]),
                    ]
                else:
                    reference_content = ""  # the grounding span content
                    d_sec = {}
                    ref_label = ""
                    for ref in turn["reference"]:
                        sp_id = ref["keys"]
                        sp_label = ref["values"]
                        sec_id = d_doc_spans[sp_id]["id_sec"]
                        # rename sec_id for sorting the text sections in order.
                        if sec_id.startswith("t"):
                            sec_id = sec_id.split("_", 1)[-1] + "_0"
                        else:
                            sec_id = sec_id + "_1"
                        sec_content = d_doc_spans[sp_id]["text_sec"]
                        d_sec[sec_id] = sec_content
                        if "solution" in sp_label:
                            ref_label = "solution"
                        elif "precondition" in sp_label:
                            ref_label = "precondition"
                        if "reference" not in sp_label:
                            reference_content += "\t" + d_doc_spans[sp_id]["text_sp"]
                    reference_context = btag("grounding", reference_content)
                    sec_contents = []
                    for k, v in sorted(d_sec.items()):
                        sec_contents.append(v)
                        contexts += [
                            btag("title", ex["doc_id"]),
                            btag(
                                "doc_context", "\t".join(sec_contents)
                            ),  # use a combine of related sections as document context.
                        ]
                    if args.include_da:
                        da = get_da_name(
                            turn["da"],
                            turn["role"],
                            turn["turn_id"],
                            ref_label,
                            args.simply_da,
                        )
                        da_context = btag("da", da)
                        contexts.extend(da_context)
                    contexts.append(reference_context)
                source.append("\t".join(contexts))
                target.append(utterance)
            dial_context.append(utterance_context)
    assert len(source) == len(
        target
    ), "Need to ensure that source and target are same sized."
    if args.split == "validation":
        args.split = "val"
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(
        os.path.join(args.output_dir, "{}.source".format(args.split)),
        "w",
        encoding="utf8",
    ) as fp:
        fp.write("\n".join(source))
        fp.close()
    with open(
        os.path.join(args.output_dir, "{}.target".format(args.split)),
        "w",
        encoding="utf8",
    ) as fp:
        fp.write("\n".join(target))
        fp.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        type=str,
        required=True,
        help="Data split is 'train', 'validation' or 'test'",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        help="Path for caching the downloaded data by HuggingFace Datasets",
    )
    parser.add_argument(
        "--role",
        type=str,
        default="agent",
        help="which role's utterance for generation",
    )
    parser.add_argument(
        "--full_doc",
        type=bool,
        default=True,
        help="whether use entire document",
    )
    parser.add_argument(
        "--include_da",
        type=bool,
        default=False,
        help="whether to include DA as input",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="path to output the data files",
    )

    args = parser.parse_args()
    load_doc2dial_seq2seq(args)


if __name__ == "__main__":
    main()
