from typing import Any, Dict, List, Union

from torch.utils.data import Dataset
from transformers import AutoTokenizer

from prompt import PROMPT_DICT
from util import read_jsonl

MAX_CONTEXT_LENGTH = 512


class SimpleEvaluationDataset(Dataset):
    def __init__(
        self,
        data_name: str,
        data_path_template: str,
        prompt_strategy: str,
        tokenizer: AutoTokenizer,
        few_shot_data_path: str,
        num_shot: int,
    ):
        self.data_name, self.data_path_template = data_name, data_path_template
        self.tokenizer, self.prompt_strategy = tokenizer, prompt_strategy
        self.num_shot, self.few_shot_data_path = num_shot, few_shot_data_path

        raw_example_list = self._load_data(data_path_template)
        self.actual_example_length = len(raw_example_list)
        if prompt_strategy == "pairwise":
            few_example_list = self._load_data(few_shot_data_path)
            assert len(few_example_list) == num_shot
            self.few_example_score = [float(e["score"]) for e in few_example_list]
        else:
            few_example_list = None

        self.examples = self._build_example(raw_example_list, few_example_list)

        self._check_example_size()

    def _check_example_size(self):

        assert self.actual_example_length * 2 * len(self.few_example_score) == len(self.examples), (
            self.actual_example_length,
            len(self.few_example_score),
            len(self.examples),
        )

    def _load_data(self, data_path_template: str) -> List[List[Dict[str, Any]]]:
        data = read_jsonl(data_path_template.format(data_name=self.data_name))

        # strip!
        output_data = []
        for el in data:
            for k, v in el.items():
                if isinstance(v, str):
                    el[k] = v.strip()
                elif isinstance(v, list) and all([isinstance(u, str) for u in v]):
                    el[k] = [u.strip() for u in v]
            output_data.append(el)
        return output_data

    def __getitem__(self, index):
        return self.examples[index]

    def __len__(self):
        return len(self.examples)

    def _build_example(
        self,
        example_list: List[Dict[str, Any]],
        few_example_list: Union[None, List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        level_key = "turn_level"
        template = PROMPT_DICT[level_key][self.prompt_strategy]
        assert few_example_list is not None
        examples = _make_example_with_pairwise_prompt(example_list, few_example_list, template, self.tokenizer)
        return examples


def _make_example_with_pairwise_prompt(
    example_list: List[Dict[str, Any]],
    few_example_list: List[Dict[str, Any]],
    template: str,
    tokenizer: AutoTokenizer,
):
    output = []
    keys_in_interest = ["history", "response", "dialogue"]
    for example in example_list:
        id_, score = example["id_"], float(example["score"])
        texts = {k: example[k] for k in keys_in_interest if k in example}
        texts = {k: "\n".join(v) if k in ["history", "dialogue"] else v for k, v in texts.items()}
        texts["knowledge"] = (
            "\n".join(["Background Knowledge:"] + example["knowledge"]) + "\n\n"
            if "knowledge" in example and example["knowledge"] is not None
            else ""
        )

        for few_example in few_example_list:
            few_texts = {k: few_example[k] for k in keys_in_interest if k in few_example}
            few_texts = {k: "\n".join(v) if k in ["history", "dialogue"] else v for k, v in few_texts.items()}
            few_texts["knowledge"] = (
                "\n".join(["Background Knowledge:"] + few_example["knowledge"]) + "\n\n"
                if "knowledge" in few_example and few_example["knowledge"] is not None
                else ""
            )
            # real example first
            res = {
                **{k + "_1": v for k, v in texts.items()},
                **{k + "_2": v for k, v in few_texts.items()},
            }
            if res["knowledge_1"] != "":
                assert "Background Knowledge:\n" in res["knowledge_1"]
                res["knowledge_1"] = res["knowledge_1"].replace("Background Knowledge:", "Background Knowledge A:")
            if res["knowledge_2"] != "":
                assert "Background Knowledge:\n" in res["knowledge_2"]
                res["knowledge_2"] = res["knowledge_2"].replace("Background Knowledge:", "Background Knowledge B:")

            text_1 = template.format(**res)
            # real example last
            res2 = {
                **{k + "_2": v for k, v in texts.items()},
                **{k + "_1": v for k, v in few_texts.items()},
            }
            if res2["knowledge_1"] != "":
                assert "Background Knowledge:\n" in res2["knowledge_1"]
                res2["knowledge_1"] = res2["knowledge_1"].replace("Background Knowledge:", "Background Knowledge A:")
            if res2["knowledge_2"] != "":
                assert "Background Knowledge:\n" in res2["knowledge_2"]
                res2["knowledge_2"] = res2["knowledge_2"].replace("Background Knowledge:", "Background Knowledge B:")
            text_2 = template.format(**res2)
            input_ids_both = [tokenizer(text_)["input_ids"] for text_ in [text_1, text_2]]
            few_id_, few_score = few_example["id_"], float(few_example["score"])

            output.append(
                {
                    "input_ids": input_ids_both[0],
                    "score": score,
                    "ref_score": few_score,
                    "id_": f"{id_}|||{few_id_}|||first",
                }
            )
            output.append(
                {
                    "input_ids": input_ids_both[1],
                    "score": score,
                    "ref_score": few_score,
                    "id_": f"{few_id_}|||{id_}|||second",
                }
            )
    return output
