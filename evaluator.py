import gc
from functools import partial
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, BitsAndBytesConfig, LlamaForCausalLM

from dataset import SimpleEvaluationDataset
from prompt import TARGET_TOKENS


class Evaluator:
    def __init__(
        self,
        model,
        tokenizer: AutoTokenizer,
        batch_size: int,
        temperature: float = 1.0,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.temperature = temperature
        target_token = TARGET_TOKENS["pairwise"]
        targt_token_ids = self.tokenizer(target_token, return_tensors="pt")["input_ids"]
        assert all(
            [len(e) == 2 for e in targt_token_ids]
        ), targt_token_ids  # bos for llama / eos for T5

        if isinstance(self.model, LlamaForCausalLM) or isinstance(
            self.model.base_model.model, LlamaForCausalLM
        ):
            self.target_token_ids = targt_token_ids[:, 1]
        else:
            raise NotImplementedError

        # only for causal LM
        self.is_left_padding = isinstance(self.model, LlamaForCausalLM) or isinstance(
            self.model.base_model.model, LlamaForCausalLM
        )
        if not self.is_left_padding:
            raise ValueError()

    @torch.inference_mode()
    def evaluate(
        self, dataset: SimpleEvaluationDataset
    ) -> Tuple[List[Dict[str, float]], Any]:
        loader = self._build_data_loader(
            dataset, self.batch_size, self.tokenizer.pad_token_id, self.is_left_padding
        )

        output_list, misc_info = [], {"all_target_probs": [], "example_ids": []}
        for batch_idx, batch in enumerate(tqdm(loader)):
            id_, answer_score = batch["id_"], batch["score"]
            input_ids, attention_mask = [
                batch[k].to("cuda") for k in ["input_ids", "attention_mask"]
            ]
            bs, seq_len = input_ids.shape

            try:
                logits = self.model(
                    input_ids=input_ids, attention_mask=attention_mask
                ).logits[:, seq_len - 1]
                probs = F.softmax(logits / self.temperature, -1).cpu()
                prediction_score, all_target_probs = self._elicit_model_prediction(
                    probs
                )
            except RuntimeError as err:
                print("[E] ", err)
                prediction_score = [-1.0] * bs
                all_target_probs = [
                    [1 / len(self.target_token_ids)] * len(self.target_token_ids)
                ] * bs
                gc.collect()
                torch.cuda.empty_cache()

            assert len(prediction_score) == len(all_target_probs) == bs
            output_list.extend(
                [
                    {"score": s, "pred": p}
                    for s, p in zip(answer_score, prediction_score)
                ]
            )
            misc_info["all_target_probs"].extend(all_target_probs)
            misc_info["example_ids"].extend(id_)

        output_list, misc_info = self._post_process_pairwise_evaluation(
            output_list,
            misc_info,
            dataset.few_example_score,
        )

        return output_list, misc_info

    def _post_process_pairwise_evaluation(
        self,
        prev_output_list: List[Dict[str, float]],
        misc_info: Dict[str, List[Any]],
        few_example_score: List[float],
    ) -> Tuple[List[Dict[str, float]], Dict[str, List[Any]]]:
        """
        1. Aggregrate all information into exmple-wise
        """
        comparison_size = 2 * len(few_example_score)
        prev_answers = [e["score"] for e in prev_output_list]
        all_target_probs = misc_info["all_target_probs"]
        example_ids_list = misc_info["example_ids"]

        assert len(all_target_probs) == len(prev_answers) == len(example_ids_list)
        assert all([len(e) == 2 for e in all_target_probs])
        candidate_prob_avg = np.array(all_target_probs).sum(1).mean().tolist()
        print(f"Average score for candiates: {round(candidate_prob_avg, 2)}")
        # if candidate_prob_avg < 0.8:
        #     raise ValueError(candidate_prob_avg)

        """
        2. Get the actual prediction of an evaulation system
        """
        new_answers = []
        target_example_score = [
            [[None, None] for __ in range(len(few_example_score))]
            for _ in range(len(example_ids_list) // comparison_size)
        ]
        target_example_id = [  # For sanity check
            [[None, None] for __ in range(len(few_example_score))]
            for _ in range(len(example_ids_list) // comparison_size)
        ]
        for idx, id_ in enumerate(example_ids_list):
            first_id, second_id, main_loc = id_.split("|||")
            example_index, few_shot_index = (
                idx // comparison_size,
                (idx % comparison_size) // 2,
            )
            loc = int(main_loc == "second")
            if few_shot_index == 0 and loc == 0:
                new_answers.append(prev_answers[idx])
            tgt_received_score = all_target_probs[idx][loc]
            target_example_score[example_index][few_shot_index][
                loc
            ] = tgt_received_score
            target_example_id[example_index][few_shot_index][loc] = [
                first_id,
                second_id,
            ][loc]

        # sanity check
        for e in target_example_id:
            e = [s for sl in e for s in sl]
            assert len(set(e)) == 1

        for ex_index, ex_result in enumerate(target_example_score):
            for few_index in range(len(few_example_score)):
                assert len(ex_result[few_index]) == 2
                target_example_score[ex_index][few_index] = (
                    sum(ex_result[few_index]) / 2
                )

        target_example_score, few_example_score = np.array(
            target_example_score
        ), np.array(few_example_score)
        prediction = (target_example_score * few_example_score).mean(1).tolist()
        target_example_score = (
            target_example_score.tolist()
        )  # few-shot example들에 대한 평균 예측 값
        """
        3. Prepare the output
        """
        assert len(new_answers) == len(prediction) == len(target_example_score)
        output_list = [{"score": a, "pred": p} for a, p in zip(new_answers, prediction)]
        misc_info["processed_all_target_probs"] = target_example_score
        return output_list, misc_info

    def _elicit_model_prediction(self, probs) -> Tuple[List[float], List[float]]:
        assert probs.dim() == 2
        target_probs = torch.gather(
            probs, 1, self.target_token_ids.repeat(probs.size(0), 1)
        )

        assert target_probs.size(1) == 2
        bs = target_probs.size(0)
        return [-1.0] * bs, target_probs.numpy().tolist()

    @staticmethod
    def _build_data_loader(
        dataset: SimpleEvaluationDataset,
        batch_size: int,
        pad_token_id: int,
        is_left_pad: bool,
    ) -> DataLoader:
        def collate_fn(batch, pad_id: int):
            input_ids, answer_score, id_ = [
                [e[k] for e in batch] for k in ["input_ids", "score", "id_"]
            ]

            if is_left_pad:
                input_ids = [s[::-1] for s in input_ids]
            input_ids = [torch.tensor(id_) for id_ in input_ids]
            input_ids = pad_sequence(input_ids, True, pad_id)
            if is_left_pad:
                input_ids = torch.tensor([s[::-1] for s in input_ids.numpy().tolist()])
            attention_mask = (input_ids != pad_id).long()
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "score": answer_score,
                "id_": id_,
            }

        return DataLoader(
            dataset,
            batch_size,
            shuffle=False,
            collate_fn=partial(collate_fn, pad_id=pad_token_id),
        )


def get_base_model_and_tokenizer(path: str) -> Tuple[LlamaForCausalLM, AutoTokenizer]:
    config_for_4bit = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = LlamaForCausalLM.from_pretrained(
        path, quantization_config=config_for_4bit, device_map="auto"
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(path)
    tokenizer.pad_token = "[PAD]"
    tokenizer.padding_side = "left"
    return model, tokenizer
