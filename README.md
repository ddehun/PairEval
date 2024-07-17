# PairEval
Official Code Repository for the paper ["PAIREVAL: Open-domain Dialogue Evaluation with Pairwise Comparison"](https://arxiv.org/pdf/2404.01015) (COLM 2024).


### Abstract
![Overall illustration of PairEval](https://github.com/ddehun/PairEval/blob/main/paireval_main.png?raw=true)
Building a reliable and automated evaluation metric is a necessary but challenging problem for open-domain dialogue systems. Recent studies proposed evaluation metrics that assess generated responses by considering their relevance to previous dialogue histories. Although effective, these metrics evaluate individual responses directly rather than considering their relative quality compared to other responses. To handle this, we propose PAIREVAL, a novel dialogue evaluation metric for assessing responses by comparing their quality against responses in different conversations. PAIREVAL is built on top of open-sourced and moderate-size language models, and we make them specialized in pairwise comparison between dialogue responses. Extensive experiments on multiple benchmarks demonstrate that our metric exhibits a higher correlation with human judgments than baseline metrics. We also find that the proposed comparative metric is more robust in detecting common failures from open-domain dialogue systems, including repetition and speaker insensitivity.

---
### QuickStart
1. Install the following packages.
```
torch
transformers
accelerate
bitsandbytes
scipy
tqdm
```

2. Download our LoRA checkpoints and datasets from [here]() and locate them on the main directory.

3. Obtain your access to [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf).

4. Execute the following code to evaluate PairEval on the preprocessed turn-level FED meta-evaluation dataset released by [this](https://aclanthology.org/2020.sigdial-1.28/) paper.

```
python inference.py
```

5. Check evaluation results on ```output/``` directory.

### Evaluation of Custom Dataset
1. Please reformat your dataset following ```data/evaluaton/fed_turn.jsonl```.
2. change ```--eval_data_name``` argument in ```args.py```.


### FAQ

Please make an issue on this repository or directly contact to ddehun@kaist.ac.kr. 