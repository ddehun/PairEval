from peft import PeftModel

from args import get_args
from dataset import SimpleEvaluationDataset
from evaluator import Evaluator, get_base_model_and_tokenizer
from util import report_correlation, set_seed, write_jsonl, write_pickle

set_seed(42)
args = get_args()
"""
0. Configuration
"""
# model
adapter_path = args.adapter_path.format(args.adatper_name)
base_lm_name = args.base_model_path
# test set
eval_data = args.eval_data_name
eval_fname_format = args.eval_data_path
# comparison examples
num_comparison_example, comparison_type = (
    args.num_comparison_example,
    args.comparison_type,
)
comparison_fname = args.comparison_fname.format(comparison_type, num_comparison_example)
batch_size = args.batch_size
# output
output_fname = args.output_fname
print("[*] Output fname: ", output_fname)
del args


"""
1. Model
"""
model, tokenizer = get_base_model_and_tokenizer(base_lm_name)
model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)

"""
2. Dataset
"""
dataset = SimpleEvaluationDataset(
    eval_data,
    eval_fname_format,
    "pairwise",
    tokenizer,
    comparison_fname,
    num_comparison_example,
)

"""
3. Evaluation
"""
evaluator = Evaluator(model, tokenizer, batch_size)
result, misc = evaluator.evaluate(dataset)

# report and save
report_correlation(result)
write_jsonl(result, output_fname, True)
write_pickle(misc, output_fname.replace(".jsonl", ".pck"), True)
