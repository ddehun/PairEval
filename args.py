from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--eval_data_name", type=str, default="fed_turn")
    parser.add_argument("--adatper_name", type=str, default="dd")
    parser.add_argument("--num_comparison_example", type=int, default=3)
    parser.add_argument("--comparison_type", type=str, default="random", choices=["random", "test"])
    parser.add_argument("--adapter_path", type=str, default="ckpt/{}")
    parser.add_argument("--base_model_path", type=str, default="llama-transformers/Llama-2-7b-chat-hf/")
    parser.add_argument("--comparison_fname", type=str, default="data/shot/{}.{}.jsonl")
    parser.add_argument("--eval_data_path", type=str, default="./data/evaluation/{data_name}.jsonl")
    parser.add_argument(
        "--output_fname",
        type=str,
        default="output/{eval_data_name}/pairval.{adatper_name}.comparison-{comparison_type}-{num_comparison_example}.jsonl",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()
    args.output_fname = args.output_fname.format(**vars(args))
    return args
