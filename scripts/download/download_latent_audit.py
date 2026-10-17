from huggingface_hub import snapshot_download

REPOS = [
    ("meta-llama/Llama-3.2-1B-Instruct", "model", None),
    ("zen-E/CODI-llama3.2-1b-Instruct", "model", None),
    ("internlm/SIM_COT-LLaMA3-CODI-1B", "model", None),
    ("openai-community/gpt2", "model", ["config.json", "generation_config.json", "model.safetensors", "tokenizer.json",
                                        "tokenizer_config.json", "vocab.json", "merges.txt"]),
    ("zen-E/CODI-gpt2", "model", None),
    ("internlm/SIM_COT-GPT2-CODI", "model", ["*.json", "*.safetensors", "merges.txt"]),
    ("AlbertTan/CoLaR", "model", ["logs/colar/qsa-gsm/colar-final/*"]),
    ("nvidia/llama-embed-nemotron-8b", "model", None),
    ("openai/gsm8k", "dataset", ["main/*"]),
]


def main() -> None:
    for repo, repo_type, patterns in REPOS:
        path = snapshot_download(repo, repo_type=repo_type, allow_patterns=patterns)
        print(f"{repo}: {path}")


if __name__ == "__main__":
    main()
