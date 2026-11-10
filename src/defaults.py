DEFAULT_MODEL_DIR = "model/Qwen2.5-0.5B-Instruct"
DEFAULT_OUTPUT_DIR = "outputs"
DEFAULT_LLAMA_MODEL_DIR = "model/Llama-3.2-1B-Instruct"
DEFAULT_LLAMA_CHECKPOINT_DIR = "artifacts/llama_3_2_1b_instruct/square_zero_noise"
DEFAULT_LLAMA_BUNDLE_DIR = "artifacts/llama_3_2_1b_instruct/release_bundle"
DEFAULT_LLAMA_CORRECTNESS_DIR = "outputs/evaluation/llama_3_2_1b_instruct/correctness"
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
DEFAULT_MAX_NEW_TOKENS = 8
DEFAULT_SEED = 20260323
DEFAULT_STAGE_B_DTYPE = "float32"
DEFAULT_STAGE_B_SCALE_RANGE = (0.95, 1.05)
DEFAULT_STAGE_C_KAPPA_SAMPLES = 1000
DEFAULT_STAGE_C_FFN_SCALE_RANGE = (0.95, 1.05)

DEFAULT_PROMPTS = [
    "请用一句话介绍你自己。",
    "什么是 Transformer block？",
    "请解释一下注意力机制。",
    "Hello, how are you?",
    "Write one sentence about machine learning.",
]
