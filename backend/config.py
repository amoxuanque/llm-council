"""Configuration for the LLM Council."""

import os
from dotenv import load_dotenv

load_dotenv()

# SiliconFlow API key
SILICONFLOW_API_KEY = os.getenv("OPENROUTER_API_KEY")

# API endpoint (硅基流动)
SILICONFLOW_API_URL = os.getenv(
    "SILICONFLOW_API_URL",
    os.getenv("OPENROUTER_API_URL", "https://api.siliconflow.cn/v1/chat/completions"),
)

# Council members - diverse models from different providers for maximum collective intelligence
# Selection rationale:
#   - DeepSeek-R1: Strong reasoning model (chain-of-thought), excels at logic/math/code
#   - Qwen3.5-397B: Latest Qwen MoE (397B total, 17B active), strong multilingual + long context
#   - Kimi-K2: Moonshot's agent model (1T total, 32B active), different training paradigm
#   - GLM-4.6: Zhipu's latest, good at Chinese understanding and generation
COUNCIL_MODELS = [
    "deepseek-ai/DeepSeek-R1",
    "Qwen/Qwen3.5-397B-A17B",
    "moonshotai/Kimi-K2-Instruct-0905",
    "zai-org/GLM-4.6",
]

# Chairman model - synthesizes final response
# IMPORTANT: Chairman must NOT be a council member to avoid bias
# DeepSeek-V3.2 is a strong general-purpose model not in the council
CHAIRMAN_MODEL = "deepseek-ai/DeepSeek-V3.2"

# Title generation model (fast and cheap)
TITLE_MODEL = "deepseek-ai/DeepSeek-V3.2"

# Data directory for conversation storage
DATA_DIR = "data/conversations"
