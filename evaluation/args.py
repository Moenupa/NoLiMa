from dataclasses import dataclass, field
from typing import TypedDict

DEFAULT_TASK_TEMPLATE = "You will answer a question based on the following book snippet:\n\n{haystack}\n\nUse the information provided in the book snippet to answer the question. Your answer should be short and based on either explicitly stated facts or strong, logical inferences.\n\nQuestion: {question}\n\n Return only the final answer with no additional explanation or reasoning."


def args_to_dict(args) -> dict:
    """
    Convert dataclass arguments to a dictionary.
    """
    return {k: v for k, v in dict(args).items() if not k.startswith("_")}


@dataclass
class ModelArgs:
    """
    Arguments for LLM/VLM Identifier and its configuration
    """

    model: str = field(
        default="Qwen2.5-VL-7B-Instruct",
        metadata={"help": "HF model path or local dir path for the LLM/VLM"},
    )
    api_key: str = field(
        default="EMPTY",
        metadata={"help": "API key for the LLM/VLM service"},
    )
    api_url: str = field(
        default="http://localhost:1025/v1",
        metadata={"help": "API URL for the LLM/VLM service"},
    )
    api_provider: str = field(
        default="vllm",
        metadata={"help": "API provider type, e.g., openai, vllm, etc."},
    )
    max_tokens: int = field(
        default=192,
        metadata={"help": "Maximum tokens to generate"},
    )
    temperature: float = field(
        default=0.0,
        metadata={"help": "Temperature for generation"},
    )
    top_p: float = field(
        default=1.0,
        metadata={"help": "Top-p sampling parameter"},
    )
    timeout: int = field(
        default=200,
        metadata={"help": "Timeout for API requests in seconds"},
    )
    max_retries: int = field(
        default=3,
        metadata={"help": "Maximum number of retries for API requests"},
    )
    system_prompt: str = field(
        default="You are a helpful assistant",
        metadata={"help": "Default system prompt for the model"},
    )
    tokenizer_type: str = field(
        default="huggingface",
        metadata={"help": "Type of tokenizer to use"},
    )
    tokenizer_model: str = field(
        default=None,
        metadata={
            "help": "Model path or local dir path, e.g. Qwen/Qwen2.5-VL-7B-Instruct."
        },
    )


@dataclass
class DataArgs:
    """
    Arguments for data sources, settings, and templates
    """

    needle_set_path: str = field(
        default="../data/needlesets/needle_set.json",
        metadata={"help": "Path to a file containing the needle tests configuration"},
    )
    haystack_dir: str = field(
        default="../data/haystack/rand_shuffle",
        metadata={"help": "Directory containing the haystack files"},
    )
    task_template: str = field(
        default=DEFAULT_TASK_TEMPLATE,
        metadata={"help": "Task template name"},
    )
    use_default_system_prompt: bool = field(
        default=True,
        metadata={"help": "Use default system prompt"},
    )
    context_length: int = field(
        default=None,
        metadata={"help": "Context length for the needle placement"},
    )
    document_depth_percent_min: float = field(
        default=0,
        metadata={"help": "Minimum document depth percentage"},
    )
    document_depth_percent_max: float = field(
        default=100,
        metadata={"help": "Maximum document depth percentage"},
    )
    document_depth_percent_intervals: float = field(
        default=35,
        metadata={"help": "Number of intervals between min and max depth"},
    )
    shift: float = field(
        default=0,
        metadata={
            "help": "Shift for needle placement, applied to the beginning of the haystack"
        },
    )
    static_depth: float = field(
        default=-1,
        metadata={"help": "Static depth for needle placement"},
    )


@dataclass
class ExpArgs:
    """
    Arguments for identifying, collecting, and reporting experiments.
    """

    metric: str = field(
        default="EM",
        metadata={"help": "Evaluation metric"},
    )
    log_placements_dir: str = field(
        default="",
        metadata={
            "help": "Directory to save needle placement logs. If empty, no logs are saved."
        },
    )
    parent_results_dir: str = field(
        default="evaluation_results",
        metadata={"help": "Parent directory to save results"},
    )
    base_seed: int = field(
        default=42,
        metadata={"help": "Base seed for random operations"},
    )
    prevent_duplicate_tests: bool = field(
        default=True,
        metadata={
            "help": "Prevent duplicate tests in the evaluation, fallback to cached results instead."
        },
    )


class NeedleTestItem(TypedDict):
    id: str
    reasoning_type: str
    system_prompt: str
    task_template: str
    needle: str
    questions: dict[str, str]
    character_set: list[str]
    tests: dict[str, dict]
