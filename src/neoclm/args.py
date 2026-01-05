import os
from dataclasses import dataclass, field
from typing import Optional, Union

from transformers import MODEL_FOR_CAUSAL_LM_MAPPING, TrainingArguments
from transformers.trainer_utils import SchedulerType, IntervalStrategy
from transformers.utils.versions import require_version

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

@dataclass
class ModelArgs:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `hf auth login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to trust the execution of code from datasets/models defined on the Hub."
                " This option should only be set to `True` for repositories you trust and in which you have read the"
                " code, as it will execute code present on the Hub on your local machine."
            )
        },
    )
    dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataArgs:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The number of processes to use for the preprocessing. "
                "If None (default), automatically set to 0.75 of CPU cores divided by world size for world-size aware parallelism."
            )
        },
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )

    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")

        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."
        
        # Auto-set preprocessing_num_workers if not provided
        if self.preprocessing_num_workers is None:
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            cpu_count = os.cpu_count() or 1
            self.preprocessing_num_workers = int(max(1, cpu_count * 0.75 / world_size))


@dataclass
class OpinionatedTrainArgs(TrainingArguments):
    seed: int = field(default=1228, metadata={"help": "Random seed that will be set at the beginning of training."})
    data_seed: Optional[int] = field(default=1001, metadata={"help": "Random seed to be used with data samplers."})

    lr_scheduler_type: Union[SchedulerType, str] = field(
        default="cosine",
        metadata={"help": "The scheduler type to use."},
    )

    warmup_ratio: float = field(
        default=0.01, metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."}
    )

    bf16: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to use bf16 (mixed) precision instead of 32-bit. Requires Ampere or higher NVIDIA"
                " architecture or using CPU (use_cpu) or Ascend NPU. This is an experimental API and it may change."
            )
        },
    )

    torch_compile: bool = field(
        default=True, metadata={"help": "If set to `True`, the model will be wrapped in `torch.compile`."}
    )

    eval_strategy: Union[IntervalStrategy, str] = field(
        default="steps",
        metadata={"help": "The evaluation strategy to use."},
    )

    eval_steps: Optional[float] = field(
        default=0.05,
        metadata={
            "help": (
                "Run an evaluation every X steps. Should be an integer or a float in range `[0,1)`. "
                "If smaller than 1, will be interpreted as ratio of total training steps."
            )
        },
    )

    report_to: Union[None, str, list[str]] = field(
        default="wandb", metadata={"help": "The list of integrations to report the results and logs to."}
    )

    logging_steps: float = field(
        default=1,
        metadata={
            "help": (
                "Log every X update steps. If X >= 1, treated as an absolute number of steps. "
                "If 0 <= X < 1, treated as a ratio of total training steps."
            )
        },
    )

    save_total_limit: Optional[int] = field(
        default=2,
        metadata={
            "help": (
                "If a value is passed, will limit the total amount of checkpoints. Deletes the older checkpoints in"
                " `output_dir`. When `load_best_model_at_end` is enabled, the 'best' checkpoint according to"
                " `metric_for_best_model` will always be retained in addition to the most recent ones. For example,"
                " for `save_total_limit=5` and `load_best_model_at_end=True`, the four last checkpoints will always be"
                " retained alongside the best model. When `save_total_limit=1` and `load_best_model_at_end=True`,"
                " it is possible that two checkpoints are saved: the last one and the best one (if they are different)."
                " Default is unlimited checkpoints"
            )
        },
    )

    # because we do causal LM training, these can be preset
    metric_for_best_model: Optional[str] = field(
        default="eval_loss", metadata={"help": "The metric to use to compare two different models."}
    )

    greater_is_better: Optional[bool] = field(
        default=False, metadata={"help": "Whether the `metric_for_best_model` should be maximized or not."}
    )

    overwrite_output_dir: bool = field(
        default=True,
        metadata={
            "help": (
                "Overwrite the content of the output directory. "
                "Use this to continue training if output_dir points to a checkpoint directory."
            )
        },
    )

    sweep_lr: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Learning rate sweep mode. Provide comma-separated learning rates to test. "
                "Each LR will be trained for sweep_lr_steps steps. Example: --sweep_lr 1e-5,3e-5,1e-4,3e-4,1e-3"
            )
        },
    )

    sweep_lr_steps: int = field(
        default=100,
        metadata={
            "help": "Number of training steps to run for each learning rate when using --sweep_lr."
        },
    )

    def __post_init__(self):
        super().__post_init__()
        
        # Parse comma-separated sweep_lr string into list of floats
        if self.sweep_lr is not None:
            try:
                self.sweep_lr = sorted(set([float(lr.strip()) for lr in self.sweep_lr.split(',')]))
            except ValueError as e:
                raise ValueError(
                    f"Invalid --sweep_lr format. Expected comma-separated floats, got: {self.sweep_lr}"
                ) from e