import torch
from tabulate import tabulate
from datetime import datetime
import time
import uuid
import humanize as h
import string

from transformers import TrainerCallback


def get_trainable_params(model: torch.nn.Module, verbose=False, logger=None) -> int:
    stdout = logger.info if logger else print

    trainable_params = []

    total = 0
    for n, p in model.named_parameters():
        if p.requires_grad:
            trainable_params.append([p.numel(), n, str(list(p.shape))])
            total += p.numel()

    if verbose and trainable_params:
        trainable_params.append([f"{total:,}", f"Total ({h.metric(total, '')})", ""])
        table_str = tabulate(
                trainable_params,
                headers=["# Count", "Param Name", "Shape"],
                tablefmt="rounded_outline",
                colalign=("right", "left", "left"),
            )
        stdout(f"Layer-wise trainable parameters:\n{table_str}")

    return total


class LogParamsCallback(TrainerCallback):
    """A TrainerCallback that logs layer-wise trainable parameters at the start of training."""
    def __init__(self, logger):
        if logger is None:
            raise ValueError("Logger must be provided to LogParamsCallback.")
        self._logger = logger
        self._done = False

    def on_train_begin(self, args, state, control, **kwargs):
        if self._done:
            return control
        
        # args.should_log is usually True only for Rank 0
        if args.should_log:
            _ = get_trainable_params(kwargs["model"], verbose=True, logger=self._logger)
            
        self._done = True
        return control
    

ALPH = string.digits + string.ascii_lowercase

def to_base36(n: int) -> str:
    if n == 0:
        return "0"
    out = []
    while n:
        n, r = divmod(n, 36)
        out.append(ALPH[r])
    return "".join(reversed(out))

def base36_ts() -> str:
    ts = int(time.time())
    return to_base36(ts)

def from_base36(s: str) -> int:
    return int(s, 36)

def to_datetime_str(base36_ts: str) -> str:
    """Convert a base36 timestamp back to datetime string format."""
    ts = from_base36(base36_ts)
    dt = datetime.fromtimestamp(ts)
    return dt.strftime("%y%m%d_%H%M%S")

def datetime_str() -> str:
    return datetime.now().strftime("%y%m%d_%H%M%S")

def prepend(base: str) -> str:
    """Generate a unique name by appending a timestamp and a short UUID to the base string."""
    timestamp = datetime_str()
    short_uuid = str(uuid.uuid4())[:4]
    return f"{timestamp}-{short_uuid}___{base}"
