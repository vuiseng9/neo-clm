#!/usr/bin/env python
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# /// script
# dependencies = [
#     "transformers==4.57.3",
#     "albumentations >= 1.4.16",
#     "accelerate >= 0.12.0",
#     "torch >= 1.3",
#     "datasets >= 2.14.0",
#     "sentencepiece != 0.1.92",
#     "protobuf",
#     "evaluate",
#     "scikit-learn",
# ]
# ///

"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import logging
import math
import os
import sys

from itertools import chain

import datasets
import evaluate
import torch
from datasets import load_dataset

import transformers
from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    default_data_collator,
    is_torch_xla_available,
    set_seed,
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint, SchedulerType
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from kit.utils import LogParamsCallback
import humanize
from kit.args import ModelArgs, DataArgs, OpinionatedTrainArgs
from kit.ds import split_streaming_dataset

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.57.0")

require_version("datasets>=2.14.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArgs, DataArgs, OpinionatedTrainArgs))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    if training_args.sweep_lr is None:
        
        clm(model_args, data_args, training_args)

    else:
        import copy
        import json
        
        logger.info(f"-" * 80)
        logger.info(f"LR SWEEP MODE: Testing {len(training_args.sweep_lr)} learning rates")
        logger.info(f"Each LR will run for {training_args.sweep_lr_steps} steps")
        logger.info(f"Learning rates to test: {training_args.sweep_lr}")
        logger.info(f"-" * 80)
        
        SWEEP_LR_WARMUP_STEPS = 20

        # keep base copy once before the loop
        base_training_args = copy.deepcopy(training_args)
        base_output_dir = training_args.output_dir
        
        # Notify user about overrides for sweep
        overrides = []
         
        if training_args.do_train is False:
            overrides.append("do_train: False -> True (training enabled during sweep)")
            training_args.do_train = True

        if training_args.max_steps and training_args.max_steps != training_args.sweep_lr_steps:
            overrides.append(f"max_steps: {training_args.max_steps} -> {training_args.sweep_lr_steps}")
            training_args.max_steps = training_args.sweep_lr_steps

        if training_args.do_eval:
            overrides.append(f"do_eval: True -> False (evaluation disabled during sweep)")
            training_args.do_eval = False 

        if training_args.do_predict:
            overrides.append(f"do_predict: True -> False (prediction disabled during sweep)")
            training_args.do_predict = False

        if training_args.save_strategy != "no":
            overrides.append(f"save_strategy: {training_args.save_strategy} -> no")
            training_args.eval_strategy = "no"  # Don't eval during LR sweep

        if training_args.eval_strategy != "no":
            overrides.append(f"eval_strategy: {training_args.eval_strategy} -> no")
            training_args.save_strategy = "no"  # Don't save checkpoints during LR sweep

        if training_args.learning_rate:
            overrides.append(f"learning_rate: will be set by sweep (user value {training_args.learning_rate} ignored)")

        if training_args.warmup_ratio != 0.0:
            overrides.append(f"warmup_ratio: {training_args.warmup_ratio} -> 0.0")
            training_args.warmup_ratio = 0.0
        
        if training_args.warmup_steps == 0:
            overrides.append(f"warmup_steps is not set, forcing: {training_args.warmup_steps} -> {SWEEP_LR_WARMUP_STEPS}")
            training_args.warmup_steps = SWEEP_LR_WARMUP_STEPS

        if training_args.sweep_lr_steps < SWEEP_LR_WARMUP_STEPS:
            overrides.append(f"sweep_lr_steps: {training_args.sweep_lr_steps} < {SWEEP_LR_WARMUP_STEPS}, adjusting warmup_steps to 0")
            training_args.warmup_steps = 0

        if training_args.warmup_steps > 0:
            overrides.append(f"lr_scheduler_type: {training_args.lr_scheduler_type} -> {SchedulerType.CONSTANT_WITH_WARMUP}")
            training_args.lr_scheduler_type = SchedulerType.CONSTANT_WITH_WARMUP
        else:
            overrides.append(f"lr_scheduler_type: {training_args.lr_scheduler_type} -> {SchedulerType.CONSTANT}")
            training_args.lr_scheduler_type = SchedulerType.CONSTANT

        if overrides:
            logger.info("The following arguments will be overridden for sweep:")
            for override in overrides:
                logger.info(f"  - {override}")
            logger.info(f"-" * 80)

        results = []
        
        for idx, each_lr in enumerate(training_args.sweep_lr, 1):
            logger.info(f"{'-'*80}")
            logger.info(f"[{idx}/{len(training_args.sweep_lr)}] Testing LR = {each_lr}")
            logger.info(f"{'-'*80}")
            
            # Create a copy for this specific run from the base
            lr_training_args = copy.deepcopy(training_args)
            lr_training_args.sweep_lr = None       # Prevent recursion
            lr_training_args.learning_rate = each_lr
            lr_training_args.output_dir = f"{base_output_dir}/lr_{each_lr:.1e}"
            lr_training_args.run_name = f"{training_args.run_name}_lr_{each_lr:.1e}"
                       
            # Run training for this LR
            clm(model_args, data_args, lr_training_args)
            
            # Properly close wandb run before next iteration
            # otherwise metrics get mixed up
            try:
                import wandb
                wandb.finish()
            except Exception:
                pass
            
            # Try to read metrics
            metrics_file = f"{lr_training_args.output_dir}/trainer_state.json"
            if os.path.exists(metrics_file):
                with open(metrics_file, 'r') as f:
                    state = json.load(f)
                    # Extract final loss from log history
                    losses = [log.get('loss') for log in state.get('log_history', []) if 'loss' in log]
                    final_loss = losses[-1] if losses else None
                    results.append({
                        'lr': each_lr,
                        'final_loss': final_loss,
                        # 'all_losses': losses
                    })
            else:
                results.append({'lr': each_lr, 'final_loss': None, 'error': 'No metrics found'})
                    
        # Print summary
        logger.info(f"{'-'*80}")
        logger.info("LR SWEEP RESULTS")
        logger.info(f"{'-'*80}")
        logger.info(f"{'Learning Rate':<20} {'Final Loss':<15}")
        logger.info(f"{'-'*35}")
        
        valid_results = [r for r in results if r.get('final_loss') is not None]
        for r in results:
            loss_str = f"{r['final_loss']:.4f}" if r.get('final_loss') else "ERROR"
            logger.info(f"{r['lr']:<20.2e} {loss_str:<15}")
        
        if valid_results:
            best_result = min(valid_results, key=lambda x: x['final_loss'])
            logger.info(f"Best LR: {best_result['lr']:.2e} (loss: {best_result['final_loss']:.4f})")
            
            # Save results to file
            results_file = f"{base_output_dir}/sweep_lr_results.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to: {results_file}")
        
        logger.info(f"{'-'*80}\n")
        return  # Exit after LR sweep


def clm(model_args=None, data_args=None, training_args=None):
    if model_args is None and data_args is None and training_args is None:
        raise ValueError("model_args, data_args, training_args must be provided when calling clm()")
    
    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            streaming=data_args.streaming,
            trust_remote_code=model_args.trust_remote_code,
        )
        if "validation" not in raw_datasets:
            if data_args.streaming:
                dataset_stream = load_dataset(
                    data_args.dataset_name,
                    data_args.dataset_config_name,
                    split="train",
                    cache_dir=model_args.cache_dir,
                    token=model_args.token,
                    streaming=data_args.streaming,
                    trust_remote_code=model_args.trust_remote_code,
                )
                raw_datasets = split_streaming_dataset(dataset_stream, data_args.validation_split_percentage)
            else:
                raw_datasets["validation"] = load_dataset(
                    data_args.dataset_name,
                    data_args.dataset_config_name,
                    split=f"train[:{data_args.validation_split_percentage}%]",
                    cache_dir=model_args.cache_dir,
                    token=model_args.token,
                    streaming=data_args.streaming,
                    trust_remote_code=model_args.trust_remote_code,
                )
                raw_datasets["train"] = load_dataset(
                    data_args.dataset_name,
                    data_args.dataset_config_name,
                    split=f"train[{data_args.validation_split_percentage}%:]",
                    cache_dir=model_args.cache_dir,
                    token=model_args.token,
                    streaming=data_args.streaming,
                    trust_remote_code=model_args.trust_remote_code,
                )
    else:
        data_files = {}
        dataset_args = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        extension = (
            data_args.train_file.split(".")[-1]
            if data_args.train_file is not None
            else data_args.validation_file.split(".")[-1]
        )
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = data_args.keep_linebreaks
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            **dataset_args,
        )
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets:
            if data_args.streaming:
                dataset_stream = load_dataset(
                    extension,
                    data_files=data_files,
                    split="train",
                    cache_dir=model_args.cache_dir,
                    token=model_args.token,
                    **dataset_args,
                )
                raw_datasets = split_streaming_dataset(dataset_stream, data_args.validation_split_percentage)
            else:
                raw_datasets["validation"] = load_dataset(
                    extension,
                    data_files=data_files,
                    split=f"train[:{data_args.validation_split_percentage}%]",
                    cache_dir=model_args.cache_dir,
                    token=model_args.token,
                    **dataset_args,
                )

                raw_datasets["train"] = load_dataset(
                    extension,
                    data_files=data_files,
                    split=f"train[{data_args.validation_split_percentage}%:]",
                    cache_dir=model_args.cache_dir,
                    token=model_args.token,
                    **dataset_args,
                )

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    # --- NOTE: Make model honor block_size ----------------------------------
    if not hasattr(config, "max_position_embeddings"):
        raise ValueError(
            "Model config has no `max_position_embeddings`. "
            "Cannot determine maximum context length."
            ""
        )

    if data_args.block_size:
        if data_args.block_size != config.max_position_embeddings:
            config.max_position_embeddings = data_args.block_size
       
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script. "
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if model_args.model_name_or_path:
        dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
            dtype=dtype,
        )
    else:
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=model_args.trust_remote_code)
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(f"Training new model from scratch - Total size={n_params / 2**20:.2f}M params")
        model.config.n_params = humanize.metric(n_params).replace(" ", "")

    # --- NOTE:Vocab Alignment between Tokenizer and Model --------------------------
    assert tokenizer.eos_token is not None, "Tokenizer must define eos_token"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        # DO NOT DO the following, this creates new id for same token value. 
        # tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

    tokzr_nvocab = len(tokenizer)
    model_nvocab = model.get_input_embeddings().weight.shape[0]
    if tokzr_nvocab != model_nvocab:
        model.resize_token_embeddings(tokzr_nvocab)

    # align regardless
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    # --- End of Vocab Alignment ---------------------------------------

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    # NOTE: tokenizer.model_max_length is not adjusted yet, just so
    # tokenization is rather fixed size for more of the time.
    # we adjust below.
    if training_args.do_train:
        column_names = list(raw_datasets["train"].features)
    else:
        column_names = list(raw_datasets["validation"].features)
    text_column_name = "text" if "text" in column_names else column_names[0]

    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            output = tokenizer(examples[text_column_name])
        # clm input could be much much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                " before being passed to the model."
            )
        return output

    with training_args.main_process_first(desc="dataset map tokenization"):
        if not data_args.streaming:
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
        else:
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                remove_columns=column_names,
            )
    
    # NOTE: At this point
    # model.config.max_position_embeddings is aligned to block size regardless and if set.
    # tokenizer.model_max_length is still OOTB.
    # we align tokenizer.model_max_length to model.config.max_position_embeddings here.
    # block_size is used for chunking below. it must be data_args.block_size if set
    # else must be model.config.max_position_embeddings.
    if tokenizer.model_max_length != model.config.max_position_embeddings:
        tokenizer.model_max_length = model.config.max_position_embeddings

    if data_args.block_size:
        block_size = data_args.block_size
    else:
        block_size = model.config.max_position_embeddings

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/process#map

    with training_args.main_process_first(desc="grouping texts together"):
        if not data_args.streaming:
            lm_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc=f"Grouping texts in chunks of {block_size}",
            )
        else:
            lm_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
            )

    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = lm_datasets["train"]
        if data_args.max_train_samples is not None:
            if data_args.streaming:
                train_dataset = train_dataset.take(data_args.max_train_samples)
            else:
                max_train_samples = min(len(train_dataset), data_args.max_train_samples)
                train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = lm_datasets["validation"]
        if data_args.max_eval_samples is not None:
            if data_args.streaming:
                eval_dataset = eval_dataset.take(data_args.max_eval_samples)
            else:
                max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
                eval_dataset = eval_dataset.select(range(max_eval_samples))

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits.argmax(dim=-1)

        metric = evaluate.load("accuracy", cache_dir=model_args.cache_dir)

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics but we need to shift the labels
            labels = labels[:, 1:].reshape(-1)
            preds = preds[:, :-1].reshape(-1)
            return metric.compute(predictions=preds, references=labels)

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        processing_class=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=default_data_collator,
        compute_metrics=compute_metrics if training_args.do_eval and not is_torch_xla_available() else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if training_args.do_eval and not is_torch_xla_available()
        else None,
    )

    trainer.add_callback(LogParamsCallback(logger))
    
    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        if data_args.streaming:
            metrics["train_samples"] = max_train_samples
        else:
            metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        if data_args.streaming:
            metrics["eval_samples"] = max_eval_samples
        else:
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-generation"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
