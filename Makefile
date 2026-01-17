

WANDB_PROJECT ?= neo-clm
OUTROOT ?= /root/work/run
gpulist ?= 0
extra_args ?=
ckpt ?= roneneldan/TinyStories-33M
postfix ?= run0

ifeq ($(sweep_lr),1)
WANDB_PROJECT := sweeplr-$(WANDB_PROJECT)
extra_args += --sweep_lr 1e-3,3e-3,5e-3,8e-3,1e-4,3e-4,5e-4,8e-4,1e-5,3e-5,5e-5,8e-5 --sweep_lr_steps 150 --warmup_steps 30
endif

check-postfix:
ifeq ($(postfix),)
	$(error postfix must be provided. Usage: make <target> postfix=something)
endif

gpulist-check-busy:
ifeq ($(force),1)
	@echo "No gpulist-check-busy (force=1)"
else
	@pids=$$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $(gpulist) 2>&1); \
	if echo "$$pids" | grep -qi "no devices\|invalid\|unable"; then \
		echo "\n\n[Error]: Invalid id(s) in gpulist=$(gpulist). Please revise gpulist=..."; \
		exit 1; \
	elif [ -n "$$pids" ]; then \
		echo "\n\n[Error]: Active job(s) found on gpulist=$(gpulist). Please revise gpulist=... or use force=1"; \
		exit 1; \
	fi
endif

INSTALL_MSG = "[Info:] Please log in W&B (wandb login) and HF (hf auth login) before starting runs"
install-dev:
	pip install -e .

eval-tinystories:
	CUDA_VISIBLE_DEVICES=$(CUDADEV) neoclm-eval-ts $(ckpt)

__pretrain-tinystories: gpulist-check-busy check-postfix
	mkdir -p $(OUTROOT)/$(WANDB_PROJECT)/$(runlabel) && \
	WANDB_PROJECT=$(WANDB_PROJECT) \
	CUDA_VISIBLE_DEVICES=$(gpulist) python run_clm.py \
		$(model_cfg) \
		--dataset_name roneneldan/TinyStories --block_size 512 \
		--learning_rate $(lr) --num_train_epochs 2 \
		--do_train --do_eval \
		--per_device_train_batch_size 128 \
		--per_device_eval_batch_size 128 \
		--eval_steps 500 \
		--save_steps 2000 \
		--logging_steps 1 \
		--run_name $(runlabel) \
		--output_dir $(OUTROOT)/$(WANDB_PROJECT)/$(runlabel) \
		$(extra_args)

# This follows roneneldan/TinyStories-33M, but with full vocab
gptneo-67.3M-ts:
	$(MAKE) __pretrain-tinystories \
	model_cfg="--model_type gpt_neo \
		--config_overrides hidden_size=768,num_layers=4,num_heads=16,window_size=256 \
		--tokenizer_name EleutherAI/gpt-neo-125m" \
	runlabel=$@-$(postfix) lr=3e-4

# This follows roneneldan/TinyStories-8M, but with full vocab
gptneo-19.3M-ts:
	$(MAKE) __pretrain-tinystories \
	model_cfg="--model_type gpt_neo \
		--config_overrides hidden_size=256,num_layers=8,num_heads=16,window_size=256 \
		--tokenizer_name EleutherAI/gpt-neo-125m" \
	runlabel=$@-$(postfix) lr=1e-3

llama2-77.5M-ts: 
	$(MAKE) __pretrain-tinystories \
	model_cfg="--model_type llama \
		--config_overrides hidden_size=768,num_hidden_layers=4,num_attention_heads=16,num_key_value_heads=16,head_dim=48,intermediate_size=2048 \
		--tokenizer_name meta-llama/Llama-2-7b-hf" \
	runlabel=$@-$(postfix) lr=1e-3

## TODO: batch size has changed
gpt2-19.3M-ts:
	$(MAKE) __pretrain-tinystories \
	model_cfg="--model_type gpt2 \
		--config_overrides n_embd=256,n_layer=8,n_head=16 \
		--tokenizer_name openai-community/gpt2" \
	runlabel=$@-$(postfix) lr=1e-3


