

WANDB_PROJECT ?= neo-clm
OUTROOT ?= /root/work/run
CUDADEV ?= 0
DATAROOT ?= /root/work/dataset
ckpt ?= roneneldan/TinyStories-33M

check-postfix:
ifeq ($(postfix),)
	$(error postfix must be provided. Usage: make <target> postfix=something)
endif

eval-tinystories:
	CUDA_VISIBLE_DEVICES=$(CUDADEV) python kit/evals/ts/tinystories_qualitative.py $(ckpt)

gpt2-ts-sweep-lr: check-postfix
	mkdir -p $(OUTROOT)/$(WANDB_PROJECT)/$@-$(postfix) && \
	WANDB_PROJECT=$(WANDB_PROJECT) \
	CUDA_VISIBLE_DEVICES=$(CUDADEV) python run_clm.py \
		--model_type gpt2 --config_overrides n_embd=256,n_layer=8,n_head=16 \
		--tokenizer_name openai-community/gpt2 \
		--dataset_name roneneldan/TinyStories --block_size 512 \
		--per_device_train_batch_size 256 \
		--sweep_lr 1e-3,3e-3,5e-3,8e-3,1e-4,3e-4,5e-4,8e-4,1e-5,3e-5,5e-5,8e-5 \
		--sweep_lr_steps 250 \
		--warmup_steps 30 \
		--run_name $@-$(postfix) \
		--output_dir $(OUTROOT)/$(WANDB_PROJECT)/$@-$(postfix)

# Notes on default: 
# adamw_torch_fused, cosine scheduler, warmup_ratio=0.01, 
# save_total_limit=2, by best eval_loss
# seed and data_seed set by default

gpt2-ts: check-postfix
	mkdir -p $(OUTROOT)/$(WANDB_PROJECT)/$@-$(postfix) && \
	WANDB_PROJECT=$(WANDB_PROJECT) \
	CUDA_VISIBLE_DEVICES=$(CUDADEV) python run_clm.py \
		--model_type gpt2 --config_overrides n_embd=256,n_layer=8,n_head=16 \
		--tokenizer_name openai-community/gpt2 \
		--dataset_name roneneldan/TinyStories --block_size 512 \
		--learning_rate 8e-4 --num_train_epochs 2 \
		--do_train --do_eval \
		--per_device_train_batch_size 256 --per_device_eval_batch_size 256 \
		--eval_steps 200 \
		--save_steps 1000 \
		--run_name $@-$(postfix) \
		--output_dir $(OUTROOT)/$(WANDB_PROJECT)/$@-$(postfix)


gptneo-ts-sweep-lr: check-postfix
	mkdir -p $(OUTROOT)/$(WANDB_PROJECT)/$@-$(postfix) && \
	WANDB_PROJECT=$(WANDB_PROJECT) \
	CUDA_VISIBLE_DEVICES=$(CUDADEV) python run_clm.py \
		--model_type gpt_neo --config_overrides hidden_size=768,num_layers=4,num_heads=16,window_size=256 \
		--tokenizer_name EleutherAI/gpt-neo-125m \
		--dataset_name roneneldan/TinyStories --block_size 512 \
		--per_device_train_batch_size 128 \
		--sweep_lr 5e-3,5e-4,1e-3,2e-3,8e-4,3e-4,3e-3,1e-4,5e-5,1e-5 \
		--sweep_lr_steps 150 \
		--warmup_steps 30 \
		--run_name $@-$(postfix) \
		--output_dir $(OUTROOT)/$(WANDB_PROJECT)/$@-$(postfix)

gptneo-ts: check-postfix
	mkdir -p $(OUTROOT)/$(WANDB_PROJECT)/$@-$(postfix) && \
	WANDB_PROJECT=$(WANDB_PROJECT) \
	CUDA_VISIBLE_DEVICES=$(CUDADEV) python run_clm.py \
		--model_type gpt_neo --config_overrides hidden_size=768,num_layers=4,num_heads=16,window_size=256 \
		--tokenizer_name EleutherAI/gpt-neo-125m \
		--dataset_name roneneldan/TinyStories --block_size 512 \
		--learning_rate 3e-4 --num_train_epochs 2 \
		--do_train --do_eval \
		--per_device_train_batch_size 128 --per_device_eval_batch_size 128 \
		--eval_steps 500 \
		--save_steps 2000 \
		--run_name $@-$(postfix) \
		--output_dir $(OUTROOT)/$(WANDB_PROJECT)/$@-$(postfix)

gptneo-ts-v10k-sweep-lr: check-postfix
	mkdir -p $(OUTROOT)/$(WANDB_PROJECT)/$@-$(postfix) && \
	WANDB_PROJECT=$(WANDB_PROJECT) \
	CUDA_VISIBLE_DEVICES=$(CUDADEV) python run_clm.py \
		--model_type gpt_neo --config_overrides hidden_size=768,num_layers=4,num_heads=16,window_size=256 \
		--tokenizer_name vuiseng9/bpe-10.0k-tinystories \
		--dataset_name roneneldan/TinyStories --block_size 512 \
		--per_device_train_batch_size 128 \
		--sweep_lr 1e-3,3e-3,5e-3,8e-3,1e-4,3e-4,5e-4,8e-4,1e-5,3e-5,5e-5,8e-5 \
		--sweep_lr_steps 150 \
		--warmup_steps 30 \
		--run_name $@-$(postfix) \
		--output_dir $(OUTROOT)/$(WANDB_PROJECT)/$@-$(postfix)

gptneo-ts-v10k: check-postfix
	mkdir -p $(OUTROOT)/$(WANDB_PROJECT)/$@-$(postfix) && \
	WANDB_PROJECT=$(WANDB_PROJECT) \
	CUDA_VISIBLE_DEVICES=$(CUDADEV) python run_clm.py \
		--model_type gpt_neo --config_overrides hidden_size=768,num_layers=4,num_heads=16,window_size=256 \
		--tokenizer_name vuiseng9/bpe-10.0k-tinystories \
		--dataset_name roneneldan/TinyStories --block_size 512 \
		--learning_rate 5e-4 --num_train_epochs 2 \
		--do_train --do_eval \
		--per_device_train_batch_size 128 --per_device_eval_batch_size 128 \
		--eval_steps 500 \
		--save_steps 2000 \
		--run_name $@-$(postfix) \
		--output_dir $(OUTROOT)/$(WANDB_PROJECT)/$@-$(postfix)

# llama-dense: check-postfix
# 	mkdir -p $(OUTROOT)/$(WANDB_PROJECT)/$@-$(postfix) && \
# 	WANDB_PROJECT=$(WANDB_PROJECT) \
# 	CUDA_VISIBLE_DEVICES=$(CUDADEV) python run_clm.py \
# 		--model_type llama \
# 		--config_overrides hidden_size=256,num_hidden_layers=8,num_attention_heads=16,num_key_value_heads=16,head_dim=16,intermediate_size=1024 \
# 		--tokenizer_name meta-llama/Llama-2-7b-hf --use_fast_tokenizer \
# 		--dataset_name roneneldan/TinyStories --block_size 512 \
# 		--preprocessing_num_workers 16 --seed $(SEED) \
# 		--optim adamw_torch_fused --learning_rate 1e-3 --lr_scheduler_type cosine --warmup_ratio 0.01 --num_train_epochs 2 \
# 		--do_train --do_eval --bf16 --torch_compile \
# 		--per_device_train_batch_size 256 --per_device_eval_batch_size 256 \
# 		--eval_strategy steps --eval_steps 200 \
# 		--logging_steps 1 --report_to wandb --project $(WANDB_PROJECT) --run_name $@-$(postfix) \
# 		--save_strategy steps --save_steps 1000 --save_total_limit 2 \
# 		--metric_for_best_model eval_loss --greater_is_better false \
# 		--overwrite_output_dir --output_dir $(OUTROOT)/$(WANDB_PROJECT)/$@-$(postfix)

