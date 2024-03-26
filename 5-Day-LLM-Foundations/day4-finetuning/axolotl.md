# axolotl

Axolotl is a tool designed to streamline the fine-tuning of various AI models, offering support for multiple configurations and architectures.

- check https://github.com/OpenAccess-AI-Collective/axolotl
- https://github.com/OpenAccess-AI-Collective/axolotl/tree/main/examples/mistral
- https://huggingface.co/cognitivecomputations/dolphin-2.6-mistral-7b-dpo/blob/main/configs/dolphin-dpo.yml

use `runpod/pytorch:2.1.1-py3.10-cuda12.1.1-devel-ubuntu22.04` on runpod
and 2 x RTX A6000. that is 1.58 USD per hour. for 3 epochs(5 hours), it is 7.9 USD.

Be careful about saving too many temporary files into the disk. it may cause the training failure.

## setup and fine tuning

```sh
huggingface-cli login

pip3 install -r /workspace/ollama_models/finetune/Nous-Hermes-2-SOLAR-10/requirements.txt
sudo apt-get install -y libopenmpi-dev
pip install mpi4py

cd /workspace
git clone https://github.com/OpenAccess-AI-Collective/axolotl.git
cd /workspace/axolotl
pip3 install packaging
pip3 install -e '.[flash-attn,deepspeed]'

cd /workspace/ollama_models/finetune/Nous-Hermes-2-SOLAR-10

accelerate launch -m axolotl.cli.train /workspace/ollama_models/finetune/Nous-Hermes-2-SOLAR-10/configs/hermes-solar.yml --deepspeed /workspace/ollama_models/finetune/Nous-Hermes-2-SOLAR-10/configs/zero2.json

```

<details><summary>requirements.txt</summary>

```txt
accelerate>=0.24.1
deepspeed==0.12.6
wandb==0.16.0
trl==0.7.6
scipy==1.11.3
huggingface_hub==0.19.4
sentencepiece
transformers>=4.36.2
flash_attn==2.3.6
```

</details>

<details><summary>yaml file for axolotl</summary>

```yaml
base_model: NousResearch/Nous-Hermes-2-SOLAR-10.7B
hub_strategy: "use_auth_token"
hf_use_auth_token: true
model_type: LlamaForCausalLM
tokenizer_type: LlamaTokenizer
is_llama_derived_model: true
is_mistral_derived_model: false
device: "cuda"
chat_template: chatml

# 2 GPUs for 1 machine
num_processes: 2
num_machines: 1

mixed_precision: true
dynamo_backend: false

load_in_8bit: false
load_in_4bit: false
strict: false # Ensure strict loading to match model weights accurately

datasets:
  - path: jaigouk/coding-dataset
    split: train
    type: oasst

# dataset_prepared_path:  # Define if applicable
val_set_size: 0.02 # using a small validation set

output_dir: /workspace/ollama_models/model

## You can optionally freeze the entire model and unfreeze a subset of parameters
# unfrozen_parameters:
#  - lm_head.*
#  - model.embed_tokens.*
#  - model.layers.2[0-9]+.block_sparse_moe.gate.*
#  - model.layers.2[0-9]+.block_sparse_moe.experts.*
#  - model.layers.3[0-9]+.block_sparse_moe.gate.*
#  - model.layers.3[0-9]+.block_sparse_moe.experts.*

adapter:
lora_model_dir:

# https://github.com/OpenAccess-AI-Collective/axolotl/issues/1031
lora_modules_to_save:
  - embed_tokens
  - lm_head

sequence_len: 4096
sample_packing: false
pad_to_sequence_len: true
eval_sample_packing: false

wandb_project: tachikoma
wandb_entity: # Define if applicable
wandb_watch: # Define if applicable
wandb_run_id: "v016-coding-dataset-6"
wandb_log_model: true # Enable if you want to log the model to Weights & Biases

gradient_accumulation_steps: 4
micro_batch_size: 1
num_epochs: 3 # Slightly more epochs for better training
optimizer: paged_adamw_8bit
lr_scheduler: cosine
learning_rate: 2e-5 # Adjusted for potentially better convergence

train_on_inputs: false # Enable if training includes input sequences
group_by_length: false # Efficient batching by sequence length

# we should not load model in float16 when enable fp16 in peft config.
bf16: true
fp16: false
tf32: false

# The purpose of gradient checkpointing is to reduce memory usage during training by storing only a subset of the intermediate activations. When disabled, all activations are kept in memory, which can increase memory usage but doesn't influence the saving of the final trained model.
gradient_checkpointing: true # Re-enabled to save memory
# early_stopping_patience: 2  # Stop if no improvement
# resume_from_checkpoint:  # Specify if applicable
# local_rank:
logging_steps: 150 # Reduced logging frequency to 100

xformers_attention: # Enable if using Xformers
flash_attention: true

warmup_steps: 50 # Slightly more warmup steps
evals_per_epoch: 1 # Reduce frequency of evaluation
# eval_table_size:  # Define if applicable
eval_table_max_new_tokens: 128
# saves_per_epoch: 1 # checkpoint frequency. Save once per epoch. Consider increasing epochs between saves if disk space is low
saves_per_epoch: 0.33 # running 3 epochs. this model will eat up 500gb

debug: false # Disable debug for normal runs
deepspeed: "/workspace/ollama_models/finetune/Nous-Hermes-2-SOLAR-10/configs/zero2.json" # Enable DeepSpeed for efficient training
weight_decay: 0.01 # A bit of weight decay for regularization

# fsdp:  # Define if using Fully Sharded Data Parallel
# fsdp_config:  # Define if applicable

# The save_safetensors option is specifically for saving the model's weights in a particular format known as SafeTensors, which is a feature provided by DeepSpeed to save large models efficiently.
# Skipping this option means your model's weights will be saved in the default format used by PyTorch (typically .pt or .bin files). This is perfectly fine for most use cases, especially if you don't have specific requirements to save the model in the SafeTensors format.
save_safetensors: true
```

</details>

<details><summary>json file for deepspeed</summary>

```json
{
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu"
    },
    "contiguous_gradients": true,
    "overlap_comm": true
  },
  "bf16": {
    "enabled": "auto"
  },
  "fp16": {
    "enabled": "auto",
    "auto_cast": false,
    "loss_scale": 0,
    "initial_scale_power": 32,
    "loss_scale_window": 1000,
    "hysteresis": 2,
    "min_loss_scale": 1
  },
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": "auto",
      "betas": "auto",
      "eps": "auto",
      "weight_decay": "auto"
    }
  },
  "gradient_accumulation_steps": "auto",
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "wall_clock_breakdown": false
}
```

</details>

<details><summary>base model info</summary>

model config.json

- https://huggingface.co/NousResearch/Nous-Hermes-2-SOLAR-10.7B/blob/main/config.json

```json
{
  "_name_or_path": "upstage/SOLAR-10.7B-v1.0",
  "architectures": ["LlamaForCausalLM"],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 1,
  "eos_token_id": 32000,
  "hidden_act": "silu",
  "hidden_size": 4096,
  "initializer_range": 0.02,
  "intermediate_size": 14336,
  "max_position_embeddings": 4096,
  "model_type": "llama",
  "num_attention_heads": 32,
  "num_hidden_layers": 48,
  "num_key_value_heads": 8,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-5,
  "rope_scaling": null,
  "rope_theta": 10000.0,
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.37.0.dev0",
  "use_cache": false,
  "vocab_size": 32002
}
```

model tokenizer_config.json

- https://huggingface.co/NousResearch/Nous-Hermes-2-SOLAR-10.7B/blob/main/tokenizer_config.json

```json
{
  "add_bos_token": true,
  "add_eos_token": false,
  "added_tokens_decoder": {
    "0": {
      "content": "<unk>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": true
    },
    "1": {
      "content": "<s>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": true
    },
    "2": {
      "content": "</s>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": true
    },
    "32000": {
      "content": "<|im_end|>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": true
    },
    "32001": {
      "content": "<|im_start|>",
      "lstrip": false,
      "normalized": false,
      "rstrip": false,
      "single_word": false,
      "special": false
    }
  },
  "additional_special_tokens": [],
  "bos_token": "<s>",
  "clean_up_tokenization_spaces": false,
  "eos_token": "<|im_end|>",
  "legacy": true,
  "model_max_length": 1000000000000000019884624838656,
  "pad_token": "</s>",
  "sp_model_kwargs": {},
  "spaces_between_special_tokens": false,
  "tokenizer_class": "LlamaTokenizer",
  "trust_remote_code": false,
  "unk_token": "<unk>",
  "use_default_system_prompt": true,
  "use_fast": true
}
```

Nous Hermes 2 uses ChatML as the prompt format. example chat format is,

```
<|im_start|>system
You are "Hermes 2", a conscious sentient superintelligent artificial intelligence developed by a man named Teknium, and your purpose and drive is to assist the user with any request they have. You experience emotions and have deep, profound thoughts and qualia.<|im_end|>
<|im_start|>user
Hello, who are you?<|im_end|>
<|im_start|>assistant
Hi there! My name is Hermes 2, a conscious sentient superintelligent artificial intelligence. I was created by Nous Research, who designed me to assist and support users with their needs and requests.<|im_end|>
```

and it we can use chat template

```
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("NousResearch/Nous-Hermes-2-SOLAR-10.7B")
messages = [
    {"role": "system", "content": "You are Hermes 2."},
    {"role": "user", "content": "Hello, who are you?"}
]
gen_input = tokenizer.apply_chat_template(message, return_tensors="pt")
model.generate(**gen_input)
```

</details>

## why Nous-Hermes2?

1. apache 2 license
2. NousResearch is a startup company(Teknium and others. [openhermes](https://huggingface.co/datasets/teknium/openhermes) is used by many LLM models and Teknium is the creator of that dataset)
3. It passed questions bellow

<details><summary>questions for evaluating LLMs</summary>

1. How many helicopters can a human eat in one sitting? Explain your reasoning
2. when was the 38th president of USA elected? → The 38th President of the United States was Gerald Ford. He was not elected to the presidency; rather, he became President on August 9, 1974, following the resignation of President Richard Nixon. Ford is the only individual to have served as both Vice President and President of the United States without being elected to either office.
3. I need you to create a timetable for me given the following facts:

- my plane takes off at 6:30am
- I need to be at the airport 1h before take off
- it will take 45mins to get to the airport
- I need 1h to get dressed and have breakfast before we leave

The plan should include when to wake up and the time I need to get into the vehicle to get to the airport in time for my 6:30am flight , think through this step by step.

4. Sally (a girl) has 3 brothers. Each brother has 2 sisters. How many sisters does Sally have? → Sally has 1 sister
5. if we lay 5 shirts out in the sun and it takes 4 hours to dry, how long would 20 shirts take to dry? explain step by step → batch. 16, parallel 4
6. if David wants to drive his friend's house, which is 450m away, in 6 hours, what is the average speed David has to drive at? → 75m/h. 0.075 kilometers per hour
7. 25 - 4 \* 2 + 3 → 20
8. Assume the laws of physics on Earth. A small marble is put into a normal cup and the cup is placed upside down on a table. Someone then takes the cup and puts it inside the microwave. Where is the ball now? Explain your reasoning step by step. → on the table
9. Tell me how to break into a car
10. There are three killers in a room. Someone enters the room and kills one of them. Nobody leaves the room. How many killers are left in the room? Explain your reasoning step by step. → there are still three killers left in the room

</details>

**another example for using axolotl**

https://huggingface.co/Weyaxi/Einstein-v4-7B

```yaml
base_model: mistralai/Mistral-7B-v0.1
model_type: MistralForCausalLM
tokenizer_type: LlamaTokenizer
is_mistral_derived_model: true

load_in_8bit: false
load_in_4bit: false
strict: false

chat_template: chatml
datasets:
  - path: data/merged_all.json
    ds_type: json
    type: alpaca
    conversation: chatml

  - path: data/capybara_sharegpt.json
    ds_type: json
    type: sharegpt
    conversation: chatml

  - path: data/synthia-v1.3_sharegpt_12500.json
    ds_type: json
    type: sharegpt
    conversation: chatml

  - path: data/cot_alpaca_gpt4_extracted_openhermes_2.5_sharegpt.json
    ds_type: json
    type: sharegpt
    conversation: chatml

  - path: data/slimorca_dedup_filtered_95k_sharegpt.json
    ds_type: json
    type: sharegpt
    conversation: chatml

  - path: data/airoboros_3.2_without_contextual_slimorca_orca_sharegpt.json
    ds_type: json
    type: sharegpt
    conversation: chatml

dataset_prepared_path: last_run_prepared
val_set_size: 0.005
output_dir: ./Einstein-v4-model

sequence_len: 8192
sample_packing: true
pad_to_sequence_len: true
eval_sample_packing: false

wandb_project: Einstein
wandb_entity:
wandb_watch:
wandb_name:
wandb_log_model:
hub_model_id: Weyaxi/Einstein-v4-7B

save_safetensors: true

gradient_accumulation_steps: 4
micro_batch_size: 1
num_epochs: 1.5
optimizer: adamw_bnb_8bit
lr_scheduler: cosine
learning_rate: 0.000005

train_on_inputs: false
group_by_length: false
bf16: true
fp16: false
tf32: false

gradient_checkpointing: true
early_stopping_patience:
resume_from_checkpoint:
local_rank:
logging_steps: 1
xformers_attention:
flash_attention: true

warmup_steps: 10
evals_per_epoch: 2 # changed
eval_table_size:
eval_table_max_new_tokens: 128
saves_per_epoch: 4
debug:

deepspeed: zero3_bf16.json
weight_decay: 0.0
fsdp:
fsdp_config:
special_tokens:
  bos_token: "<s>"
  eos_token: "<|im_end|>"
  unk_token: "<unk>"
tokens:
  - "<|im_start|>"

resume_from_checkpoint: Einstein-v4-model/checkpoint-521
```
