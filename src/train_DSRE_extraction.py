import os
import pickle

import fire
import pandas as pd
import torch
import wandb
from accelerate import FullyShardedDataParallelPlugin, Accelerator
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp.api import FullOptimStateDictConfig
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, DataCollatorForLanguageModeling
from trl import SFTTrainer, SFTConfig
from datasets import Dataset


def load_model(model_name, bnb_config):
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map="auto",
                                                 low_cpu_mem_usage=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def create_bnb_config():
    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_compute_dtype=torch.bfloat16)


def create_peft_config():
    return LoraConfig(lora_alpha=128, lora_dropout=0.2, r=16, bias="none", task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
                                      "lm_head"])


def train(args):
    model = args["base_model"]
    tokenizer = args["tokenizer"]
    dataset = args["dataset"]
    dataset_name = args["dataset_name"]
    output_dir = args["output_dir"]
    batch_size = args["batch_size"]
    grad_acc_steps = args["grad_acc_steps"]
    max_steps = args["max_steps"]
    learning_rate = args["learning_rate"]
    accelerator = args["accelerator"]

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    peft_config = create_peft_config()
    model = get_peft_model(model, peft_config)

    model.print_trainable_parameters()

    model = accelerator.prepare_model(model)

    sft_config = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_acc_steps,
        learning_rate=learning_rate,
        logging_steps=10,
        max_steps=max_steps,
        report_to="wandb",
        save_total_limit=2,
        bf16=True,
        optim="paged_adamw_8bit"
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        args=sft_config
    )

    model.config.use_cache = False

    trainer.train()

    output_dir_checkpoint = os.path.join(output_dir, f"checkpoint-{dataset_name}")
    trainer.model.save_pretrained(output_dir_checkpoint)

    output_dir_wandb = output_dir_checkpoint.split("trained_adapters/")[1]
    wandb.save(os.path.join(output_dir_wandb, "*"))


def generate_prompt(row, message, replacements, tokenizer):
    replacements['user']['Abstract_Text'] = row['abstract']
    _user_prompt = message[1]['content'].format_map(replacements['user'])
    _system_prompt = message[0]['content'].format_map(replacements['system'])
    _assistant_prompt = message[2]['content'].format_map(replacements['assistant'])

    messages = [{"role": "system", "content": _system_prompt, }, {"role": "user", "content": _user_prompt},
                {"role": "assistant", "content": _assistant_prompt}]

    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def generate_prompts(dataset_df, message, replacements, tokenizer, task):
    data = []

    for i, row in dataset_df.iterrows():
        bsm_response = dataset_df.iloc[i]['final_classification']
        replacements['user']['DSRE_Response'] = bsm_response
        replacements['assistant'] = {}
        replacements['assistant']['answer'] = dataset_df.iloc[i][task]

        data.append(
            {"text": generate_prompt(row, message, replacements, tokenizer), "labels": dataset_df.iloc[i][task]})

    return pd.DataFrame(data)


def main(base_model_name="HuggingFaceH4/zephyr-7b-beta", batch_size=4, max_steps=150, learning_rate=1e-4, epochs=1,
         grad_acc_steps=1, task='single'):
    replacements_default = pickle.load(open("data/prompts/replacements_default.pickle", "rb"))

    generation_args_default = pickle.load(open("data/prompts/generation_args_default.pickle", "rb"))
    generation_args_default['sep_token'] = '<|assistant|>'

    generation_args = generation_args_default.copy()

    generation_args['temperature'] = 0.2
    generation_args['top_k'] = 30
    generation_args['top_p'] = 0.2

    task = task.lower()
    if task == 'single':
        result_message = pickle.load(open("data/prompts/single_sdg_finetuned.pickle", "rb"))
        result_message.append({"role": "assistant", "content": "SDG {answer}"})
    elif task == 'multi':
        result_message = pickle.load(open("data/prompts/all_sdg_finetuned.pickle", "rb"))
        result_message.append({"role": "assistant", "content": "{answer}"})
    else:
        raise ValueError(f"Task {task} not supported.")

    fsdp_plugin = FullyShardedDataParallelPlugin(
        state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
        optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False), )

    accelerator = Accelerator(fsdp_plugin=fsdp_plugin)

    bnb_config = create_bnb_config()
    model, tokenizer = load_model(base_model_name, bnb_config)

    data_df = pd.read_csv('data/extraction_train.csv')
    data = generate_prompts(data_df, result_message, replacements_default, tokenizer, task)

    dataset = Dataset.from_pandas(data)

    if epochs:
        max_steps = int(len(data) / (batch_size * grad_acc_steps) * epochs)

    dataset_name = 'extraction_' + task

    wandb.init(project="SDG Classification training", name=f"{base_model_name} - trained on {dataset_name}",
               config={"learning_rate": learning_rate, "batch_size": batch_size, "architecture": base_model_name,
                       "dataset": dataset_name, "max_steps": max_steps, })

    output_dir = f"trained_adapters/{base_model_name}"
    train({'base_model': model, 'tokenizer': tokenizer, 'dataset': dataset, 'dataset_name': dataset_name,
           'output_dir': output_dir, 'batch_size': batch_size, 'grad_acc_steps': grad_acc_steps, 'max_steps': max_steps,
           'learning_rate': learning_rate, 'accelerator': accelerator})


if __name__ == "__main__":
    fire.Fire(main)
