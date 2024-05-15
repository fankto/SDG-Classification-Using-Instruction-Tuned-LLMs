import os

import fire
import torch
import wandb
from accelerate import FullyShardedDataParallelPlugin, Accelerator
from datasets import load_dataset, Dataset
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, PeftModel
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp.api import FullOptimStateDictConfig
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, DataCollatorForLanguageModeling
from trl import SFTTrainer, SFTConfig

from config import HF_API_KEY, Z_TRAIN_PATH, ZO_UP_PATH, O_PATH, ZO_PATH


def load_model(model_name, bnb_config):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        token=HF_API_KEY,
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_API_KEY)

    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def create_bnb_config(double_quant):
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=double_quant,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )


def create_peft_config():
    return LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head"
        ]
    )


def train(args):
    model = args["base_model"]
    tokenizer = args["tokenizer"]
    dataset = args["dataset"]
    dataset_name = args["dataset_name"]
    output_dir = args["output_dir"]
    batch_size = args["batch_size"]
    grad_acc_steps = args["grad_acc_steps"]
    max_steps = args["max_steps"]
    max_seq_length = args["max_seq_length"]
    learning_rate = args["learning_rate"]
    accelerator = args["accelerator"]
    double_quant = args["double_quant"]

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    peft_config = create_peft_config()
    model = get_peft_model(model, peft_config)

    model.print_trainable_parameters()

    model = accelerator.prepare_model(model)

    training_args = SFTConfig(
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
        train_dataset=dataset["train"],
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        args=training_args
    )

    model.config.use_cache = False

    trainer.train()

    output_dir_checkpoint = os.path.join(output_dir, f"checkpoint-{dataset_name}-{double_quant}")
    trainer.model.save_pretrained(output_dir_checkpoint)

    output_dir_wandb = output_dir_checkpoint.split("trained_adapters/")[1]
    wandb.save(os.path.join(output_dir_wandb, "*"))

    del model
    del trainer
    torch.cuda.empty_cache()


def generate_prompts(row, system_prompt, user_prompt, tokenizer):
    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': user_prompt.format(Abstract_Text=row['abstract'])},
        {'role': 'assistant', 'content': f"The abstract primarily contributes to SDG {row['sdg']}"}
    ]
    return {
        "text": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    }


def get_system_prompt(sdg_description):
    system_prompt = """You are an expert in scientific research, policy analysis and sustainable development. Determine to which Sustainable Development Goal (SDG) the paper described by the abstract contributes the most.\nThe SDGs are:\n{SDG_List}\n\nExpected Output Format:\nThe abstract primarily contributes to SDG [SDG Number] - [SDG Title].\n\nExample Output:\nThe abstract primarily contributes to SDG 1 - No Poverty"""
    return system_prompt.format(SDG_List=sdg_description)


def get_user_prompt():
    user_prompt = """Analyze the abstract and determine to which Sustainable Development Goal (SDG) the paper described by the abstract contributes the most.\nAbstract and Title:\n\"{Abstract_Text}\""""
    return user_prompt


def main(base_model_name="HuggingFaceH4/zephyr-7b-beta", train_on='z', load_in_4bit=True, double_quant=True, checkpoint_name=None, batch_size=4,
         max_steps=150, max_seq_length=8192, learning_rate=1e-4, epochs=1, grad_acc_steps=1):
    sdg_description = open("data/prompts/SDG_Description.txt", "r").read()
    system_prompt = get_system_prompt(sdg_description)
    user_prompt = get_user_prompt()

    dataset_paths = {
        'z': Z_TRAIN_PATH,
        'o': O_PATH,
        'zo': ZO_PATH,
        'zo_up': ZO_UP_PATH
    }

    dataset_path = dataset_paths[train_on]

    if dataset_path is None:
        raise ValueError(f"Dataset {train_on} not supported.")

    dataset = load_dataset("csv", data_files=dataset_path, split="train")
    dataset = dataset.shuffle(seed=42)

    fsdp_plugin = FullyShardedDataParallelPlugin(
        state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
        optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
    )

    accelerator = Accelerator(fsdp_plugin=fsdp_plugin)

    if load_in_4bit:
        bnb_config = create_bnb_config(double_quant)
        model, tokenizer = load_model(base_model_name, bnb_config)
    else:
        model, tokenizer = load_model(base_model_name, None)

    dataset = dataset.map(
        generate_prompts,
        fn_kwargs={
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "tokenizer": tokenizer
        }
    )

    hf_dataset = Dataset.from_dict({'text': [example["text"] for example in dataset]})
    dataset_name = train_on
    train_on = {"train": hf_dataset}

    print(train_on["train"][0])

    if checkpoint_name:
        checkpoint_name = f"trained_adapters/{base_model_name}/{checkpoint_name}"
        model = PeftModel.from_pretrained(model, checkpoint_name)

    if epochs:
        max_steps = int(len(train_on["train"]) / (batch_size * grad_acc_steps) * epochs)

    wandb.init(
        project="SDG Classification training",
        name=f"{base_model_name} - trained on {dataset_name}",
        config={
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "architecture": base_model_name,
            "dataset": dataset_name,
            "max_steps": max_steps,
            "max_seq_length": max_seq_length
        }
    )

    output_dir = f"trained_adapters/{base_model_name}"
    train({'base_model': model,
           'tokenizer': tokenizer,
           'dataset': train_on,
           'dataset_name': dataset_name,
           'output_dir': output_dir,
           'batch_size': batch_size,
           'grad_acc_steps': grad_acc_steps,
           'max_steps': max_steps,
           'max_seq_length': max_seq_length,
           'learning_rate': learning_rate,
           'accelerator': accelerator,
           'double_quant': double_quant})


if __name__ == "__main__":
    fire.Fire(main)
