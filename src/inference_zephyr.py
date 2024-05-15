import os
import datetime
import os
import re
import time

import fire
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from datasets import load_dataset
from peft import PeftModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, \
    classification_report
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, pipeline, GenerationConfig

import wandb
from config import Z_TEST_PATH

if torch.cuda.is_available():
    print("CUDA is available. Running on GPU.")
else:
    print("CUDA not available. Running on CPU.")

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)


def calculate_metrics(ground_truth_np, prediction_np):
    # Calculate metrics
    accuracy = accuracy_score(ground_truth_np, prediction_np)
    precision = precision_score(ground_truth_np, prediction_np, average='weighted', zero_division=0)
    recall = recall_score(ground_truth_np, prediction_np, average='weighted', zero_division=0)
    f1 = f1_score(ground_truth_np, prediction_np, average='weighted')

    report = classification_report(ground_truth_np, prediction_np, output_dict=True)

    return accuracy, f1, precision, recall, report


def record_wandb(decoded_outputs, generated_outputs, sdg_numbers, true_labels, ids):
    output_table = wandb.Table(columns=["ids", "decoded_output", "generated_output", "sdg_number", "true_label"],
                               data=[[ids, decoded_outputs, generated_outputs, sdg_numbers, true_labels]])
    wandb.log({"output_table": output_table})

    output_table = pd.DataFrame({"ids": ids, "decoded_output": decoded_outputs, "generated_output": generated_outputs,
                                 "sdg_number": sdg_numbers, "true_label": true_labels})
    date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # create the folder if it does not exist
    if not os.path.exists("results/finetuning/results"):
        os.makedirs("results/finetuning/results")
    if not os.path.exists("results/finetuning/results/{run_name}".format(run_name=wandb.run.name)):
        os.makedirs("results/finetuning/results/{run_name}".format(run_name=wandb.run.name))
    output_table.to_csv("results/finetuning/results/{run_name}/{date}.csv".format(run_name=wandb.run.name, date=date))


def record_confusion_matrix(true_labels, predicted_labels, model_name, trained_on, dataset_name):
    # save the confustion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(16, 16))
    # the axis labels should be the sdg number as well as the sdg name
    sdg_names = {1: "No Poverty", 2: "Zero Hunger", 3: "Good Health and Well-being", 4: "Quality Education",
                 5: "Gender Equality", 6: "Clean Water and Sanitation", 7: "Affordable and Clean Energy",
                 8: "Decent Work and Economic Growth", 9: "Industry, Innovation and Infrastructure",
                 10: "Reduced Inequalities", 11: "Sustainable Cities and Communities",
                 12: "Responsible Consumption and Production",
                 13: "Climate Action", 14: "Life Below Water", 15: "Life on Land",
                 16: "Peace and Justice Strong Institutions",
                 17: "Partnerships to achieve the Goal", 18: "No SDG"}
    sdg_names = {k: f"{k}: {v}" for k, v in sdg_names.items()}
    # sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=range(1, 19), yticklabels=range(1, 18))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=sdg_names.values(), yticklabels=sdg_names.values())
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion matrix for {model_name} trained on {trained_on} and tested on {dataset_name}")

    # Save plot to disk
    plt.savefig("temp_cm.png")

    # Log the confusion matrix on wandb
    wandb.log({"confusion_matrix": wandb.Image("temp_cm.png")})

    # Delete temporary file
    os.remove("temp_cm.png")
    plt.close()


def generate_prompts(row, system_prompt, user_prompt, tokenizer):
    messages = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': user_prompt.format(Abstract_Text=row['abstract'])},
        {'role': 'assistant',
         'content': f"The abstract primarily contributes to SDG {row['sdg']}"}
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


def infer_on_(pipe, dataset, counter, data_path, generation_config):
    true_labels = []
    predicted_labels = []

    decoded_outputs = []
    generated_outputs = []

    ids = []

    sdg_description = open("data/prompts/SDG_Description.txt", "r").read()
    system_prompt = get_system_prompt(sdg_description)
    user_prompt = get_user_prompt()

    batch_size = 16

    for i in range(counter, len(dataset), batch_size):
        batch = dataset[i:i + batch_size]
        prompts = []
        labels = []

        start_time = time.time()
        for _, row in batch.iterrows():
            _user_prompt = user_prompt.format(Abstract_Text=row["abstract"])

            messages = [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {"role": "user", "content": _user_prompt},
            ]

            label = row['sdg']

            prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(prompt)
            labels.append(label)

        try:
            outputs = pipe(prompts, generation_config=generation_config)
        except Exception as e:
            print(f"Error encountered in batch starting at index {i}: {e}")
            print("Skipping this batch and moving to the next.")
            continue  # Skip the rest of the loop and move to the next batch

        for output, label, row in zip(outputs, labels, batch.iterrows()):
            # remove the intial text from the decoded output
            sep_token = '<|assistant|>' if '<|assistant|>' in output[0]['generated_text'] else '[/INST]'
            generated_output = output[0]['generated_text'].split(sep_token)[1]
            regex_pattern = r'\b(1[0-7]|[0-9])(\.0)?\b'
            sdg_label = re.search(regex_pattern, generated_output)
            if sdg_label:
                sdg_number = int(sdg_label.group(1))
            else:
                sdg_number = 18

            predicted_labels.append(sdg_number)
            true_labels.append(int(label))
            decoded_outputs.append(output)
            generated_outputs.append(generated_output)
            ids.append(row[1]['id'])
            dataset.loc[row[0], 'prediction'] = sdg_number

            # save it to file
            try:
                dataset.to_csv(data_path, index=False)
            except OSError:
                os.makedirs(os.path.dirname(data_path))
                dataset.to_csv(data_path, index=False)

        print(f"Time for one batch, size {batch_size}: {time.time() - start_time} seconds")

    if len(true_labels) > 0:
        # calculate the metrics and log them on wandb
        accuracy, f1, precision, recall, report = calculate_metrics(true_labels, predicted_labels)
        wandb.log({"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall, "classification_report": report})
        print(f"Accuracy: {accuracy}, F1: {f1}, Precision: {precision}, Recall: {recall}")
        print("Classification report:", report)

        # create the confusion matrix and log it on wandb
        record_wandb(decoded_outputs, generated_outputs, predicted_labels, true_labels, ids)

    return true_labels, predicted_labels


def main(model_name="HuggingFaceH4/zephyr-7b-beta", max_seq_length=4096, testset="zora", trained_on=None,
         load_in_4bit=True, double_quant=True):
    start_time = time.time()
    testset_dict = {"zora": Z_TEST_PATH,
                    }

    data_path = testset_dict[testset.lower()]
    if data_path is None:
        raise ValueError(f"Dataset {testset} not supported.")

    dataset_name = testset
    test_data = pd.read_csv(data_path)

    # check if test_data already contains a column 'prediction', else add it
    if 'prediction' not in test_data.columns:
        test_data['prediction'] = None
        counter = 0
    else:
        # set the counter to the index of the first row that does not have a prediction yet
        counter = test_data[test_data['prediction'].isnull()].index[0]

    print(f"Starting inference on {len(test_data)} samples, starting at index {counter}.")

    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=double_quant,
            bnb_4bit_quant_type='nf4'
        )
    else:
        quantization_config = None
        double_quant = False

    wandb.init(
        project="SDG Classification inference",
        name=f"{model_name} - trained on {trained_on} - tested on {dataset_name}",
        config={
            "architecture": model_name,
            "finetuned on": trained_on,
            "tested on": dataset_name,
            "max_seq_length": max_seq_length,
            "load in 4bit": load_in_4bit,
            "double quant": double_quant
        }
    )

    generation_config = GenerationConfig(
        max_new_tokens=32,
        temperature=0.7,
        top_k=20,
        top_p=0.95,
        do_sample=True
    )

    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 device_map="auto",
                                                 torch_dtype=torch.bfloat16,
                                                 trust_remote_code=True,
                                                 quantization_config=quantization_config,
                                                 )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, torch_dtype=torch.bfloat16, device_map="auto")
    if trained_on:
        adapter_name = f"trained_adapters/{model_name}/checkpoint-{trained_on}"
        model = PeftModel.from_pretrained(model, adapter_name)
        pipe.model = model

    date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print("Pipeline device:", pipe.device)
    print(f"Time for Setup: {time.time() - start_time} seconds")
    true_labels, predicted_labels = infer_on_(pipe, test_data, counter,
                                              f"data/predictions/{model_name}{dataset_name}_{date}.csv", generation_config)
    record_confusion_matrix(true_labels, predicted_labels, model_name, trained_on, dataset_name)


if __name__ == "__main__":
    fire.Fire(main)

# %%
