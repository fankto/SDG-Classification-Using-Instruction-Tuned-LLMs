import os
import datetime
import os
import re

import fire
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from datasets import load_dataset
from peft import PeftModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer

import wandb
from config import HF_API_KEY, Z_TEST_PATH

def calculate_metrics(ground_truth_np, prediction_np):
    # Calculate metrics
    accuracy = accuracy_score(ground_truth_np, prediction_np)
    precision = precision_score(ground_truth_np, prediction_np, average='macro', zero_division=0)
    recall = recall_score(ground_truth_np, prediction_np, average='macro', zero_division=0)
    f1 = f1_score(ground_truth_np, prediction_np, average='macro')

    report = classification_report(ground_truth_np, prediction_np, output_dict=True)

    return accuracy, f1, precision, recall, report


def record_wandb(decoded_outputs, generated_outputs, sdg_numbers, true_labels):
    output_table = wandb.Table(columns=["decoded_output", "generated_output", "sdg_number", "true_label"],
                               data=[[decoded_outputs, generated_outputs, sdg_numbers, true_labels]])
    wandb.log({"output_table": output_table})

    output_table = pd.DataFrame({"decoded_output": decoded_outputs, "generated_output": generated_outputs,
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


def infer_on_(model, tokenizer, dataset):
    true_labels = []
    predicted_labels = []

    decoded_outputs = []
    generated_outputs = []

    for entry in dataset['test']:
        label = entry['sdg']
        text = entry['abstract']

        input_ids = tokenizer(text, return_tensors='pt').input_ids.cuda()
        output = model.generate(inputs=input_ids, temperature=0.7, max_new_tokens=1024)
        decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

        # remove the intial text from the decoded output
        generated_output = decoded_output[(len(text) - len('<s>')):]
        regex_pattern = r'\b(1[0-7]|[0-9])(\.0)?\b'
        sdg_label = re.search(regex_pattern, generated_output)
        if sdg_label:
            sdg_number = int(sdg_label.group(1))
        else:
            sdg_number = 18

        predicted_labels.append(sdg_number)
        true_labels.append(int(label))
        decoded_outputs.append(decoded_output)
        generated_outputs.append(generated_output)

    if len(true_labels) > 0:
        # calculate the metrics and log them on wandb
        accuracy, f1, precision, recall, report = calculate_metrics(true_labels, predicted_labels)
        wandb.log({"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall, "classification_report": report})
        print(f"Accuracy: {accuracy}, F1: {f1}, Precision: {precision}, Recall: {recall}")
        print("Classification report: ", report)

        # create the confusion matrix and log it on wandb
        record_wandb(decoded_outputs, generated_outputs, predicted_labels, true_labels)

    return true_labels, predicted_labels


def main(model_name="meta-llama/Llama-2-7b-chat-hf", max_seq_length=4096, testset="z", trained_on=None,
         load_in_4bit=True):
    testset_dict = {"z": Z_TEST_PATH}

    data_path = testset_dict[testset.lower()]
    if data_path is None:
        raise ValueError(f"Dataset {testset} not supported.")

    dataset_name = testset
    testset = load_dataset("csv", data_files={
        "test": data_path
    })

    wandb.init(
        project="SDG Classification inference",
        name=f"{model_name} - trained on {trained_on} - tested on {dataset_name}",
        config={
            "architecture": model_name,
            "finetuned on": trained_on,
            "tested on": dataset_name,
            "max_seq_length": max_seq_length,
            "load in 4bit": load_in_4bit
        }
    )

    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        quantization_config = None

    model = AutoModelForCausalLM.from_pretrained(model_name,
                                                 device_map="auto",
                                                 torch_dtype=torch.bfloat16,
                                                 trust_remote_code=True,
                                                 token=HF_API_KEY,
                                                 quantization_config=quantization_config
                                                 )
    if trained_on:
        adapter_name = f"trained_adapters/{model_name}/checkpoint-{trained_on}"
        model = PeftModel.from_pretrained(model, adapter_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True, token=HF_API_KEY)

    true_labels, predicted_labels = infer_on_(model, tokenizer, testset)
    record_confusion_matrix(true_labels, predicted_labels, model_name, trained_on, dataset_name)


if __name__ == "__main__":
    fire.Fire(main)
