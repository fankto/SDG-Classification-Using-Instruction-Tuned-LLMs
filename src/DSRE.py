import os
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist
import pandas as pd
import torch
from pathlib import Path

embeddings_path = Path('../data/embeddings')


class SDGAbstractsAnalyzer:
    def __init__(self, gold_label_data_df):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-base-en-v1.5')
        self.model = AutoModel.from_pretrained('BAAI/bge-base-en-v1.5').to(self.device)
        self.gold_label_data_df = gold_label_data_df
        embeddings_file = embeddings_path / 'gold_embeddings.pt'
        if embeddings_file.is_file():
            self.gold_embeddings = torch.load(str(embeddings_file))
            print("Loaded gold embeddings from file")
        else:
            os.makedirs(embeddings_path, exist_ok=True)
            self.gold_embeddings = self._embed_abstracts(gold_label_data_df['abstract'].tolist())
            torch.save(self.gold_embeddings, str(embeddings_file))
            print("Saved gold embeddings to file")
        self.sdg_centers = self._calculate_sdg_centers()

    def _embed_abstracts(self, abstracts, batch_size=128):
        all_embeddings = torch.tensor([], device=self.device)

        for i in range(0, len(abstracts), batch_size):
            batch_abstracts = abstracts[i:i+batch_size]
            encoded_input = self.tokenizer(batch_abstracts, padding=True, truncation=True, return_tensors='pt')
            encoded_input = {key: val.to(self.device) for key, val in encoded_input.items()}  # Moving input to the correct device

            with torch.no_grad():
                model_output = self.model(**encoded_input)

            cls_embeddings = model_output[0][:, 0]
            normalized_embeddings = torch.nn.functional.normalize(cls_embeddings, p=2, dim=1)

            all_embeddings = torch.cat((all_embeddings, normalized_embeddings), dim=0)

        return all_embeddings


    def _calculate_sdg_centers(self):
        sdg_centers = {}
        for sdg, group in self.gold_label_data_df.groupby('sdg'):
            sdg_embeddings = self.gold_embeddings[group.index]
            sdg_center = sdg_embeddings.mean(dim=0)
            sdg_centers[sdg] = sdg_center
        return sdg_centers

    def find_closest_abstracts_and_sdg(self, new_abstract, n_closest=3):
        new_embedding = self._embed_abstracts([new_abstract]).cpu().numpy()
        gold_embeddings_np = self.gold_embeddings.cpu().numpy()
        distances = cdist(new_embedding, gold_embeddings_np, 'cosine')[0]
        closest_indices = distances.argsort()[:n_closest]
        closest_abstracts = self.gold_label_data_df.iloc[closest_indices]
        sdg_distances = {sdg: cosine_similarity(new_embedding, sdg_center.unsqueeze(0).cpu().numpy())[0][0] for
                         sdg, sdg_center in self.sdg_centers.items()}
        closest_sdg = max(sdg_distances, key=sdg_distances.get)
        return closest_abstracts, closest_sdg

    def closest_abstract_sdg_as_string(self, closest_abstracts):
        closest_abstract_sdgs = set(closest_abstracts['sdg'].tolist())
        closest_abstract_sdgs = sorted(closest_abstract_sdgs)
        closest_abstract_sdgs = ', '.join(str(sdg) for sdg in closest_abstract_sdgs)
        return closest_abstract_sdgs

    def regenerate_embeddings(self):
        self.gold_embeddings = self._embed_abstracts(self.gold_label_data_df['abstract'].tolist())
        torch.save(self.gold_embeddings, '../data/embeddings/gold_embeddings.pt')
        self.sdg_centers = self._calculate_sdg_centers()


from transformers import BitsAndBytesConfig
from src.config import HF_API_KEY
from transformers import AutoModelForCausalLM
from transformers import pipeline
import time


class BranchSolver:
    def __init__(self, name, pipe, message, replacements, generation_args):
        self.name = name
        self.pipe = pipe
        self.message = message
        self.replacements = replacements
        self.generation_args = generation_args
        self.results = pd.DataFrame()
        self.start_time = None

    def _generate(self, row, _system_prompt):
        self.replacements['user']['Abstract_Text'] = row['abstract']
        _user_prompt = self.message[1]['content'].format_map(self.replacements['user'])

        messages = [
            {
                "role": "system",
                "content": _system_prompt,
            },
            {
                "role": "user",
                "content": _user_prompt
            }
        ]

        prompt = self.pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        input_ids = self.pipe.tokenizer.encode(prompt, return_tensors="pt").to(self.pipe.device)

        outputs = self.pipe.model.generate(
            input_ids,
            max_new_tokens=self.generation_args['max_new_tokens'],
            do_sample=self.generation_args['do_sample'],
            temperature=self.generation_args['temperature'],
            top_k=self.generation_args['top_k'],
            top_p=self.generation_args['top_p'],
            num_return_sequences=self.generation_args['num_return_sequences']
        )

        generation_time = time.time() - self.start_time
        generated_text = self.pipe.tokenizer.decode(outputs[0], skip_special_tokens=True)

        expected_sdg = row['gold_label'] if 'gold_label' in row else row['expected sdg']
        new_data = {
            'id': row['id'],
            'abstract': row['abstract'],
            'expected sdg': expected_sdg,
            f'{self.name} Prompt': _system_prompt + _user_prompt,
            f'{self.name} Generated Text': generated_text.split(self.generation_args['sep_token'])[1],
            f'{self.name} Generation Time': generation_time
        }
        new_row = pd.DataFrame(new_data, index=[0])
        self.results = pd.concat([self.results, new_row], ignore_index=True)

    def _generate_text(self, data):
        _system_prompt = self.message[0]['content'].format_map(self.replacements['system'])

        for i, row in data.iterrows():
            self.start_time = time.time()
            self._generate(row, _system_prompt)

    def solve(self, data):
        self._generate_text(data)
        return self.results

    def _generate_with_follow_up(self, row, _system_prompt, follow_up_prompt):
        self._generate(row, _system_prompt)

        start_time = time.time()
        follow_up_message = {
            "role": "user",
            "content": follow_up_prompt
        }

        messages = [
            {"role": "system", "content": _system_prompt},
            {"role": "user", "content": self.message[1]['content'].format_map(self.replacements['user'])},
            {"role": "assistant", "content": self.results.iloc[-1][f'{self.name} Generated Text']},
            follow_up_message
        ]
        follow_up_prompt_formatted = self.pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        input_ids_follow_up = self.pipe.tokenizer.encode(follow_up_prompt_formatted, return_tensors="pt").to(self.pipe.device)
        outputs_follow_up = self.pipe.model.generate(
            input_ids_follow_up,
            max_new_tokens=self.generation_args['max_new_tokens'],
            do_sample=self.generation_args['do_sample'],
            temperature=self.generation_args['temperature'],
            top_k=self.generation_args['top_k'],
            top_p=self.generation_args['top_p'],
            num_return_sequences=self.generation_args['num_return_sequences']
        )

        follow_up_generated_text = self.pipe.tokenizer.decode(outputs_follow_up[0], skip_special_tokens=True)

        end_time = time.time()
        generated_text = follow_up_generated_text.split(self.generation_args['sep_token'])[1]
        generation_time = end_time - start_time

        follow_up_prompt_full = '\n'.join([message['content'] for message in messages])

        self.results.loc[self.results.index[-1], f'{self.name} Follow Up Prompt'] = follow_up_prompt_full
        self.results.loc[self.results.index[-1], f'{self.name} Follow Up Response'] = generated_text
        self.results.loc[self.results.index[-1], f'{self.name} Follow Up Generation Time'] = generation_time


class BranchSolverEmbedding(BranchSolver):
    def __init__(self, name, pipe, message, replacements, generation_args, gold_label_data_df):
        super().__init__(name, pipe, message, replacements, generation_args)
        self.SDGAbstractsAnalyzer = SDGAbstractsAnalyzer(gold_label_data_df)

    def _generate_text(self, data):
        _system_prompt = self.message[0]['content'].format_map(self.replacements['system'])

        for i, row in data.iterrows():
            self.start_time = time.time()
            closest_abstracts, closest_sdg = self.SDGAbstractsAnalyzer.find_closest_abstracts_and_sdg(
                data.iloc[0]['abstract'])
            if closest_sdg in closest_abstracts['sdg'].tolist():
                closest_sdg = ""
            closest_abstract_sdgs = self.SDGAbstractsAnalyzer.closest_abstract_sdg_as_string(closest_abstracts)
            self.replacements['user']['closest_sdg_centroid'] = closest_sdg
            self.replacements['user']['closest_abstract_sdgs'] = closest_abstract_sdgs
            self._generate(row, _system_prompt)

    def regenerate_embeddings(self):
        self.SDGAbstractsAnalyzer.regenerate_embeddings()


class MergeModule(BranchSolver):
    def __init__(self, name, pipe, message, replacements, generation_args):
        super().__init__(name, pipe, message, replacements, generation_args)
        self.branches = []

    def _generate_text(self, data, results):
        _system_prompt = self.message[0]['content'].format_map(self.replacements['system'])
        for i, row in data.iterrows():
            self.start_time = time.time()
            self.replacements['user']['Branches_Text'] = '\n\n'.join(
                [branch.name + ": " + results[branch.name].iloc[i][branch.name + " Generated Text"] + "\n" for branch in
                 self.branches])
            self._generate(row, _system_prompt)

    def solve(self, data, results):
        self._generate_text(data, results)
        return self.results

    def set_brancheSolvers(self, branches):
        self.branches = branches


class ResultBranchSolver(BranchSolver):
    def __init__(self, name, pipe, message, replacements, generation_args, full_resulst_df):
        super().__init__(name, pipe, message, replacements, generation_args)
        self.full_resulst_df = full_resulst_df

    def _generate_text(self, data):
        _system_prompt = self.message[0]['content'].format_map(self.replacements['system'])

        for i, row in data.iterrows():
            self.start_time = time.time()
            expected_sdg = self.full_resulst_df.iloc[i]['expected sdg']
            bsm_response = self.full_resulst_df.iloc[i]['Merge Module Generated Text']
            self.replacements['user']['sdg'] = expected_sdg
            self.replacements['user']['DSRE_Response'] = bsm_response
            self._generate(row, _system_prompt)


class ExplainationExtractorBranchSolver(BranchSolver):
    def __init__(self, name, pipe, message, replacements, generation_args, full_resulst_df):
        super().__init__(name, pipe, message, replacements, generation_args)
        self.full_resulst_df = full_resulst_df

    def _generate_text(self, data):
        _system_prompt = self.message[0]['content'].format_map(self.replacements['system'])

        for i, row in data.iterrows():
            self.start_time = time.time()
            expected_sdgs = self.full_resulst_df.iloc[i]['selected sdgs']
            bsm_response = self.full_resulst_df.iloc[i]['Merge Module Generated Text']
            self.replacements['user']['SDG_List'] = expected_sdgs
            self.replacements['user']['DSRE_Response'] = bsm_response
            self._generate(row, _system_prompt)


class AbstractClassifier:
    def __init__(self, base_model_name="meta-llama/Llama-2-7b-chat-hf"):
        self.base_model = None
        self.pipe = None
        self.tokenizer = None
        self._load_model(base_model_name)
        self.brancheSolvers = []
        self.mergeModule = None

    def _load_model(self, base_model_name):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )

        self.base_model = AutoModelForCausalLM.from_pretrained(base_model_name,
                                                               device_map="auto",
                                                               torch_dtype=torch.bfloat16,
                                                               trust_remote_code=True,
                                                               token=HF_API_KEY,
                                                               quantization_config=quantization_config,
                                                               load_in_4bit=True,
                                                               low_cpu_mem_usage=True
                                                               )
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True, trust_remote_code=True,
                                                       token=HF_API_KEY)
        self.pipe = pipeline("text-generation", model=self.base_model, tokenizer=self.tokenizer,
                             torch_dtype=torch.bfloat16, device_map="auto")

    def add_branch_solver(self, branch_solver):
        self.brancheSolvers.append(branch_solver)

    def clear_branch_solvers(self):
        self.brancheSolvers = []

    def add_merge_module(self, merge_module):
        self.MergeModule = merge_module
        self.MergeModule.set_brancheSolvers(self.brancheSolvers)

    def add_evaluation_branch_solver(self, evaluation_branch_solver):
        self.evaluation_branch_solver = evaluation_branch_solver

    def solve(self, data):
        results = {}
        for branch_solver in self.brancheSolvers:
            results[branch_solver.name] = branch_solver.solve(data)

        self.MergeModule.solve(data, results)

    def create_full_results(self):
        full_df = self.brancheSolvers[0].results
        for branch_solver in self.brancheSolvers[1:]:
            full_df = full_df.merge(branch_solver.results, on=['id', 'abstract', 'expected sdg'])
        full_df = full_df.merge(self.MergeModule.results, on=['id', 'abstract', 'expected sdg'])

        branch_names = [branch_solver.name for branch_solver in self.brancheSolvers] + [self.MergeModule.name]

        full_df['total_time'] = full_df[[f'{branch_name} Generation Time' for branch_name in branch_names]].sum(axis=1)

        return full_df

    def evaluate(self, data):
        self.evaluation_branch_solver.solve(data)
        return self.evaluation_branch_solver.results
