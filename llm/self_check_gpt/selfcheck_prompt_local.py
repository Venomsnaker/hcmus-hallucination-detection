from typing import List
from tqdm import tqdm

import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer


class SelfCheckPromptLocal:
    def __init__(
        self,
        model_name="Qwen/Qwen3-4B-Instruct-2507",
        prompt_template_path="",
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype="auto", device_map="auto"
        )
        with open(prompt_template_path) as f:
            self.prompt_template = f.read()
        self.output_mapping = {"no": 1.0, "n/a": 0.5, "yes": 0.0}
        self.not_defined_verdict = set()

    def _postprocess_verdict(
        self,
        verdict,
    ):
        verdict = verdict.lower().strip()

        if "yes" in verdict:
            verdict = "yes"
        elif "no" in verdict:
            verdict = "no"
        else:
            if verdict not in self.not_defined_verdict:
                print(f"WARNING: '{verdict}' not defined.")
                self.not_defined_verdict.add(verdict)
            verdict = "n/a"
        return self.output_mapping[verdict]

    def _generate_verdict(self, prompt: str, max_tokens=5):
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(**model_inputs, max_new_tokens=max_tokens)
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
        content = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        return content

    def set_prompt_template(self, prompt_template: str):
        self.prompt_template = prompt_template

    def predict_hallucination(
        self, sentences: List[str], sample_responses: List[str], verbose: bool = False
    ):
        n_sentences = len(sentences)
        n_sample_responses = len(sample_responses)
        scores = np.zeros((n_sentences, n_sample_responses))

        for i in tqdm(range(n_sentences), disable=not verbose):
            sentence = sentences[i]

            for j, sample_response in enumerate(sample_responses):
                sample_response = sample_response.strip()
                verdict_prompt = self.prompt_template.format(
                    context=sample_response, sentence=sentence
                )
                verdict = self._generate_verdict(verdict_prompt)
                scores[i, j] = self.postprocess_verdict(verdict)
        score_per_sentence = scores.mean(axis=-1)
        return score_per_sentence.tolist()
