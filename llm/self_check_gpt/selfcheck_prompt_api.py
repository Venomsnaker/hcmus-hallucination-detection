import time
from typing import List
from tqdm import tqdm

import numpy as np

from openai import OpenAI


class SelfCheckPromptAPI:
    def __init__(
        self,
        model="gpt-4o-mini-2024-07-18",
        api_key="",
        prompt_template_path="",
        retries=3,
    ):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        with open(prompt_template_path) as f:
            self.prompt_template = f.read()
        self.output_mapping = {
            "no": 1.0,
            "n/a": 0.5,
            "yes": 0.0,
        }
        self.not_defined_verdict = set()
        self.retries = retries

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
        for attempt in range(self.retries):
            try:
                request_kwargs = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_completion_tokens": max_tokens,
                }

                response = self.client.chat.completions.create(**request_kwargs)
                return response.choices[0].message.content
            except Exception as e:
                if attempt + 1 < self.retries:
                    time.sleep(1)
                else:
                    raise e

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
                scores[i, j] = self._postprocess_verdict(verdict)
        score_per_sentence = scores.mean(axis=-1)
        return score_per_sentence.tolist()
