from typing import List
import numpy as np
import spacy

import torch
import bert_score


class SelfCheckBERTScore:
    def __init__(self, default_model="en", rescale_with_baseline=True):
        self.nlp = spacy.load("en_core_web_sm")
        self.default_model = default_model
        self.rescale_with_baseline = rescale_with_baseline

    @torch.no_grad()
    def predict_hallucination(self, sentences: List[str], samples: List[str]):
        n_sentences = len(sentences)
        n_samples = len(samples)
        scores = np.zeros((n_sentences, n_samples))

        for i, sample in enumerate(samples):
            sample_sentences = [sent for sent in self.nlp(sample).sents]
            sample_sentences = [
                sent.text.strip() for sent in sample_sentences if len(sent) > 3
            ]
            n_sample_sentences = len(sample_sentences)
            refs = [x for x in sentences for _ in range(n_sample_sentences)]
            cands = sample_sentences * n_sentences

            p, r, f1 = bert_score.score(
                cands,
                refs,
                lang=self.default_model,
                verbose=False,
                rescale_with_baseline=self.rescale_with_baseline,
            )
            arr_f1 = f1.reshape(n_sentences, n_sample_sentences)
            arr_f1_max_axis1 = arr_f1.max(axis=1).values.numpy()
            scores[:, i] = arr_f1_max_axis1
        scores = 1 - scores.mean(axis=-1)
        return scores.tolist()
