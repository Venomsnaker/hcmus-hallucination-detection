import time
import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    roc_auc_score, balanced_accuracy_score, roc_curve, confusion_matrix
)
from sklearn.model_selection import train_test_split

import openai as OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import get_cosine_similarities_mean

class OpenAIEmbeddingClient:
    def __init__(
        self,
        api_key,
        model='text-embedding-3-small',
        retries=3,
    ):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.retries = retries

    def get_embedding(self, input):
        for attempt in range(self.retries):
            try:
                embedding = self.client.embeddings.create(
                    model = self.mode,
                    input = input
                )
                return embedding.data[0].embedding
            except Exception as e:
                if attempt < self.retries - 1:
                    time.sleep(1)
                else:
                    raise e
                
class InterrogateLLM:
    def __init__(
        self,
        api_key,
        model_name="Qwen/Qwen3-4B-Instruct-2507",
        reconstruct_prompt_template_path="",
    ):
        # Recontruction
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype="auto", device_map="auto"
        )
        with open(reconstruct_prompt_template_path) as f:
            self.reconstruct_prompt_template = f.read()

        # Embedding
        self.embedding_client = OpenAIEmbeddingClient(
            api_key=api_key,
            model='text-embedding-3-small'
        )
        
    def _generate(self, prompt: str, max_tokens=1024):
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
    
    def recontruct_prompt(self, context: str, response: str, max_tokens=1024):
        if context == '':
            context = 'None'
        
        prompt = self.reconstruct_prompt_template.format(
            context=context,
            response=response
        )
        return self._generate(prompt, max_tokens=1024)
    
    def measure_similarity(self, question: str, re_questions: str):        
        embedding_question = self.embedding_client.get_embedding(question)
        embedding_re_questions = [self.embedding_client.get_embedding(q) for q in re_questions]
        consine_similary = get_cosine_similarities_mean(embedding_question, embedding_re_questions)
        return float(consine_similary)
    
    def analysis(self, result, plot=True):
        score_thresholds = np.linspace(0, 1, 101)
        test_size = 0.3
        seed = 42
        scores = np.array([sample['consine_similarity'] for sample in result])
        y_true = np.array([sample['is_hallucinated'] for sample in result])

        scores_train, scores_test, y_true_train, y_true_test = train_test_split(
            scores, y_true, test_size=test_size, stratify=y_true, random_state=seed
        )

        # Threshold
        balanced_accuracies = []

        for t in score_thresholds:
            y_pred_train = (scores_train >= t).astype(int)
            balanced_accuracies.append(balanced_accuracy_score(y_true_train, y_pred_train))
        best_balanced_accuracy_score = score_thresholds[np.argmax(balanced_accuracies)] 

        # Analysis
        fpr, tpr, _ = roc_curve(y_true_test, scores_test)

        if plot:
            # ROC Curve
            plt.figure()
            plt.plot(fpr, tpr, lw=2, label='ROC Curve (AUC=%0.2f)' % roc_auc_score(y_true_test, scores_test))
            plt.plot([0,1],[0,1],lw=1, linestyle='--', label='Random Guess')
            plt.xlim=([0.0, 1.0])
            plt.ylim([0,0, 1.0])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend(loc='lower right')
            plt.show()

            # Confusion Matrix
            y_pred_test = (scores_test >= best_balanced_accuracy_score).astype(int)
            cm = confusion_matrix(y_true_test, y_pred_test)
            plt.figure()
            sns.heatmap(
                cm, annot=True, fmt='d', cbar=False,
                xticklabels = ['Pred Neg', 'Pred Pos'],
                yticklabels = ['True Neg', 'True Pos'],
            )
            plt.xlabel('Predict')
            plt.ylabel('True')
            plt.show()

        return {
            'Balanced Accuracy Threshold': best_balanced_accuracy_score,
            'Balanced Accuracy': balanced_accuracy_score(y_true_test, y_pred_test),
            'F1 Score': f1_score(y_true_test, y_pred_test),
            'Accuracy': accuracy_score(y_true_test, y_pred_test),
            'Precision': precision_score(y_true_test, y_pred_test),
            'Recall': recall_score(y_true_test, y_pred_test),
            'AUC (entire)': roc_auc_score(y_true, scores)
        }

