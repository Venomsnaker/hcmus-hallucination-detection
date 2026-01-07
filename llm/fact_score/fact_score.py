from transformers import AutoModelForCausalLM, AutoTokenizer

class FactScoreRetriever:
    def __init__(
        self,
        model_name:str = "Qwen/Qwen3-4B-Instruct-2507",
        retrieve_user_prompt_path: str = "",
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype="auto", device_map="auto"
        )
        with open(retrieve_user_prompt_path) as f:
            self.retrieve_user_prompt_template = f.read()
        
    def _generate(self, user_prompt: str, max_tokens=2056):
        messages = [
            {"role": "user", "content": user_prompt}
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self.tokenizer([inputs], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(**model_inputs, max_new_tokens=max_tokens)
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
        ouputs = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        return ouputs
    
    def _postprocess_facts(facts: str):
        res = []
        
        for fact in facts.split('\n'):
            fact = fact.strip()
            if fact:
                sub_facts = [f.strip() for f in fact.split('.') if f.strip()] 
                res.extend(sub_facts)
        return res
        
    def retrieve_facts(self, response: str):
        prompt = self.retrieve_user_prompt_template.format(response=response)
        facts = self._generate(user_prompt=prompt, max_tokens=2056)
        return self._postprocess_facts(facts)
        
        
class FactScoreVerifier:
    def __init__(
        self,
        model_name:str = "Qwen/Qwen3-4B-Instruct-2507",
        verfiy_user_prompt_path: str = "",
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype="auto", device_map="auto"
        )
        with open(verfiy_user_prompt_path) as f:
            self.verify_user_prompt_template = f.read()
        self.output_mapping = {
            'yes': 0,
            'n/a': 0.5,
            'no': 1
        }
            
    def _generate(self, user_prompt: str, max_tokens=4):
        messages = [
            {"role": "user", "content": user_prompt}
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self.tokenizer([inputs], return_tensors="pt").to(self.model.device)
        generated_ids = self.model.generate(**model_inputs, max_new_tokens=max_tokens)
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
        content = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        return content
    
    def _postprocess_verdict(self, verdict: str):
        verdict = verdict.strip().lower()
        
        for output in self.output_mapping.keys():
            if output in verdict:
                verdict = output
                break
        if verdict not in self.output_mapping.keys():
            verdict = "n/a"
        return self.output_mapping[verdict]
                
    def verfiy_facts(self, facts):
        res = []
        
        for fact in facts:
            prompt = self.verify_user_prompt_template.format(fact=fact)
            verdict = self._generate(user_prompt=prompt, max_tokens=4)
            res.append(self.postprocess_verdict(verdict))
        return res