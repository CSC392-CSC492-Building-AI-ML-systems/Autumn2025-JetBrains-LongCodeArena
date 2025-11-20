import os

import numpy as np
from openai import OpenAI
import tiktoken
import torch
from together import Together

OPENAI_SYSTEM_PROMPT = 'You are a code quality assesing engine.'

class OptionsScoringModel:
    def __init__(self,
                 model_name: str) -> None:
        self.model_name = model_name
        self.is_openai = model_name.startswith('gpt')
        
        if self.is_openai:
            token = os.getenv('OPENAI_API_KEY')
            if token is None:
                raise RuntimeError('The env variable OPENAI_API_KEY must be set!')
            self.model = OpenAI(api_key=token)
            self.tokenizer = tiktoken.encoding_for_model(model_name)
        else:
            token = os.getenv('TOGETHER_API_KEY')
            if token is None:
                raise RuntimeError('The env variable TOGETHER_API_KEY must be set!')
            self.model = Together(api_key=token)
            self.tokenizer = tiktoken.get_encoding("cl100k_base")


    def score_options(self, query: str, options: list[str]) -> torch.Tensor:
        if self.is_openai:
            return self._score_options_openai(query, options)
        else:
            return self._score_options_together(query, options)
        

    def _score_options_openai(self, query: str, options: list[str]) -> torch.Tensor:
        logit_bias = dict()
        for opt in options:
            tok_ids = self.tokenizer.encode(opt)
            assert len(tok_ids) == 1, 'Only single token options are supported'
            logit_bias[tok_ids[0]] = 100
        
        completion = self.model.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": OPENAI_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            max_tokens=1,
            temperature=0.3,
            n=1,
            logprobs=True,
            top_logprobs=20,
            logit_bias=logit_bias
        )

        logprobs = np.full(2, np.nan)
        choice = completion.choices[0]
        opt_to_idx = {t: n for n, t in enumerate(options)}
        min_lp = 0
        
        for logprob_item in choice.logprobs.content[0].top_logprobs:
            tok = logprob_item.token
            lp = logprob_item.logprob
            min_lp = min(min_lp, lp)
            if tok in opt_to_idx:
                logprobs[opt_to_idx[tok]] = lp
        
        logprobs[np.isnan(logprobs)] = min_lp - 2.3
        assert not np.isnan(logprobs).any()
        return torch.from_numpy(logprobs)

    def _score_options_together(self, query: str, options: list[str]) -> torch.Tensor:
        logit_bias = dict()
        for opt in options:
            tok_ids = self.tokenizer.encode(opt)
            assert len(tok_ids) == 1, 'Only single token options are supported'
            logit_bias[str(tok_ids[0])] = 100
        
        completion = self.model.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": OPENAI_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            max_tokens=1,
            temperature=0.3,
            n=1,
            logprobs=1,
            logit_bias=logit_bias
        )

        logprobs = np.full(2, np.nan)
        choice = completion.choices[0]
        opt_to_idx = {t: n for n, t in enumerate(options)}
        min_lp = 0
        
        if choice.logprobs and choice.logprobs.top_logprobs:
            for token_dict in choice.logprobs.top_logprobs:
                for tok, lp in token_dict.items():
                    min_lp = min(min_lp, lp)
                    if tok in opt_to_idx:
                        logprobs[opt_to_idx[tok]] = lp
        
        logprobs[np.isnan(logprobs)] = min_lp - 2.3
        assert not np.isnan(logprobs).any()
        return torch.from_numpy(logprobs)
