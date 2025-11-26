import torch
from openai import OpenAI


class OptionsScoringModel:
    """
    Scores a small set of string options (like ['A', 'B']) by asking an
    OpenAI chat model to continue the prompt and using token logprobs
    for the first output token.
    """
    def __init__(self, api_key: str, model_name: str):
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name

    def score_options(self, prompt, options):
        # We expect something like options = ["A", "B"]
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1,
            temperature=0,
            logprobs=True,          # ask for token logprobs :contentReference[oaicite:0]{index=0}
            top_logprobs=len(options),
        )

        # First generated token
        token_info = resp.choices[0].logprobs.content[0]
        top = {t.token: t.logprob for t in token_info.top_logprobs}

        scores = []
        for opt in options:
            # Try exact token and " opt" (with leading space)
            logprob = top.get(opt)
            if logprob is None:
                logprob = top.get(" " + opt, None)
            if logprob is None:
                # If the option isn't in top_logprobs, give a very low score
                logprob = -1e9
            scores.append(logprob)

        return torch.tensor(scores)
