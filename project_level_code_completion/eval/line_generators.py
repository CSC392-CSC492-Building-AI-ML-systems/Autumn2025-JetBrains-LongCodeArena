import logging
import os
import random
from dataclasses import dataclass
from typing import Dict

import jsonlines
import numpy as np
import torch
from evaluate import load
from thefuzz import fuzz
from tqdm.auto import tqdm
from transformers import AutoTokenizer, StoppingCriteria, StoppingCriteriaList

from data_classes.datapoint_base import DatapointBase
from data_classes.datapoint_commit_dataset import DatapointCommitDataset
from model_hub.model_inference import get_input_data, get_model


@dataclass
class GeneratorConfig:
    input_data_path: str
    seq_max_len: int
    context_max: int
    model: str
    device: str
    best_perplexity: float
    tokenizer_path: str
    composer: str
    seed: int
    results_path: str


logging.basicConfig(level=logging.ERROR)


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


@dataclass
class GenerationResults:
    prediction: list[str]
    gt: list[str]

    def append_result(self, prediction, gt):
        self.prediction.append(prediction)
        self.gt.append(gt)

# ===================================================
# IBM GRANITE IMPROVEMENTS
# ------------------------------------------------------------
# These helpers replace HF exact_match metric with a safer,
# dependency-free fallback version.
#
def _compute_em_local(preds: list[str], refs: list[str]):
    assert len(preds) == len(refs)
    n = len(preds)
    if n == 0:
        return {"exact_match": 0.0}
    hits = sum(1 for p, r in zip(preds, refs) if p.strip() == r.strip())
    return {"exact_match": hits / n}

def get_em_metric():
    try:
        metric = load("evaluate-metric/exact_match")
        return lambda preds, refs: metric.compute(predictions=preds, references=refs)
    except Exception:
        pass
    try:
        metric = load("exact_match")
        return lambda preds, refs: metric.compute(predictions=preds, references=refs)
    except Exception:
        pass
    return _compute_em_local
# ============================================================


class LineGeneratorBase:
    def __init__(self, model, device, max_seq_len, results_path):
        self.model = model
        self.device = device
        self.max_seq_len = max_seq_len
        self.results_path = results_path
        self.generation_results: Dict[str, GenerationResults] = dict()

    def choose_lines(self, datapoint) -> list[int]:
        raise NotImplementedError

    @staticmethod
    def _get_context(datapoint: DatapointBase, line_num: int) -> (str, str):
        """Method returns context and a line to predict"""
        context = "\n".join([datapoint.context] + [datapoint.get_prefix(line_num)])
        gt_line = datapoint.get_line(line_num)
        return context, gt_line

    @staticmethod
    def _get_zero_context(datapoint, line_num) -> (str, str):
        """Method returns context and a line to predict"""
        context = datapoint.get_prefix(line_num)
        gt_line = datapoint.get_line(line_num)
        return context, gt_line


    def generate_line(self, datapoint):
        raise NotImplementedError

    def calculate_exact_match(self):
        raise NotImplementedError

    def _load_tokenizer(self):
        raise NotImplementedError

    def tokenize(self, text):
        raise NotImplementedError

    def decode(self, generated_token_ids):
        raise NotImplementedError

    def _get_generation_config(self):
        raise NotImplementedError

    # @staticmethod
    # def _get_completion_lines(datapoint):
    #     return datapoint['completion'].split("\n")

    def aggregate_metric(self, metric_result):
        agg_result = 0.
        agg_len = 0
        metric_name = None
        for sc_name, sc_score in metric_result.items():
            agg_result += list(sc_score.values())[0] * len(self.generation_results[sc_name].gt)
            agg_len += len(self.generation_results[sc_name].gt)
            metric_name = list(sc_score.keys())[0]
        if len(metric_result) > 0:
            return {metric_name: agg_result / agg_len}

    def save_results(self, results):
        with jsonlines.open(self.results_path, 'a') as writer:
            writer.write(results)


class SpecificLineGenerator(LineGeneratorBase):
    @staticmethod
    def load_lines(datapoint: DatapointBase) -> dict[str, list[int]]:
        return datapoint.completion_lines

    @staticmethod
    def sample_noninformative(non_informative_lines: list[int], sample_size: int = 6, seed: int = 42):
        local_random = random.Random(seed)
        local_sample_size = min(len(non_informative_lines), sample_size)
        return local_random.sample(non_informative_lines, local_sample_size)


class LineGeneratorHF(SpecificLineGenerator):
    def __init__(self, model, device, max_seq_len, results_path, tokenizer_path):
        super().__init__(model, device, max_seq_len, results_path)
        self.tokenizer_path = tokenizer_path
        self._tokenizer: AutoTokenizer
        self._load_tokenizer()

    @torch.inference_mode()
    def generate_line(self, datapoint: DatapointBase, use_zero_context: bool = False) -> dict[str, int]:
        dict_of_lines = self.load_lines(datapoint)
        gen_config = self._get_generation_config()
        for sc_name, list_of_lines in dict_of_lines.items():
            # print('='*25, sc_name, '='*25)
            self.generation_results[sc_name] = GenerationResults(list(), list())
            for line_num in list_of_lines:
                context, gt_line = self._get_context(datapoint, line_num)
                if use_zero_context:
                    context, gt_line = self._get_zero_context(datapoint, line_num)
                # When the context is too long, we want to crop the beginning for more efficient tokenization
                if len(context) > self.max_seq_len * 6:
                    context = context[-self.max_seq_len * 6:]
                input_ids = self.tokenize(context)[..., -self.max_seq_len:]
                if input_ids.size(-1) < 1:
                    new_size = torch.Size(list(input_ids.size())[:-1] + [1])
                    fill_id = getattr(self._tokenizer, 'bos_token_id', None)
                    if fill_id is None:
                        fill_id = getattr(self._tokenizer, 'eos_token_id', None)
                    if fill_id is None:
                        fill_id = getattr(self._tokenizer, 'pad_token_id', None)
                    if fill_id is None:
                        fill_id = 0
                    input_ids = torch.full(new_size, fill_id, dtype=torch.long)
                input_ids = input_ids.to(self.device)

                # IBM improvement: provide attention_mask to avoid warnings
                # attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=input_ids.device)
                # out = self.model.generate(input_ids, attention_mask=attention_mask, **gen_config)

                out = self.model.generate(input_ids, **gen_config)
                out = out[..., input_ids.size(-1):]
                prediction = self.decode(out)
                prediction = prediction.strip("\n")
                prediction_line = prediction.split("\n")[0]
                self.save_results({'original_prediction': prediction, 'prediction_line': prediction_line, 'ground_truth': gt_line, 'line_class': sc_name, 'zero_context': use_zero_context})
                self.generation_results[sc_name].append_result(prediction=prediction_line, gt=gt_line)

        # datapoint.pop('completion_lines', None)
        return {k: len(v) for k, v in dict_of_lines.items()}

    def _load_tokenizer(self):
        self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)

    def tokenize(self, text):
        return self._tokenizer(text, return_tensors='pt', padding=False)['input_ids']

    def _get_generation_config(self):
        class StopOnNewLine(StoppingCriteria):
            def __init__(self, tokenizer):
                self.stop_ids = set()
                for k, tok_id in tokenizer.vocab.items():
                    s = tokenizer.convert_tokens_to_string([k])
                    if '\n' in s:
                        self.stop_ids.add(tok_id)
                self._num_generated_tokens = 0

            def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
                assert input_ids.shape[0] == 1  # only batch_size 1 is supported
                if self._num_generated_tokens < 5:
                    self._num_generated_tokens += 1
                    return False
                elif input_ids[0, -1].item() in self.stop_ids:
                    self._num_generated_tokens = 0
                    return True
                else:
                    self._num_generated_tokens += 1
                    return False

        stopping_criteria = StoppingCriteriaList([StopOnNewLine(self._tokenizer)])
        # newline_token_id = self._tokenizer.encode('\n', add_special_tokens=False)[0]
        return {
            'max_new_tokens': 100,
            'do_sample': False,
            'stopping_criteria': stopping_criteria,
            'eos_token_id': self._tokenizer.eos_token_id,
            'pad_token_id': self._tokenizer.eos_token_id,
        }

    def decode(self, generated_token_ids):
        return self._tokenizer.batch_decode(generated_token_ids, skip_special_tokens=True)[0]

    def calculate_exact_match(self):
        # IBM alternative — uncomment to switch:
        # results = {}
        # for sc_name, gen_res in self.generation_results.items():
        #     if len(gen_res.gt) > 0:
        #         preds = [p.strip() for p in gen_res.prediction]
        #         refs = [r.strip() for r in gen_res.gt]
        #         results[sc_name] = self._compute_em(preds, refs)
        # return results
        exact_match = load("evaluate-metric/exact_match")
        results = dict()
        for sc_name, gen_res in self.generation_results.items():
            if len(gen_res.gt) > 0:
                results[sc_name] = exact_match.compute(
                    references=[item.strip() for item in gen_res.gt],
                    predictions=[item.strip() for item in gen_res.prediction],
                )
        return results

    def calculate_edit_similarity(self):
        similarity = 0.
        count = 0
        result = dict()
        for sc_name, gen_res in self.generation_results.items():
            for pred, gt in zip(gen_res.prediction, gen_res.gt):
                similarity += fuzz.ratio(pred, gt)
                count += 1
            if count > 0:
                result[sc_name] = {'edit_similarity': similarity / count}
        return result



class LineGeneratorOpenAI(SpecificLineGenerator):
    """OpenAI-backed line generator that mirrors HF generator behavior but uses chat completions.

    Notes:
    - Expects DatapointCommitDataset instances with text fields `context` and `completion` present.
    - Does not use tokenization or logits; crops long string context by characters.
    - Saves per-sample predictions to results_path as jsonlines, same keys as HF generator.
    """
    def __init__(self, model_name: str, results_path: str, max_context_chars: int = 16000, language: str = "python", max_repo_ctx_tokens: int = 1024):
        super().__init__(model=None, device="cpu", max_seq_len=0, results_path=results_path)
        from openai import OpenAI
        import os

        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model_name = model_name
        self.max_context_chars = max_context_chars
        self.language = language
        self.max_repo_ctx_tokens = max_repo_ctx_tokens
        # Setup token encoding for safe truncation
        try:
            import tiktoken
            try:
                self._enc = tiktoken.encoding_for_model(model_name)
            except Exception:
                self._enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self._enc = None  # fallback: will truncate by characters only
        # Model token limits (approx). gpt-3.5-turbo supports 16385 tokens.
        self._model_max_tokens = 16385
        self._gen_max_tokens = 32
        self._prompt_overhead = 600  # instruction, chat formatting, safety margin

    def _system_message(self) -> str:
        return (
            "You are a precise code completion assistant.\n"
            "Respond with EXACTLY one physical line of source code and nothing else.\n"
            "Do NOT output markdown, triple backticks, XML/HTML tags, quotes, labels, or explanations.\n"
            "Do NOT echo the words 'Next line', 'Answer', or any similar label.\n"
            "Do NOT output '```python' or any newline '\n' characters. Only predict a SINGLE line."
            "Preserve indentation exactly as required by the file.\n"
            "Return only the characters of that line, with no leading or trailing blank lines.\n"
            "If the next line is empty, return an empty string (no spaces, no text)."
        )

    def _build_prompt(self, repo_context: str, file_prefix: str, file_path: str) -> str:
        header = (
            f"Language: {self.language}\n"
            f"File: {file_path}\n"
            "Task: Predict the next single physical line that follows the prefix below.\n"
            "Return ONLY the next line exactly as it should appear. Do not include any extra text or formatting.\n"
        )
        return (
            f"{header}\n"
            f"<REPO_CONTEXT>\n{repo_context}\n</REPO_CONTEXT>\n\n"
            f"<FILE_PREFIX>\n{file_prefix}\n</FILE_PREFIX>\n\n"
            f"Provide only the next line."
        )

    def _count_tokens(self, text: str) -> int:
        if self._enc is None:
            return len(text) // 4  # very rough fallback
        return len(self._enc.encode(text))

    def _trim_to_fit(self, repo_ctx: str, prefix: str) -> tuple[str, str, dict]:
        """Trim repo_ctx first (from the head) to fit model token window, preserving prefix as much as possible.
        If prefix alone is too long, trim its head as well (keep the tail).
        Returns (repo_ctx_trimmed, prefix_trimmed, stats).
        """
        # initial char-based cap on repo_ctx
        if len(repo_ctx) > self.max_context_chars:
            repo_ctx = repo_ctx[-self.max_context_chars:]

        # Compute allowed tokens for content
        allowed = self._model_max_tokens - self._gen_max_tokens - self._prompt_overhead
        # Estimate tokens for static parts of the prompt (tags + instruction). Roughly accounted in overhead.

        if self._enc is None:
            # If no tokenizer, apply coarse character trimming once more
            combined = repo_ctx + "\n" + prefix
            if len(combined) > allowed * 3:  # coarse 3 chars/token heuristic
                keep_chars = allowed * 3
                combined = combined[-keep_chars:]
                # Naively split back: bias to keep prefix whole when possible
                # Keep full prefix up to its length; the rest goes to repo ctx
                pf_part = min(len(prefix), len(combined))
                prefix_trimmed = combined[-pf_part:]
                repo_ctx_trimmed = combined[: len(combined) - pf_part]
                return repo_ctx_trimmed, prefix_trimmed, {
                    'allowed_tokens': allowed,
                    'tokenizer': 'approx_chars',
                }
            return repo_ctx, prefix, {'allowed_tokens': allowed, 'tokenizer': 'approx_chars'}

        # Token-aware path
        rc_ids = self._enc.encode(repo_ctx)
        pf_ids = self._enc.encode(prefix)
        rc_len = len(rc_ids)
        pf_len = len(pf_ids)
        # If prefix alone exceeds allowed, keep tail of prefix
        if pf_len > allowed:
            pf_ids = pf_ids[-allowed:]
            return "", self._enc.decode(pf_ids), {
                'allowed_tokens': allowed,
                'repo_ctx_tokens': 0,
                'prefix_tokens': len(pf_ids)
            }
        # Reserve space for full prefix, trim repo_ctx if needed
        rc_allowed = max(0, allowed - pf_len)
        # Cap repo context to avoid overwhelming prefix
        rc_allowed = min(rc_allowed, self.max_repo_ctx_tokens)
        if rc_len > rc_allowed:
            rc_ids = rc_ids[-rc_allowed:]
        return self._enc.decode(rc_ids), self._enc.decode(pf_ids), {
            'allowed_tokens': allowed,
            'repo_ctx_tokens': len(rc_ids),
            'prefix_tokens': len(pf_ids)
        }

    def _chat_complete(self, system_text: str, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_text},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=self._gen_max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=["\n"],
        )
        text = response.choices[0].message.content or ""
        # Normalize newlines and strip
        text = text.replace("\r\n", "\n").replace("\r", "\n").strip("\n")
        return text

    def generate_line(self, datapoint: DatapointBase, use_zero_context: bool = False) -> dict[str, int]:
        dict_of_lines = self.load_lines(datapoint)
        for sc_name, list_of_lines in dict_of_lines.items():
            self.generation_results[sc_name] = GenerationResults(list(), list())
            for line_num in list_of_lines:
                if use_zero_context:
                    prompt_prefix, gt_line = self._get_zero_context(datapoint, line_num)
                    repo_ctx = ""
                else:
                    full_context, gt_line = self._get_context(datapoint, line_num)
                    # full_context = context + "\n" + prefix; split to avoid duplication
                    # We reconstruct repo context and file prefix explicitly
                    repo_ctx = datapoint.context or ""
                    prompt_prefix = datapoint.get_prefix(line_num)

                # Token-aware trimming to fit within the model window
                repo_ctx, prompt_prefix, fit_stats = self._trim_to_fit(repo_ctx, prompt_prefix)

                # Determine file path for better grounding
                if getattr(datapoint, "completion_dict", None):
                    try:
                        file_path = list(datapoint.completion_dict.keys())[0]
                    except Exception:
                        file_path = "<unknown>"
                else:
                    file_path = "<unknown>"

                prompt = self._build_prompt(repo_ctx, prompt_prefix, file_path)
                prediction = self._chat_complete(self._system_message(), prompt)
                # Prefer content inside <line>...</line>
                norm = prediction.replace("\r\n", "\n").replace("\r", "\n").strip("\n")
                prediction_line = ""
                if "<line>" in norm and "</line>" in norm:
                    try:
                        start = norm.index("<line>") + len("<line>")
                        end = norm.index("</line>", start)
                        prediction_line = norm[start:end]
                    except ValueError:
                        prediction_line = ""
                if prediction_line == "":
                    # Fallbacks: drop fences and pick first non-empty line
                    lines = norm.split("\n") if norm else []
                    non_fence_lines = [ln for ln in lines if not ln.strip().startswith("```")]
                    for ln in non_fence_lines:
                        clean_ln = ln
                        if clean_ln.lstrip().lower().startswith("next line:"):
                            parts = clean_ln.split(":", 1)
                            clean_ln = parts[1] if len(parts) > 1 else ""
                        if clean_ln.strip() != "":
                            prediction_line = clean_ln
                            break
                    # Allow truly blank line as valid
                    if prediction_line == "" and non_fence_lines and all(ln.strip() == "" for ln in non_fence_lines):
                        prediction_line = ""
                    # Final fallback
                    if prediction_line == "" and lines:
                        prediction_line = lines[0]

                self.save_results({
                    'original_prediction': prediction,
                    'prediction_line': prediction_line,
                    'ground_truth': gt_line,
                    'line_class': sc_name,
                    'zero_context': use_zero_context,
                    'file_path': file_path,
                    'repo_context_chars': len(repo_ctx),
                    'prefix_chars': len(prompt_prefix),
                    'allowed_tokens': fit_stats.get('allowed_tokens', None),
                    'repo_ctx_tokens': fit_stats.get('repo_ctx_tokens', None),
                    'prefix_tokens': fit_stats.get('prefix_tokens', None)
                })
                self.generation_results[sc_name].append_result(prediction=prediction_line, gt=gt_line)

        return {k: len(v) for k, v in dict_of_lines.items()}

    def calculate_exact_match(self):
        exact_match = load("evaluate-metric/exact_match")
        results = dict()
        for sc_name, gen_res in self.generation_results.items():
            if len(gen_res.gt) > 0:
                results[sc_name] = exact_match.compute(
                    references=[item.strip() for item in gen_res.gt],
                    predictions=[item.strip() for item in gen_res.prediction],
                )
        return results

    def calculate_edit_similarity(self):
        similarity = 0.
        count = 0
        result = dict()
        for sc_name, gen_res in self.generation_results.items():
            similarity = 0.
            count = 0
            for pred, gt in zip(gen_res.prediction, gen_res.gt):
                similarity += fuzz.ratio(pred, gt)
                count += 1
            if count > 0:
                result[sc_name] = {'edit_similarity': similarity / count}
        return result


@torch.inference_mode()
def evaluate_generation(args: GeneratorConfig):
    set_seed(args.seed)
    loaded_data = get_input_data(args)
    if isinstance(loaded_data[0], dict):
        input_data = [DatapointCommitDataset(**input_dict) for input_dict in loaded_data]
    elif isinstance(loaded_data[0], DatapointCommitDataset):
        input_data = loaded_data.copy()
    else:
        raise NotImplementedError

    model, device = get_model(args)

    def calculate_metrics(model=model, device=device, use_zero_context=False, args=args, input_data=input_data):
        em_dict = dict()
        es_dict = dict()
        em_dict['all'] = list()
        es_dict['all'] = list()
        sc_counts = None
        for datapoint in tqdm(input_data):
            generator = LineGeneratorHF(model, device, max_seq_len=args.seq_max_len, tokenizer_path=args.tokenizer_path, results_path=args.results_path)
            el_counts = generator.generate_line(datapoint, use_zero_context=use_zero_context)
            if sc_counts is None:
                sc_counts = el_counts
            else:
                for k in el_counts.keys():
                    sc_counts[k] += el_counts[k]
            em = generator.calculate_exact_match()
            es = generator.calculate_edit_similarity()
            em_dict['all'].append(generator.aggregate_metric(em)['exact_match'])
            es_dict['all'].append(generator.aggregate_metric(es)['edit_similarity'])
            for sc_name in em.keys():
                if sc_name not in em_dict:
                    em_dict[sc_name] = list()
                if sc_name not in es_dict:
                    es_dict[sc_name] = list()

                try:
                    em_dict[sc_name].append(em[sc_name]['exact_match'])
                except KeyError:
                    pass
                try:
                    es_dict[sc_name].append(es[sc_name]['edit_similarity'])
                except KeyError:
                    pass
        return em_dict, es_dict, sc_counts

    def process_results(use_zero_context):
        em_dict, es_dict, sc_counts = calculate_metrics(use_zero_context=use_zero_context)
        if use_zero_context:
            print(f'Final results for zero context:')
        else:
            print(f'Final results for full context:')
        for sc_name in em_dict.keys():
            em_list = em_dict[sc_name]
            es_list = es_dict[sc_name]
            print(f'Metrics for {sc_name} lines: EM {sum(em_list) / len(em_list):.2f}, ES {sum(es_list) / len(es_list):.2f}')

        return em_dict, es_dict, sc_counts

    set_seed(args.seed)
    em_dict_0, es_dict_0, line_counts_0 = process_results(use_zero_context=True)
    set_seed(args.seed)
    em_dict, es_dict, line_counts = process_results(use_zero_context=False)
    assert line_counts_0 == line_counts, "you have different line counts"
    em_diff_dict = dict()
    for sc_name in em_dict.keys():
        em_list = em_dict[sc_name]
        em_list_0 = em_dict_0[sc_name]
        assert len(em_list) == len(em_list_0), 'your score has different lengths'
        em_diff_dict[sc_name] = {
            'positive': sum([(sc - sc_0) > 0 for sc, sc_0 in zip(em_list, em_list_0)]) / len(em_list),
            'negative': sum([(sc - sc_0) < 0 for sc, sc_0 in zip(em_list, em_list_0)]) / len(em_list),
            'zero': sum([(sc - sc_0) == 0 for sc, sc_0 in zip(em_list, em_list_0)]) / len(em_list),
        }

    return [
        {
            'em_zero': {sc_name: sum(m_list) / len(m_list) for sc_name, m_list in em_dict_0.items()},
            'es_zero': {sc_name: sum(m_list) / len(m_list) for sc_name, m_list in es_dict_0.items()},
            'em': {sc_name: sum(m_list) / len(m_list) for sc_name, m_list in em_dict.items()},
            'es': {sc_name: sum(m_list) / len(m_list) for sc_name, m_list in es_dict.items()},
        },
        {
            'em_zero_list': em_dict_0,
            'es_zero_list': es_dict_0,
            'em_list': em_dict,
            'es_list': es_dict,
        },
        em_diff_dict,
        line_counts
    ]


    # print(f'Final results for zero context: '
    #       f'EM {sum(em_list) / len(em_list):.2f}, ES {sum(es_list) / len(es_list):.2f}')


if __name__ == '__main__':
    args = GeneratorConfig(
        input_data_path="/home/glukhov/long_code_arena/lca/data/python/smol/model_inputs_composer_path_distance.json",
        seq_max_len=3500 - 30,
        context_max=3500,
        model="starcoder1b",
        device="cuda",
        best_perplexity=0.,
        tokenizer_path="bigcode/starcoderbase-1b",
        composer="path_distance",
        seed=42
    )
    print(args.input_data_path)
    out = evaluate_generation(args)
    for out_ in out:
        print(out_)
        print()
