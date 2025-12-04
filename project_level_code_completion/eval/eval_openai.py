import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf
from tqdm.auto import tqdm

from data_classes.datapoint_commit_dataset import DatapointCommitDataset
from eval.preprocess import get_composers
from eval.line_generators import LineGeneratorOpenAI, GenerationResults
from eval.preprocessors import TextOnlyPreprocessor


DATASET_NAME_MAP = {
    "small": "small_context",
    "medium": "medium_context",
    "large": "large_context",
    "huge": "huge_context",
}


@dataclass
class OpenAIEvalConfig:
    model: str
    dataset: str  # accepts "small"|"medium"|... or full name like "small_context"
    composers: str
    tokenizer: str | None
    out_root: str
    language: str
    limit: int
    max_context_chars: int
    skip_zero_context: bool = True
    repo_name: str | None = None
    repo_id: int | None = None


def _resolve_dataset_name(name: str) -> str:
    if name in DATASET_NAME_MAP:
        return DATASET_NAME_MAP[name]
    # allow passing already-resolved names like "small_context"
    return name


def _friendly_model_dir(model: str) -> str:
    if "gpt-3.5" in model:
        return "gpt-3.5"
    if "gpt-4o" in model:
        return "gpt-4o"
    return model.replace(" ", "-")


def _default_output_dir(cfg: OpenAIEvalConfig) -> str:
    dataset_name = _resolve_dataset_name(cfg.dataset)
    model_dir = _friendly_model_dir(cfg.model)
    # Optional repo-specific suffix
    repo_suffix = None
    if cfg.repo_id is not None:
        repo_suffix = f"repo-{cfg.repo_id}"
    elif cfg.repo_name:
        rn = cfg.repo_name.lower().replace("/", "-").replace("\\", "-").replace(" ", "-")
        repo_suffix = f"repo-{rn}"
    parts = [cfg.out_root, cfg.language, model_dir, dataset_name]
    if repo_suffix:
        parts.append(repo_suffix)
    parts.append("results")
    return os.path.join(*parts)


def _prepare_dataset(cfg: OpenAIEvalConfig) -> str:
    dataset_conf = OmegaConf.create({
        "path": "JetBrains-Research/lca-project-level-code-completion",
        "name": _resolve_dataset_name(cfg.dataset),
    })
    class _ComposerArgs:
        def __init__(self):
            self.composers = cfg.composers

    composer_args = {"lang_sep_symbol": "", "meta_info_sep_symbol": "METASEP\n", "extension": ""}
    composers = get_composers(_ComposerArgs(), composer_args)

    model_dir = _friendly_model_dir(cfg.model)
    out_dir = os.path.join(cfg.out_root, cfg.language, model_dir, _resolve_dataset_name(cfg.dataset), "in")
    os.makedirs(out_dir, exist_ok=True)
    prepared_dataset_path = os.path.join(out_dir, f"model_inputs_composer_{cfg.composers}.json")

    preprocessor = TextOnlyPreprocessor(
        dataset_params=dataset_conf,
        tokenizer_path=None,
        context_len_char=cfg.max_context_chars,
        **composers,
    )
    preprocessor.prepare_model_input_parallel(num_workers=1, dataset_path=prepared_dataset_path)
    return prepared_dataset_path


def _load_datapoints(prepared_path: str) -> list[DatapointCommitDataset]:
    with open(prepared_path, "r") as f:
        data = json.load(f)
    return [DatapointCommitDataset(**dp) if isinstance(dp, dict) else dp for dp in data]


def evaluate_openai(cfg: OpenAIEvalConfig) -> None:
    dataset_resolved = _resolve_dataset_name(cfg.dataset)
    print("=" * 70)
    print(f"OpenAI Evaluation: {cfg.model} on {dataset_resolved}")
    print("=" * 70)

    out_dir = _default_output_dir(cfg)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    print("\n[STEP 1/2] Preprocessing dataset (text-only, no tokenizer)...")
    prepared_path = _prepare_dataset(cfg)
    datapoints = _load_datapoints(prepared_path)

    # Optional filtering by repository
    if cfg.repo_id is not None:
        dp_before = len(datapoints)
        datapoints = [dp for dp in datapoints if getattr(dp, "repo_id", None) == cfg.repo_id]
        print(f"Applied repo_id filter ({cfg.repo_id}): {dp_before} -> {len(datapoints)} datapoints")
    if cfg.repo_name:
        rn_l = cfg.repo_name.lower()
        dp_before = len(datapoints)
        datapoints = [dp for dp in datapoints if rn_l in getattr(dp, "repo_name", "").lower()]
        print(f"Applied repo_name filter ('{cfg.repo_name}'): {dp_before} -> {len(datapoints)} datapoints")

    if cfg.limit and cfg.limit > 0:
        datapoints = datapoints[: cfg.limit]

    limit_str = str(cfg.limit) if (cfg.limit and cfg.limit > 0) else "all"
    print(f"Prepared {len(datapoints)} datapoints (limit={limit_str}).")

    results_path = os.path.join(out_dir, "generation_results.jsonl")
    print("\n[STEP 2/2] Generating with OpenAI Chat Completions...")
    generator = LineGeneratorOpenAI(
        model_name=cfg.model,
        results_path=results_path,
        max_context_chars=cfg.max_context_chars,
        language=cfg.language,
    )

    def _compute(use_zero_context: bool):
        em_dict: dict[str, list[float]] = {"all": []}
        es_dict: dict[str, list[float]] = {"all": []}
        sc_counts: dict[str, int] | None = None

        for dp in tqdm(datapoints):
            generator.generation_results = {}
            el_counts = generator.generate_line(dp, use_zero_context=use_zero_context)
            if sc_counts is None:
                sc_counts = el_counts
            else:
                for k in el_counts.keys():
                    sc_counts[k] += el_counts[k]

            em = generator.calculate_exact_match()
            es = generator.calculate_edit_similarity()

            # Aggregate across all categories
            agg_em = generator.aggregate_metric(em)
            agg_es = generator.aggregate_metric(es)
            if agg_em:
                em_dict["all"].append(agg_em["exact_match"])  # type: ignore[index]
            if agg_es:
                es_dict["all"].append(agg_es["edit_similarity"])  # type: ignore[index]

            # Per-category lists
            for sc_name in em.keys():
                em_dict.setdefault(sc_name, [])
                try:
                    em_dict[sc_name].append(em[sc_name]["exact_match"])  # type: ignore[index]
                except KeyError:
                    pass
            for sc_name in es.keys():
                es_dict.setdefault(sc_name, [])
                try:
                    es_dict[sc_name].append(es[sc_name]["edit_similarity"])  # type: ignore[index]
                except KeyError:
                    pass

        return em_dict, es_dict, sc_counts or {}

    # Compute metrics
    em_zero = es_zero = counts_zero = None
    if not cfg.skip_zero_context:
        em_zero, es_zero, counts_zero = _compute(use_zero_context=True)
    em_full, es_full, counts_full = _compute(use_zero_context=False)
    if counts_zero is not None:
        assert counts_zero == counts_full, "Line counts mismatch between zero and full context runs"

    def _avg(d: dict[str, list[float]]) -> dict[str, float]:
        out: dict[str, float] = {}
        for k, v in d.items():
            if len(v) > 0:
                out[k] = sum(v) / len(v)
        return out

    scores = {
        "em": _avg(em_full),
        "es": _avg(es_full),
        "dataset": _resolve_dataset_name(cfg.dataset),
        "composer": cfg.composers,
        "model": cfg.model,
        "limit": cfg.limit,
        "max_context_chars": cfg.max_context_chars,
        "skip_zero_context": cfg.skip_zero_context,
    }
    if cfg.repo_id is not None:
        scores["repo_id"] = cfg.repo_id
    if cfg.repo_name:
        scores["repo_name"] = cfg.repo_name
    if em_zero is not None and es_zero is not None:
        scores["em_zero"] = _avg(em_zero)
        scores["es_zero"] = _avg(es_zero)

    with open(os.path.join(out_dir, "generation_scores.json"), "w") as f:
        json.dump(scores, f, indent=2)

    print("\nSaved:", os.path.join(out_dir, "generation_scores.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--dataset", type=str, default="small")
    parser.add_argument("--composers", type=str, default="path_distance")
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--out_root", type=str, default=os.path.join(os.getcwd(), "data", "code_generation", "artifacts"))
    parser.add_argument("--language", type=str, default="python")
    parser.add_argument("--limit", type=int, default=0, help="0 or negative means use the full dataset; >0 samples N")
    parser.add_argument("--max_context_chars", type=int, default=16000)
    parser.add_argument("--skip_zero_context", action="store_true", default=True)
    parser.add_argument("--repo_name", type=str, default=None, help="Filter datapoints to repos whose name contains this substring")
    parser.add_argument("--repo_id", type=int, default=None, help="Filter datapoints to repos with this exact id")

    args = parser.parse_args()

    cfg = OpenAIEvalConfig(
        model=args.model,
        dataset=args.dataset,
        composers=args.composers,
        tokenizer=args.tokenizer,
        out_root=args.out_root,
        language=args.language,
        limit=args.limit,
        max_context_chars=args.max_context_chars,
        skip_zero_context=args.skip_zero_context,
        repo_name=args.repo_name,
        repo_id=args.repo_id,
    )
    evaluate_openai(cfg)
