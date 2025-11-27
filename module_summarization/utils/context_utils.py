import ast
import os
import re
from rank_bm25 import BM25Okapi

def collect_good_context_pathes(row):
    context = ""
    row['good_code_files'] = ast.literal_eval(str(row['good_code_files']).replace('repos', '../data/repos_clean'))
    for con_path in row['good_code_files']:
        with open(con_path, 'r') as f:
            con = f.read()
            context += '\n\n' + con
    context = context.lstrip()
    return context

def collect_good_context(row):
    return row['relevant_code_context']


def trim_context(context, tokenizer, max_len):
    tokenized_context = tokenizer.encode(context, max_length=512_000, truncation=True)
    tokenized_context = tokenized_context[:max_len]
    detokenized_context = tokenizer.decode(tokenized_context)
    return detokenized_context

def collect_bm25_files_context(row, base_repo_dir: str = "../data/repos_clean", k: int = 5, max_chars_per_file: int | None = None,):
    files_field = row["relevant_code_files"]

    if isinstance(files_field, str):
        candidate_paths = ast.literal_eval(files_field)
    else:
        candidate_paths = list(files_field)

    docs_tokens = []
    file_texts = []
    actual_paths = []

    for p in candidate_paths:
        path = str(p).replace("repos", base_repo_dir)

        if not os.path.exists(path):
            continue

        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except OSError:
            continue

        if max_chars_per_file is not None:
            text = text[:max_chars_per_file]

        tokens = re.findall(r"\w+", text.lower())
        if not tokens:
            continue

        file_texts.append(text)
        docs_tokens.append(tokens)
        actual_paths.append(path)

    if not docs_tokens:
        return collect_good_context(row)

    bm25 = BM25Okapi(docs_tokens)

    intent = row.get("intent", "")
    docname = row.get("docfile_name", "")
    query = f"{intent} {docname}"
    query_tokens = re.findall(r"\w+", text.lower())

    scores = bm25.get_scores(query_tokens)

    ranked = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True,
    )
    top_indices = ranked[:k]

    context_parts = []
    for i in top_indices:
        context_parts.append(file_texts[i])

    context = "\n\n".join(context_parts).lstrip()
    return context

def build_context(row, strategy: str = "precomputed", **kwargs):
    if strategy == "precomputed":
        return collect_good_context(row)
    elif strategy == "bm25_files":
        return collect_bm25_files_context(row, **kwargs)
    else:
        raise ValueError(f"Unknown context strategy: {strategy}")