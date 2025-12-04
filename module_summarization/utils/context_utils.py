import ast
from typing import Optional
import os 
import re
from rank_bm25 import BM25Okapi

# ========= Existing context utilities =========

def collect_good_context_pathes(row):
    context = ""
    row['good_code_files'] = ast.literal_eval(str(row['good_code_files']).replace('repos', '../data/repos_clean'))
    for con_path in row['good_code_files']:
        with open(con_path, 'r') as f:
            con = f.read()
            context += '\n\n' + con
    context = context.lstrip()
    return context

def collect_good_context(row, strategy, kwargs):
    if strategy == 'ast_clean':
        return clean_python_source(row['relevant_code_context'])
    elif strategy == 'bm25':
        return collect_bm25_files_context(row, **kwargs)
    elif strategy == 'default':
        return row['relevant_code_context']
    
def get_temperature(strategy):
    if strategy == 'bm25':
        return 1.0
    else: 
        return 0.0

def trim_context(context, tokenizer, max_len):
    tokenized_context = tokenizer.encode(context, max_length=512_000, truncation=True)
    tokenized_context = tokenized_context[:max_len]
    detokenized_context = tokenizer.decode(tokenized_context)
    return detokenized_context

#  ========= Code cleaning (imports + main-guard) =========

class NoiseStripper(ast.NodeTransformer):
    """
    - Removes all import statements (Import, ImportFrom)
    - Removes if __name__ == "__main__": blocks
    - Keeps docstrings (module, class, function)
    """

    def visit_Import(self, node: ast.Import) -> Optional[ast.AST]:
        # Drop all import statements
        return None

    def visit_ImportFrom(self, node: ast.ImportFrom) -> Optional[ast.AST]:
        # Drop all from x import y statements
        return None

    def visit_If(self, node: ast.If) -> Optional[ast.AST]:
        # Remove if __name__ == "__main__": blocks
        if self._is_main_guard(node):
            return None
        return self.generic_visit(node)

    @staticmethod
    def _is_main_guard(node: ast.If) -> bool:
        """
        Detects: if __name__ == "__main__":
        """
        test = node.test
        if not isinstance(test, ast.Compare):
            return False
        if not isinstance(test.left, ast.Name):
            return False
        if test.left.id != "__name__":
            return False
        if len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
            return False
        if len(test.comparators) != 1:
            return False

        comp = test.comparators[0]
        if isinstance(comp, ast.Constant) and comp.value == "__main__":
            return True
        # Python <3.8: ast.Str
        if hasattr(ast, "Str") and isinstance(comp, ast.Str) and comp.s == "__main__":
            return True
        return False


def clean_python_source(source: str) -> str:
    """
    Parse Python code, strip imports and main-guard blocks, keep docstrings.

    If parsing fails (e.g., incomplete code), returns the original source.
    """
    if '\x00' in source:
        source = source.replace('\x00', '')

    # Log the source for debugging
    # lines = source.splitlines()
    
    # if len(lines) > 15005:
    #     print("Problematic code around line 15005:")
    #     print("\n".join(lines[15000:15010]))  # Log a few lines before and after

    # Proceed with parsing
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        # print(f"Syntax error in source: {e}") # Log the error for debugging
        return ""

    transformer = NoiseStripper()
    new_tree = transformer.visit(tree)
    ast.fix_missing_locations(new_tree)

    try:
        cleaned = ast.unparse(new_tree)
    except Exception:
        return source

    return cleaned.strip() + "\n"

# BM25 Context
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