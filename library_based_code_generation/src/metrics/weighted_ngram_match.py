from codebleu import calc_codebleu

from .metric import Metric

class Weighted_NGram_Match(Metric):
    def __init__(self):
        pass
    
    def score(self, generated_file: str, reference_code: str, unique_apis: list[str]) -> float:
        return calc_codebleu([reference_code], [generated_file], "python")["weighted_ngram_match_score"]
    
    def name(self):
        return "Weighted_NGram_Match"