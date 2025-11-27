import re
from typing import List, Tuple

from rank_bm25 import BM25Okapi

from composers.one_completion_file_composer import OneCompletionFileComposer
from data_classes.datapoint_base import DatapointBase


class BM25Composer(OneCompletionFileComposer):
    """
    BM25-based context composer that ranks files by relevance to the target completion file.
    Uses BM25 scoring to prioritize the most relevant files as context.
    """
    
    def __init__(self, **composer_args):
        # Extract tokenizer_type before passing to parent
        self.tokenizer_type = composer_args.pop('tokenizer_type', 'regex')
        super().__init__(**composer_args)
    
    @staticmethod
    def _tokenize_with_regex(text: str) -> List[str]:
        """
        Tokenizes the input text using a simple regex.
        Separates words, numbers, and punctuation as tokens.
        """
        token_pattern = r"\w+|[^\w\s]"  # Match words, numbers, or punctuation
        return re.findall(token_pattern, text)
    
    @staticmethod
    def _tokenize_with_tiktoken(text: str, model_name: str = "gpt-4o") -> List[str]:
        """
        Tokenizes the input text using tiktoken.
        Encodes text and converts tokens to strings to match BM25 format.
        """
        try:
            import tiktoken
            enc = tiktoken.encoding_for_model(model_name)
            return [str(token) for token in enc.encode(text)]
        except ImportError:
            # Fallback to regex if tiktoken is not available
            return BM25Composer._tokenize_with_regex(text)
    
    def _get_tokenizer(self):
        """Returns the appropriate tokenizer function based on configuration."""
        if self.tokenizer_type == "tiktoken":
            return lambda text: self._tokenize_with_tiktoken(text)
        else:
            return lambda text: self._tokenize_with_regex(text)
    
    def _sort_files_by_bm25_relevance(
        self, 
        context_files: dict[str, str], 
        target_file_path: str, 
        target_file_content: str
    ) -> List[Tuple[str, str]]:
        """
        Sort files by BM25 relevance score to the target file.
        
        Args:
            context_files: Dictionary mapping file paths to their contents
            target_file_path: Path of the file being completed
            target_file_content: Content of the file being completed (for similarity scoring)
            
        Returns:
            List of (file_path, file_content) tuples sorted by relevance (highest first)
        """
        tokenizer = self._get_tokenizer()
        
        # Create a query from the target file (path + content)
        target_query = f"{target_file_path}\n{target_file_content}"
        query_tokens = tokenizer(target_query)
        
        # Tokenize all context files (path + content)
        tokenized_files = []
        file_paths = []
        
        for file_path, file_content in context_files.items():
            file_text = f"{file_path}\n{file_content}"
            tokens = tokenizer(file_text)
            tokenized_files.append(tokens)
            file_paths.append(file_path)
        
        # Create BM25 model and get scores
        if not tokenized_files:
            return []
        
        bm25 = BM25Okapi(tokenized_files)
        scores = bm25.get_scores(query_tokens)
        
        # Sort files by score (highest first) and return with their content
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        
        return [(file_paths[i], context_files[file_paths[i]]) for i in ranked_indices]
    
    def context_composer(self, datapoint: DatapointBase) -> str:
        """
        Compose context using BM25 ranking to prioritize relevant files.
        
        Args:
            datapoint: The datapoint containing context and completion information
            
        Returns:
            Formatted context string with BM25-ranked files
        """
        context = datapoint.get_context()
        completion = datapoint.get_completion()
        repo_name = datapoint.repo_name
        
        assert len(completion) == 1, 'Only one file should be completed'
        completion_path = list(completion.keys())[0]
        completion_content = list(completion.values())[0]
        
        # Remove the completion file from context to avoid self-reference
        context_files = {path: content for path, content in context.items() 
                        if path != completion_path}
        
        # Sort files by BM25 relevance to the completion file
        ranked_files = self._sort_files_by_bm25_relevance(
            context_files, completion_path, completion_content
        )
        
        # Format the context following the existing pattern
        composed_content = [
            path + self.meta_info_sep_symbol + content 
            for path, content in ranked_files
        ]
        
        # Add the completion file path at the end (without content)
        composed_content.append(completion_path + self.meta_info_sep_symbol)
        
        # Create repo metadata
        repo_metainfo = f"{self.extension}{self.lang_sep_symbol}{repo_name}{self.meta_info_sep_symbol}"
        
        return repo_metainfo + self.lang_sep_symbol.join(composed_content)