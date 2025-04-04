import pandas as pd
import ast
import tokenize
import io
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
import re
import os
from transformers import AutoTokenizer, AutoModel
import torch


class PythonPlagiarismDetector:
    def __init__(self):
        # Initialize CodeBERT
        self.model_name = "microsoft/codebert-base"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)

    def get_codebert_embedding(self, code):
        tokens = self.tokenizer(code, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = self.model(**tokens)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    def preprocess_code(self, code):
        code = re.sub(r'#.*', '', code)
        code = re.sub(r'""".*?"""', '', code, flags=re.DOTALL)
        code = re.sub(r"'''.*?'''", '', code, flags=re.DOTALL)
        code = re.sub(r'\s+', ' ', code).strip()
        return code

    def extract_tokens(self, code):
        try:
            tokens = []
            tokens_generator = tokenize.tokenize(io.BytesIO(code.encode('utf-8')).readline)
            for token in tokens_generator:
                if token.type not in [tokenize.COMMENT, tokenize.NEWLINE, tokenize.INDENT,
                                      tokenize.DEDENT, tokenize.NL]:
                    tokens.append(token.string)
            return ' '.join(tokens)
        except:
            return self.preprocess_code(code)

    def extract_ast_features(self, code):
        try:
            tree = ast.parse(code)
            node_types = {}
            for node in ast.walk(tree):
                node_type = type(node).__name__
                node_types[node_type] = node_types.get(node_type, 0) + 1
            function_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            class_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
            return {
                'node_types': node_types,
                'function_names': function_names,
                'class_names': class_names
            }
        except:
            return {'node_types': {}, 'function_names': [], 'class_names': []}

    def extract_style_metrics(self, code):
        lines = code.split('\n')
        metrics = {
            'avg_line_length': np.mean([len(line) for line in lines]) if lines else 0,
            'max_indentation': max([len(line) - len(line.lstrip()) for line in lines]) if lines else 0,
            'blank_line_ratio': sum(1 for line in lines if not line.strip()) / len(lines) if lines else 0,
        }
        return metrics

    def compare_codes(self, code1, code2):
        code1_clean = self.preprocess_code(code1)
        code2_clean = self.preprocess_code(code2)

        # CodeBERT embeddings
        emb1 = self.get_codebert_embedding(code1_clean)
        emb2 = self.get_codebert_embedding(code2_clean)

        token_similarity = float(cosine_similarity([emb1], [emb2])[0][0])

        ast1 = self.extract_ast_features(code1_clean)
        ast2 = self.extract_ast_features(code2_clean)
        node_types1 = set(ast1['node_types'].keys())
        node_types2 = set(ast2['node_types'].keys())
        node_type_similarity = len(node_types1.intersection(node_types2)) / max(len(node_types1.union(node_types2)), 1)

        style1 = self.extract_style_metrics(code1)
        style2 = self.extract_style_metrics(code2)

        style_similarity = 0

        overall_similarity = 0.5 * token_similarity + 0.3 * node_type_similarity + 0.2 * style_similarity

        return {
            'token_similarity': token_similarity,
            'ast_similarity': node_type_similarity,
            'style_similarity': style_similarity,
            'overall_similarity': overall_similarity
        }

    # def detect_plagiarism(self, submissions, threshold=0.8):
    #     suspicious_pairs = []
    #     similarity_matrix = np.zeros((len(submissions), len(submissions)))
    #     submission_ids = list(submissions.keys())
    #
    #     for i in range(len(submission_ids)):
    #         for j in range(i + 1, len(submission_ids)):
    #             id1, id2 = submission_ids[i], submission_ids[j]
    #             code1, code2 = submissions[id1], submissions[id2]
    #             result = self.compare_codes(code1, code2)
    #             similarity = result['overall_similarity']
    #
    #             similarity_matrix[i, j] = similarity
    #             similarity_matrix[j, i] = similarity
    #
    #             if similarity > threshold:
    #                 suspicious_pairs.append((id1, id2, similarity))
    #
    #     return suspicious_pairs, similarity_matrix, submission_ids

    # def detect_plagiarism_from_files(self, directory, threshold=0.8):
    #     submissions = {}
    #     for filename in os.listdir(directory):
    #         if filename.endswith('.py'):
    #             file_path = os.path.join(directory, filename)
    #             with open(file_path, 'r', encoding='utf-8') as f:
    #                 try:
    #                     code = f.read()
    #                     submissions[filename] = code
    #                 except:
    #                     print(f"Error reading {filename}")
    #     return self.detect_plagiarism(submissions, threshold)


detector = PythonPlagiarismDetector()


