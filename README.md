# Python Code Plagiarism Detection

This project implements a plagiarism detection system for Python code using a combination of CodeBERT embeddings, Abstract Syntax Tree (AST) features, and style-based similarity metrics. The system is designed to compare two Python code snippets and provide a similarity score, highlighting potential plagiarism.

## Features

- **CodeBERT Embeddings**: Uses pre-trained CodeBERT (`microsoft/codebert-base`) to generate embeddings for the input code.
- **AST Comparison**: Extracts and compares the Abstract Syntax Tree (AST) of the provided Python code, capturing structural similarities.
- **Style Metrics**: Measures code style metrics like average line length, indentation, and blank line ratio.
- **Similarity Scoring**: Combines CodeBERT token similarity, AST structure similarity, and style similarity to generate a final similarity score.
