# ClassicalReasoningEvals

A TextArena environment for evaluating models on classical mathematical reasoning tasks from standard benchmarks.

## Overview

The ClassicalReasoningEvals environment adapts popular mathematical reasoning benchmarks to the TextArena framework, enabling evaluation of language models on a wide range of mathematical problems. This environment supports both standard accuracy metrics and more advanced evaluation methods such as pass@k, consensus@k, and best@k.

## Features

- **Multiple Evaluation Methods**: Supports standard accuracy as well as pass@k, cons@k, and best@k metrics
- **Comprehensive Problem Types**: Includes grade school math, high school math, college-level math, competition math, and more
- **Robust Answer Checking**: Uses symbolic evaluation and multiple matching strategies to ensure correct answer validation
- **Configurable Sampling**: Ability to use full benchmarks or sample a subset for quicker evaluation

## Available Environments

Here are the registered environments available through TextArena:

### Grade School Math

| Environment ID | Description |
|----------------|-------------|
| `GSM8K-v0` | Full GSM8K test set with standard accuracy |
| `GSM8K-v0-subsampled` | 100 random problems from GSM8K test set |
| `GSM8K-v0-pass@16` | GSM8K with pass@16 evaluation |

### High School / Competition Math

| Environment ID | Description |
|----------------|-------------|
| `AIME-v0` | American Invitational Mathematics Examination 2024 |
| `AIME-v0-pass@16` | AIME 2024 with pass@16 evaluation |
| `AIME23-v0` | American Invitational Mathematics Examination 2023 |
| `OLYMPIADBENCH-v0` | Math Olympiad benchmark |
| `OLYMPIADBENCH-v0-pass@16` | Math Olympiad benchmark with pass@16 evaluation |
| `SAT_MATH-v0` | SAT math problems |

### International Math Exams

| Environment ID | Description |
|----------------|-------------|
| `GAOKAO_MATH-v0` | Chinese Gaokao math exam problems |
| `GAOKAO_MATH_QA-v0` | Chinese Gaokao math Q&A format |
| `GAOKAO2023EN-v0` | Chinese Gaokao 2023 (English translation) |
| `GAOKAO2024_I-v0` | Chinese Gaokao 2024 Part I |
| `GAOKAO2024_II-v0` | Chinese Gaokao 2024 Part II |
| `GAOKAO2024_MCQ-v0` | Chinese Gaokao 2024 multiple choice questions |
| `CN_MIDDLE_SCHOOL-v0` | Chinese middle school math problems |

### College-Level Math

| Environment ID | Description |
|----------------|-------------|
| `CMATH-v0` | College-level mathematics problems |
| `CMATH-v0-pass@16` | College math with pass@16 evaluation |
| `COLLEGE_MATH-v0` | Comprehensive college math benchmark |
| `COLLEGE_MATH-v0-pass@16` | College math benchmark with pass@16 evaluation |
| `MMLU_STEM-v0` | MMLU STEM subjects |

### General Math Collections

| Environment ID | Description |
|----------------|-------------|
| `MATH-v0` | Complete MATH benchmark |
| `MATH-v0-pass@16` | MATH benchmark with pass@16 evaluation | 
| `MATH-v0-subsampled` | 100 random problems from MATH |
| `MATH500-v0` | MATH500 benchmark |
| `MINERVA_MATH-v0` | Minerva math benchmark |

### Word Problems

| Environment ID | Description |
|----------------|-------------|
| `AQUA-v0` | Algebra Question Answering |
| `AQUA-v0-pass@16` | AQUA with pass@16 evaluation |
| `ASDIV-v0` | Arithmetic with diverse operations |
| `CARP_EN-v0` | Complex Arithmetic Problems (English) |
| `MAWPS-v0` | Math Word Problems |
| `SVAMP-v0` | SVAMP benchmark |
| `TABMWP-v0` | Tabular Math Word Problems |


## Evaluation Methods

The environment supports different evaluation protocols:

- **accuracy**: Standard correctness evaluation for each answer
- **x@k metrics**: For more robust evaluation including:
  - **pass@k**: Average correctness across k attempts
  - **cons@k**: Correctness based on consensus of k attempts
  - **best@k**: Binary score if any of k attempts is correct

## Answer Checking

The environment uses a comprehensive answer checking algorithm that handles:

- Exact string matching
- Multiple choice answers
- Numerical equality with appropriate tolerance
- Symbolic mathematical equality
- Matrix and vector equality

## Dataset Attributions

This environment incorporates problems from multiple mathematical benchmarks:

- **GSM8K**: [Cobbe et al., 2021] "Training Verifiers to Solve Math Word Problems"
- **MATH**: [Hendrycks et al., 2021] "Measuring Mathematical Problem Solving With the MATH Dataset" 
- **AIME**: American Invitational Mathematics Examination, licensed by the Mathematical Association of America
- **AQUA**: [Ling et al., 2017] "Program Induction by Rationale Generation: Learning to Solve and Explain Algebraic Word Problems"
- **SVAMP**: [Patel et al., 2021] "Are NLP Models really able to Solve Simple Math Word Problems?"
- **MAWPS**: [Koncel-Kedziorski et al., 2016] "MAWPS: A Math Word Problem Repository"
- **Gaokao**: Chinese National College Entrance Examination
- **Olympiad Bench**: Problems from various Mathematical Olympiads
- **MMLU STEM**: [Hendrycks et al., 2020] "Measuring Massive Multitask Language Understanding"
- **TabMWP**: [Lu et al., 2022] "Dynamic Prompt Learning via Policy Gradient for Semi-structured Mathematical Reasoning"

## Attribution

This environment's evaluation approach is inspired by and adapted from various open-source projects:

- **lm-evaluation-harness**: https://github.com/EleutherAI/lm-evaluation-harness
- **PRM800K**: https://github.com/openai/prm800k
- **CRITIC**: https://github.com/microsoft/ProphetNet/tree/master/CRITIC
- **ToRA**: https://github.com/microsoft/ToRA
- **DeepSeek-Math**: https://github.com/deepseek-ai/DeepSeek-Math
- **Main Code**: https://github.com/hkust-nlp/simpleRL-reason/blob/main/eval/grader.py

## Requirements

- sympy
- latex2sympy2
- importlib-resources

