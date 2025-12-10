# ME2: Multimodal Explanations for Mathematics Education Benchmark

[![Paper](https://img.shields.io/badge/arXiv-2504.03197-b31b1b.svg)](https://arxiv.org/abs/2504.03197)
[![Website](https://img.shields.io/badge/Website-ME2-blue)](https://me2-benchmark.github.io/)
[![Dataset](https://img.shields.io/badge/🤗-Dataset-yellow)](https://huggingface.co/datasets/jungypark/ME2)
[![AAAI 2026](https://img.shields.io/badge/AAAI-2026-green)](https://aaai.org/conference/aaai/aaai-26/)

> **Explain with Visual Keypoints Like a Real Mentor!**
> *A Benchmark for Multimodal Solution Explanation*

Official implementation of the ME2 benchmark accepted at **AAAI 2026 Main Track**.

## 📋 Overview

ME2 is a comprehensive benchmark designed to evaluate AI systems' ability to generate educational explanations that incorporate visual elements, mirroring how human instructors teach using diagrams and visual aids. The benchmark addresses a critical gap in current AI tutoring systems by focusing on **visually grounded mathematical reasoning**.

### Key Features

- **1,000 math problems** with visual keypoint annotations and explanatory text
- **Two domains**: Geometry and Algebra
- **Three evaluation tasks**:
  - **ME2_ps**: Problem Solving with visual understanding
  - **ME2_figure_caption**: Visual keypoint identification
  - **ME2_solution**: Keypoint-based explanation generation
- **Comprehensive metrics**: ROUGE, BLEU, METEOR, BERTScore

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/jungyangpark/ME2.git
cd ME2

# Install lmms-eval
cd lmms-eval
pip install -e .

# Login to Hugging Face (required for dataset access)
huggingface-cli login

# Run evaluation
python -m accelerate.commands.launch \
    --num_processes=2 \
    -m lmms_eval \
    --model gemma_hf \
    --model_args pretrained="google/gemma-3-12b-it" \
    --tasks ME2 \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix ME2_gemma \
    --output_path ./logs/ \
    --verbosity=DEBUG
```

## 📦 Installation

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (recommended)
- Conda or Micromamba

### Step-by-Step Installation

```bash
# Create environment for your target model
micromamba env create -f environments/<model>_env.yaml
micromamba activate <model>

# Or install from requirements.txt
pip install -r requirements.txt

# Install lmms-eval in editable mode
cd lmms-eval
pip install -e .
```

## 🌍 Environment Configurations

We provide pre-configured environments for various models:

| Environment | Model Type | Key Dependencies |
|-------------|------------|------------------|
| `closedLLMs_env.yaml` | Closed-source APIs | google-generativeai, openai |
| `llava_env.yaml` | LLaVA models | llava, transformers |
| `mathllava_env.yaml` | Math-LLaVA | g-llava, llava |
| `mathpuma_env.yaml` | MathPuma | deepspeed, vllm, flash_attn |
| `molmo_env.yaml` | Molmo | transformers |
| `qwen_env.yaml` | Qwen-VL | qwen-vl-utils, Levenshtein |
| `ursamath_env.yaml` | UrsaMath | vllm, flash_attn |

## 📊 Dataset

The ME2 dataset is hosted on Hugging Face: [`jungypark/ME2`](https://huggingface.co/datasets/jungypark/ME2)

### Dataset Structure

Each instance contains:
- **Problem image**: Original mathematical problem with diagrams
- **Solution image**: Annotated solution with visual keypoints
- **Visual keypoints**: Important visual elements (auxiliary lines, points, angles)
- **Explanatory text**: Reference explanations grounded in visual elements

### Access

```python
from datasets import load_dataset

# Requires authentication
dataset = load_dataset("jungypark/ME2", token=True)
```

## 🎯 Evaluation Tasks

### 1. ME2_ps (Problem Solving)

Solve mathematical problems with visual understanding.

### 2. ME2_figure_caption (Visual Keypoint Identification)

Identify and describe important visual elements in mathematical diagrams.

### 3. ME2_solution (Solution Generation)

Generate complete solutions with visual references.

## 💻 Usage Examples

### Multi-GPU Evaluation

```bash
python -m accelerate.commands.launch \
    --num_processes=4 \
    -m lmms_eval \
    --model llava_hf \
    --model_args pretrained="llava-hf/llava-1.5-7b-hf" \
    --tasks ME2 \
    --batch_size 1 \
    --output_path ./logs/

python -m lmms_eval \
    --model qwen_vl \
    --model_args pretrained="Qwen/Qwen/Qwen2.5-VL-7B-Instruct" \
    --tasks ME2 \
    --batch_size 1 \
    --output_path ./logs/
```

## 🔧 Configuration

### Task YAML Structure

Tasks are defined in `lmms-eval/lmms_eval/tasks/ME2/`:

```
ME2/
├── ME2.yaml                  # Task group definition
├── ME2_solution.yaml         # Solution generation task
├── ME2_ps.yaml              # Problem solving task
├── ME2_figure_caption.yaml  # Visual keypoint task
└── utils.py                 # Evaluation utilities
```

### Custom Generation Parameters

Edit task YAML files to customize:
```yaml
generation_kwargs:
  max_new_tokens: 2048        # Increase for longer outputs
  temperature: 0.7            # Add sampling
  top_p: 0.9
  do_sample: true
```

## 📝 Citation

If you use ME2 in your research, please cite:

```bibtex
@misc{park2025explainvisualkeypointslike,
      title={Explain with Visual Keypoints Like a Real Mentor! A Benchmark for Multimodal Solution Explanation}, 
      author={Jaewoo Park and Jungyang Park and Dongju Jang and Jiwan Chung and Byungwoo Yoo and Jaewoo Shin and Seonjoon Park and Taehyeong Kim and Youngjae Yu},
      year={2025},
      eprint={2504.03197},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2504.03197},  
}
```

## 🙏 Acknowledgments

This project is built on:
- [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) - Evaluation framework
- [Hugging Face Transformers](https://github.com/huggingface/transformers) - Model implementations
- [Accelerate](https://github.com/huggingface/accelerate) - Distributed training