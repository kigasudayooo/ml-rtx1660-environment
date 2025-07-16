# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Environment Setup
- `uv sync --group dev` - Install dependencies including dev group
- `uv run python gpu_test.py` - Test GPU availability and RTX 1660 Super compatibility

### Running ML Scripts
- `uv run python src/gpu_test.py` - Comprehensive GPU setup verification
- `uv run python src/cnn_cifar10.py` - Run CNN training on CIFAR-10 (optimized for RTX 1660 Super)
- `uv run python src/tabular_ml.py.py` - Run tabular data classification with deep neural networks
- `uv run python src/transformer_text_classification.py` - Run transformer-based text classification

### Code Quality
- `uv run ruff check` - Lint code
- `uv run ruff format` - Format code
- `uv run mypy src/` - Type checking
- `uv run pytest` - Run tests

## Architecture Overview

This is a machine learning environment specifically optimized for RTX 1660 Super GPU (6GB VRAM). The codebase demonstrates GPU-optimized implementations across different ML domains:

### Hardware Optimization Strategy
- **Memory Management**: All scripts implement VRAM-conscious batch sizes and memory monitoring
- **GPU Utilization**: Uses `torch.backends.cudnn.benchmark = True` and memory fraction limits
- **RTX 1660 Super Specific**: Accounts for 6GB VRAM limitation and lack of Tensor Cores

### Core Components

**GPU Testing (`src/gpu_test.py`)**:
- Comprehensive GPU verification including CUDA, memory, and compute capabilities
- RTX 1660 Super specific validation and performance benchmarking
- Memory allocation testing and cleanup verification

**CNN Implementation (`src/cnn_cifar10.py`)**:
- CIFAR-10 classification with memory-optimized CNN architecture
- Batch size: 32 (optimized for 6GB VRAM)
- Includes memory monitoring and automatic cleanup
- Real-time GPU memory tracking during training

**Tabular ML (`src/tabular_ml.py.py`)**:
- Deep neural network for tabular data classification
- Large batch sizes (256) suitable for non-image data
- Includes learning rate scheduling and comprehensive evaluation
- Visualization of training metrics

**Transformer Processing (`src/transformer_text_classification.py`)**:
- Text classification using DistilBERT (lightweight transformer)
- Gradient accumulation for memory efficiency
- Partial layer freezing to reduce memory usage
- Small batch size (8) with max sequence length 128

### Dependencies and Environment
- **PyTorch**: CUDA 11.8 compatible version from PyTorch index
- **UV Package Manager**: Used for dependency management and virtual environments
- **Development Tools**: Black, Ruff, MyPy, Pytest for code quality
- **ML Libraries**: Complete stack including scikit-learn, pandas, matplotlib, seaborn

### Memory Management Patterns
All scripts implement consistent memory management:
- CUDA memory fraction limits (0.85-0.9)
- Explicit `torch.cuda.empty_cache()` calls after epochs
- Memory monitoring with `torch.cuda.memory_allocated()`
- Pin memory usage in DataLoaders for GPU transfer optimization