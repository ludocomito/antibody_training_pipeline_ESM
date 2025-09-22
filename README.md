# Antibody Developability Analysis with ESM

Antibody developability classifier using ESM protein language model + logistic regression.

## Setup with uv + venv

This project uses [uv](https://github.com/astral-sh/uv) for fast Python package management with virtual environments.

### Prerequisites

Install uv:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Project Setup

1. **Create and activate virtual environment:**
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. **Install dependencies:**
```bash
# Intall via uv
uv sync

# Install main dependencies
uv pip install -e .

# Install with development dependencies
uv pip install -e ".[dev]"
```