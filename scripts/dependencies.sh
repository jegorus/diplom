#!/bin/bash
pip install -U accelerate # to fix: ImportError: Using the `Trainer` with `PyTorch` requires `accelerate>=0.20.1`: Please run `pip install transformers[torch]` or `pip install accelerate -U`
pip install -U transformers
pip install datasets
pip install peft