# Models Directory

## LegalBERT Model

Place your LegalBERT model files in the `legalbert_model/` subdirectory:

```
models/
└── legalbert_model/
    ├── config.json
    ├── pytorch_model.bin
    ├── tokenizer_config.json
    ├── tokenizer.json
    └── vocab.txt
```

The model should be compatible with Hugging Face transformers library and fine-tuned for legal text classification.

## Installation

Once you have the model files:

1. Install required dependencies:
   ```bash
   pip install torch transformers
   ```

2. The LegalBertService will automatically detect and load the model when the server starts.

## Model Requirements

- Should output binary classification (guilty/not guilty)
- Compatible with AutoModelForSequenceClassification
- Supports text truncation and padding
- Returns logits that can be converted to probabilities