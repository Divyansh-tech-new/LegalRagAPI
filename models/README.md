# Models Directory

## LegalBERT Model

You can add your fine-tuned LegalBERT model in two ways:

### Option 1: Zip File (Recommended)
Place your model zip file as `legalbert_epoch4.zip` in this directory:
```
models/
└── legalbert_epoch4.zip
```

The system will automatically extract it to `legalbert_model/` when the server starts.

### Option 2: Direct Files
Place your LegalBERT model files directly in the `legalbert_model/` subdirectory:
```
models/
└── legalbert_model/
    ├── config.json
    ├── pytorch_model.bin
    ├── tokenizer_config.json
    ├── tokenizer.json
    └── vocab.txt
```

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

## Auto-Detection

The service checks for models in this order:
1. `legalbert_epoch4.zip` (extracts automatically)
2. `legalbert_model/` directory with model files
3. Falls back to placeholder mode if neither found