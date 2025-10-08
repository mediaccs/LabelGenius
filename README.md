# LabelGenius

A lightweight research toolkit for **multimodal classification** using **GPT** (text/image/both) and **CLIP** (zero-shot or fine-tuned). Built for quick, reproducible experiments and paper appendices.

---

## Quick Links

- **[Try the Colab Demo](https://colab.research.google.com/drive/1BvVWMQ20i7kkyAYjmz__PDT9OTXGtY3H?usp=sharing)** — Run the full tutorial in your browser
- **Case Studies** — Complete replication scripts in the `Case_analysis/` folder
- **[GitHub Repository](https://github.com/mediaccs/LabelGenius)**

---

## What Can You Do?

LabelGenius helps you:
- **Label with GPT** — Classify text, images, or multimodal data using zero-shot or few-shot prompting
- **Classify with CLIP** — Run zero-shot classification or fine-tune a lightweight model
- **Evaluate results** — Get accuracy, F1 scores, and confusion matrices automatically
- **Estimate costs** — Calculate API expenses before running large-scale experiments

---

## Installation

**From PyPI (recommended):**
```bash
pip install labelgenius
```

**Pinned version:**
```bash
pip install labelgenius==0.1.1
```

**From source:**
```bash
git clone https://github.com/mediaccs/LabelGenius
cd labelgenius
pip install -e .
```

**Requirements:** Python 3.9+, dependencies auto-installed from `pyproject.toml`. To manually install dependencies, see `pyproject.toml` for the full list. For GPT features, get an API key from [OpenAI](https://platform.openai.com/api-keys).

---

## Core Functions

### GPT Classification
| Function | Purpose |
|----------|---------|
| `classification_GPT` | Text classification (zero-shot or few-shot) |
| `generate_GPT_finetune_jsonl` | Prepare training data for fine-tuning |
| `finetune_GPT` | Fine-tune a GPT model |

### CLIP Classification (if data cannot be shared with business)
| Function | Purpose |
|----------|---------|
| `classification_CLIP_0_shot` | Zero-shot classification with CLIP |
| `classification_CLIP_finetuned` | Use a fine-tuned CLIP model |
| `finetune_CLIP` | Fine-tune CLIP on your dataset |

### Utilities
| Function | Purpose |
|----------|---------|
| `auto_verification` | Calculate classification metrics |
| `price_estimation` | Estimate OpenAI API costs |

---

## Quick Start

### Import the Library
```python
from labelgenius import (
    classification_GPT,
    classification_CLIP_0_shot,
    finetune_CLIP,
    classification_CLIP_finetuned,
    auto_verification,
    price_estimation,
)
```

### Prepare Your Data
- **Text only:** CSV/Excel/JSON with your text columns
- **Images only:** Directory of images (with optional ID mapping table)
- **Multimodal:** Table with `image_id` column + images at `image_dir/<image_id>.jpg`

---

## Using GPT Models

LabelGenius supports both **GPT-4.1** and **GPT-5** with different control mechanisms:

### GPT-4.1 (Temperature-Based)
Control randomness with `temperature` (0 = deterministic, 2 = creative):

```python
df = classification_GPT(
    text_path="data/headlines.xlsx",
    category=["sports", "politics", "tech"],
    prompt="Classify this headline into one category.",
    column_4_labeling=["headline"],
    model="gpt-4.1",
    temperature=0.7,
    mode="text",
    output_column_name="predicted_category"
)
```

### GPT-5 (Reasoning and Non-Reasoning)
GPT-5 supports both reasoning and non-reasoning modes controlled by `reasoning_effort`:

```python
df = classification_GPT(
    text_path="data/headlines.xlsx",
    category=["sports", "politics", "tech"],
    prompt="Classify this headline into one category.",
    column_4_labeling=["headline"],
    model="gpt-5",
    reasoning_effort="medium",  # minimal | low | medium | high
    mode="text",
    output_column_name="predicted_category"
)
```

**Reasoning Modes:**
- `"minimal"` — Non-reasoning mode (fast, standard responses)
- `"low"` / `"medium"` / `"high"` — Reasoning modes (increasing reasoning depth)

---

## Function Reference

### `classification_GPT`

Classify text, images, or multimodal data using GPT models.

**Parameters:**
- `text_path` — Path to your data file (CSV/Excel/JSON)
- `category` — List of possible category labels
- `prompt` — Instruction prompt for classification
- `column_4_labeling` — Column(s) to use as input
- `model` — Model name (e.g., "gpt-4.1", "gpt-5")
- `temperature` — Randomness control for GPT-4.1 (0-2)
- `reasoning_effort` — Reasoning depth for GPT-5 ("minimal", "low", "medium", "high")
- `mode` — Input type: "text", "image", or "both"
- `output_column_name` — Name for prediction column
- `num_themes` — Number of themes for multi-label tasks (default: 1)
- `num_votes` — Number of runs for majority voting (default: 1)

**Example (Single-Label Text):**
```python
df = classification_GPT(
    text_path="data/articles.csv",
    category=["positive", "negative", "neutral"],
    prompt="Classify the sentiment of this article.",
    column_4_labeling=["title", "content"],
    model="gpt-4.1",
    temperature=0.5,
    mode="text",
    output_column_name="sentiment"
)
```

**Example (Multi-Label with Majority Voting):**
```python
prompt = """Label these 5 topics with 0 or 1. Return up to two 1s.
Answer format: [0,0,0,0,0]"""

df = classification_GPT(
    text_path="data/posts.xlsx",
    category=["0", "1"],
    prompt=prompt,
    column_4_labeling=["post_text"],
    model="gpt-5",
    reasoning_effort="low",
    mode="text",
    output_column_name="topic_labels",
    num_themes=5,
    num_votes=3  # Run 3 times and use majority vote
)
```

**Note:** You can control the number of labels by adjusting the prompt. For example:
- "Return exactly one 1" — Single label
- "Return up to two 1s" — Up to 2 labels
- "Return up to three 1s" — Up to 3 labels
- "Return any number of 1s" — Unrestricted multi-label

**Example (Multimodal):**
```python
df = classification_GPT(
    text_path="data/products.csv",
    category=["electronics", "clothing", "food"],
    prompt="Classify this product based on text and image.",
    column_4_labeling=["product_name", "description"],
    model="gpt-4.1",
    temperature=0.3,
    mode="both",  # Uses both text and images
    image_dir="data/product_images",
    output_column_name="category"
)
```

---

### `generate_GPT_finetune_jsonl`

Prepare training data in JSONL format for GPT fine-tuning.

**Parameters:**
- `df` — DataFrame containing your training data
- `output_path` — Where to save the JSONL file
- `system_prompt` — Instruction prompt for the model
- `input_col` — Column(s) to use as input
- `label_col` — Column(s) containing true labels

**Example:**
```python
generate_GPT_finetune_jsonl(
    df=training_data,
    output_path="finetune_data.jsonl",
    system_prompt="Classify the sentiment of this review.",
    input_col=["review_text"],
    label_col=["sentiment"]
)
```

---

### `finetune_GPT`

Launch a GPT fine-tuning job using OpenAI's API.

**Parameters:**
- `training_file_path` — Path to your JSONL training file
- `model` — Base model to fine-tune (must be a snapshot model, i.e., model name ending with a specific date, e.g., "gpt-4o-mini-2024-07-18")
- `hyperparameters` — Dict with batch_size, learning_rate_multiplier, etc.

**Returns:** Fine-tuned model identifier

**Example:**
```python
model_id = finetune_GPT(
    training_file_path="finetune_data.jsonl",
    model="gpt-4o-mini-2024-07-18",
    hyperparameters={
        "batch_size": 8,
        "learning_rate_multiplier": 0.01,
        "n_epochs": 3
    }
)
print(f"Fine-tuned model: {model_id}")
```

**Note:** GPT-5 fine-tuning availability varies. Use GPT-4 snapshots if needed.

**Using the fine-tuned model:**

After fine-tuning completes, you'll receive a model ID via email or in the Python output. Use this ID with `classification_GPT` by replacing the `model` parameter:

```python
# Use your fine-tuned model for classification
df = classification_GPT(
    text_path="data/test.csv",
    category=["positive", "negative", "neutral"],
    prompt="Classify the sentiment of this review.",
    column_4_labeling=["review_text"],
    model="ft:gpt-4o-mini-2024-07-18:your-org:model-name:abc123",  # Your fine-tuned model ID
    temperature=0.5,
    mode="text",
    output_column_name="sentiment"
)
```

---

### `classification_CLIP_0_shot`

Perform zero-shot classification using CLIP without training.

**Parameters:**
- `text_path` — Path to your data file
- `img_dir` — Directory containing images (optional)
- `mode` — Input type: "text", "image", or "both"
- `prompt` — List of category names/descriptions for zero-shot
- `text_column` — Column(s) to use for text input
- `predict_column` — Name for prediction column

**Example (Text Only):**
```python
df = classification_CLIP_0_shot(
    text_path="data/articles.csv",
    mode="text",
    prompt=["sports news", "political news", "technology news"],
    text_column=["headline", "summary"],
    predict_column="clip_category"
)
```

**Example (Image Only):**
```python
df = classification_CLIP_0_shot(
    text_path="data/image_ids.csv",
    img_dir="data/images",
    mode="image",
    prompt=["a photo of a cat", "a photo of a dog", "a photo of a bird"],
    predict_column="animal_type"
)
```

**Example (Multimodal):**
```python
df = classification_CLIP_0_shot(
    text_path="data/posts.csv",
    img_dir="data/post_images",
    mode="both",
    prompt=["advertisement", "personal photo", "news image"],
    text_column=["caption"],
    predict_column="image_category"
)
```

---

### `finetune_CLIP`

Fine-tune CLIP on your labeled dataset with a small classification head.

**Parameters:**
- `mode` — Input type: "text", "image", or "both"
- `text_path` — Path to training data
- `text_column` — Column(s) for text input (if using text)
- `img_dir` — Image directory (if using images)
- `true_label` — Column containing true labels
- `model_name` — Filename to save the trained model
- `num_epochs` — Number of training epochs
- `batch_size` — Training batch size
- `learning_rate` — Learning rate for optimizer

**Returns:** Best validation accuracy achieved

**Example:**
```python
best_acc = finetune_CLIP(
    mode="both",
    text_path="data/train.csv",
    text_column=["headline", "description"],
    img_dir="data/train_images",
    true_label="category_id",
    model_name="my_clip_model.pth",
    num_epochs=10,
    batch_size=16,
    learning_rate=1e-5
)
print(f"Best validation accuracy: {best_acc:.2%}")
```

---

### `classification_CLIP_finetuned`

Use your fine-tuned CLIP model to classify new data.

**Parameters:**
- `mode` — Input type: "text", "image", or "both"
- `text_path` — Path to test data
- `img_dir` — Image directory (if using images)
- `model_name` — Path to your trained model file
- `text_column` — Column(s) for text input
- `predict_column` — Name for prediction column
- `num_classes` — Number of categories in your task

**Example:**
```python
df = classification_CLIP_finetuned(
    mode="both",
    text_path="data/test.csv",
    img_dir="data/test_images",
    model_name="my_clip_model.pth",
    text_column=["headline", "description"],
    predict_column="predicted_category",
    num_classes=24
)
```

---

### `auto_verification`

Calculate classification metrics including accuracy, F1 scores, and confusion matrix.

**Parameters:**
- `df` — DataFrame with predictions and true labels
- `predicted_cols` — List of prediction column names
- `true_cols` — List of true label column names
- `category` — List of possible categories

**Example:**
```python
auto_verification(
    df=results,
    predicted_cols=["gpt_pred", "clip_pred"],
    true_cols=["true_label", "true_label"],
    category=["0", "1"]
)
```

**Output:** Prints accuracy, F1 scores, and displays confusion matrix visualization.

---

### `price_estimation`

Estimate the cost of OpenAI API calls.

**Parameters:**
- `response` — OpenAI API response object
- `num_rows` — Number of data rows processed
- `input_cost_per_million` — Input token cost (in USD per 1M tokens)
- `output_cost_per_million` — Output token cost (in USD per 1M tokens)
- `num_votes` — Number of API calls per row (for majority voting)

**Returns:** Estimated total cost in USD

**Example:**
```python
cost = price_estimation(
    response=api_response,
    num_rows=1000,
    input_cost_per_million=5.0,
    output_cost_per_million=15.0,
    num_votes=3
)
print(f"Estimated total cost: ${cost:.2f}")
```

---

## Important Notes

**Demo Scale:**  
The Colab demo uses ~20 samples per task for speed. Expect high variance in small-sample results.

**Training Data Requirements:**  
Fine-tuning typically requires hundreds of examples per class. Small datasets (~20 examples) may lead to overfitting or majority-class predictions.

**Image Requirements:**  
For multimodal classification, ensure your `image_id` column matches filenames in `image_dir`.

**Model Availability:**  
GPT-5 fine-tuning may not be available for all accounts. Use GPT-4 snapshots as alternatives.


---

## Contributing

We welcome contributions! Feel free to open an issue for bugs or feature requests, or submit a pull request to improve the codebase and documentation.

---

## License

LabelGenius is released under the MIT License.

