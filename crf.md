# Adding CRF Layer for Improved BIO Tagging

## What is CRF?

Conditional Random Fields (CRF) is a sequence modeling layer that enforces valid
BIO tag transitions. For citation extraction, this means:

- **Prevents invalid sequences**: Can't have `I-BIBL` without `B-BIBL` first
- **Models tag dependencies**: Probability of a tag depends on previous tags
- **Improves boundary detection**: Better at finding exact start/end of
  citations

**Expected improvement:** +0.5-2% F1 score for sequence labeling tasks

## Installation

```bash
pip install pytorch-crf
```

Or add to `pyproject.toml`:

```toml
[project.dependencies]
pytorch-crf = "^0.7.2"
```

## Implementation

### 1. Create Custom Model with CRF Layer

Create `src/perscit_model/extraction/model_crf.py`:

```python
"""Generic transformer + CRF model for citation extraction.

Works with any AutoModel (DeBERTa, RoBERTa, BERT, etc.)
"""

import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import AutoModel, PreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput


class AutoModelForTokenClassificationWithCRF(PreTrainedModel):
    """Generic transformer model with CRF layer for token classification.

    This model adds a CRF layer on top of any AutoModel encoder to enforce
    valid BIO tag transitions. Works with DeBERTa, RoBERTa, BERT, etc.
    """

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        # Load the base encoder (e.g., DeBERTa, RoBERTa, BERT)
        self.encoder = AutoModel.from_config(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(config.num_labels, batch_first=True)

        # Initialize weights
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Pass through encoder (works with any transformer)
        outputs = self.encoder(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            # Create mask for CRF (exclude -100 labels)
            crf_mask = labels != -100

            # Replace -100 with 0 for CRF (it will be masked anyway)
            labels_for_crf = labels.clone()
            labels_for_crf[~crf_mask] = 0

            # CRF loss (negative log-likelihood)
            loss = -self.crf(logits, labels_for_crf, mask=crf_mask, reduction='mean')

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def decode(self, logits, attention_mask):
        """Decode using Viterbi algorithm to get best tag sequence.

        Args:
            logits: Model output logits [batch_size, seq_len, num_labels]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            List of best tag sequences
        """
        return self.crf.decode(logits, mask=attention_mask.bool())
```

### 2. Update `create_model()` Function

Modify `src/perscit_model/extraction/model.py`:

```python
from perscit_model.extraction.model_crf import DebertaV2ForTokenClassificationWithCRF

def create_model(
    tokenizer: PreTrainedTokenizerBase,
    config_path: Path | str | None = None,
    pretrained_model_path: Path | str | None = None,
    use_crf: bool = False,  # NEW PARAMETER
) -> PreTrainedModel:
    """Create a token classification model for citation extraction.

    Args:
        tokenizer: Tokenizer with special tokens
        config_path: Path to YAML config file
        pretrained_model_path: Optional path to pretrained checkpoint
        use_crf: If True, add CRF layer on top of DeBERTa

    Returns:
        Token classification model
    """
    # Load config
    if config_path is None:
        config_path = DEFAULT_CONFIG
    config = TrainingConfig.from_yaml(config_path)

    # Determine model path to load from
    if pretrained_model_path is not None:
        model_path = str(pretrained_model_path)
    else:
        model_path = config.model_name

    # Choose model class based on use_crf flag
    if use_crf:
        from transformers import AutoConfig
        model_config = AutoConfig.from_pretrained(
            model_path,
            num_labels=len(BIO_LABELS),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )
        model = DebertaV2ForTokenClassificationWithCRF.from_pretrained(
            model_path,
            config=model_config,
            ignore_mismatched_sizes=True,
        )
    else:
        model = AutoModelForTokenClassification.from_pretrained(
            model_path,
            num_labels=len(BIO_LABELS),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            ignore_mismatched_sizes=True,
        )

    # Resize embeddings (only if loading from base model)
    if pretrained_model_path is None:
        model.resize_token_embeddings(len(tokenizer))

        with torch.no_grad():
            old_embeddings = model.get_input_embeddings().weight[
                : -len(SPECIAL_TOKENS), :
            ]
            mean_embedding = old_embeddings.mean(dim=0)
            model.get_input_embeddings().weight[-len(SPECIAL_TOKENS) :, :] = (
                mean_embedding
            )

    return model
```

### 3. Update Training Script

Pass `use_crf=True` when calling `create_model()`:

```python
# In train.py, around line 419
model = create_model(
    loader.tokenizer,
    config_path=config_path,
    pretrained_model_path=pretrained_model_path,
    use_crf=True,  # ADD THIS
)
```

Or add to config file:

```yaml
# configs/extraction/baseline.yaml
use_crf: true
```

### 4. Update Inference (Prediction)

Modify `src/perscit_model/extraction/inference.py` to use CRF decoding:

```python
def predict(self, encoding):
    """Predict labels using CRF Viterbi decoding if model has CRF."""
    inputs_on_device = {
        k: v.to(self.device)
        for k, v in encoding.items()
        if k != "offset_mapping"
    }

    with torch.no_grad():
        outputs = self.model(**inputs_on_device)

    logits = outputs.logits

    # Check if model has CRF layer
    if hasattr(self.model, 'decode'):
        # Use Viterbi decoding
        attention_mask = inputs_on_device['attention_mask']
        predictions = self.model.decode(logits, attention_mask)[0]  # First in batch
    else:
        # Use argmax (standard approach)
        predictions = logits.argmax(dim=-1).squeeze().tolist()

    labels = [ID2LABEL[p] for p in predictions]
    return labels
```

## Configuration Changes

### Training Config

```yaml
# configs/extraction/baseline_crf.yaml
model_name: microsoft/deberta-v3-base # or deberta-v3-large
use_crf: true
max_length: 512

# CRF may need slightly more epochs to converge
num_train_epochs: 6 # Up from 5
learning_rate: 0.00003

# Everything else same as baseline
per_device_train_batch_size: 16
fp16: true
early_stopping_patience: 3
```

### Training Scripts

Update Phase 1/2/3 scripts to use CRF config:

```python
# scripts/train_extraction_phase1_crf.py
train_pipeline(
    data_dir=PHASE_1_PARTITION_DIR,
    src_path=PHASE_1_SRC_PATH,
    config_path=Path(__file__).parent.parent / "configs/extraction/baseline_crf.yaml"
)
```

## Expected Results

### Without CRF (Current)

- Phase 3 Test F1: **0.9736**
- BIBL F1: ~0.99
- QUOTE F1: ~0.98

### With CRF (Expected)

- Phase 3 Test F1: **0.975-0.980** (+0.1-0.4%)
- Better boundary detection
- No invalid BIO sequences (I before B)
- More consistent predictions

## Trade-offs

### Pros ✅

- Better sequence consistency
- Improved boundary detection
- Enforces valid BIO transitions
- Often improves F1 by 0.5-2% on NER tasks

### Cons ❌

- ~10-20% slower training (CRF computation overhead)
- ~15-25% slower inference (Viterbi decoding)
- Slightly more complex model architecture
- One more hyperparameter to tune (CRF learning)

## Testing

```python
# Quick test that CRF is working
from perscit_model.extraction.model_crf import DebertaV2ForTokenClassificationWithCRF

model = create_model(loader.tokenizer, use_crf=True)
assert hasattr(model, 'crf'), "Model should have CRF layer"
assert hasattr(model, 'decode'), "Model should have decode method"
```

## When to Use CRF

**Use CRF if:**

- ✅ You want maximum accuracy
- ✅ BIO constraint violations are a problem
- ✅ You have GPU resources (training will be slower)
- ✅ Inference latency is not critical

**Skip CRF if:**

- ❌ Current performance is already excellent (97%+)
- ❌ You need fast inference
- ❌ Training time is a bottleneck
- ❌ Model already rarely violates BIO constraints

## Alternative: Post-processing BIO Constraint Fixing

If CRF is too slow, you can add post-processing to fix invalid sequences:

```python
def fix_bio_sequence(labels):
    """Fix invalid BIO sequences (e.g., I-BIBL without B-BIBL)."""
    fixed = []
    current_tag = None

    for label in labels:
        if label.startswith('I-'):
            tag_type = label[2:]  # e.g., 'BIBL' from 'I-BIBL'
            if current_tag != tag_type:
                # Invalid I without B - change to B
                fixed.append(f'B-{tag_type}')
                current_tag = tag_type
            else:
                fixed.append(label)
        elif label.startswith('B-'):
            current_tag = label[2:]
            fixed.append(label)
        else:  # 'O'
            current_tag = None
            fixed.append(label)

    return fixed
```

This is **much faster** than CRF but less sophisticated (doesn't consider all
constraints during training).

## References

- [pytorch-crf](https://github.com/kmkurn/pytorch-crf) - CRF implementation
- [Conditional Random Fields](https://en.wikipedia.org/wiki/Conditional_random_field) -
  Theory
- [Linear-Chain CRF Tutorial](https://people.cs.umass.edu/~mccallum/papers/crf-tutorial.pdf) -
  Detailed explanation
