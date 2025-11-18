# Perseus Citation Model

Machine learning models for identifying citation structures in classical texts and resolving bibliographic references to canonical URNs.

**Project Status:** 🚧 Early Development

- ✅ Data pipeline implemented (extraction task)
- ✅ Model initialization and embedding handling
- ✅ Comprehensive test suite (98 tests passing)
- ⏳ Training scripts (in progress)
- ⏳ URN resolution implementation (planned)

## Installation

**Requirements:**

- Python 3.13+
- [uv](https://github.com/astral-sh/uv) package manager (recommended) or pip

**Setup:**

```bash
# Clone repository
git clone https://github.com/your-username/perseus-citation-model.git
cd perseus-citation-model

# Install with uv (recommended)
uv sync

# Or install with pip
pip install -e ".[dev]"
```

**Dependencies:**

- `transformers` - HuggingFace transformers (DeBERTa models)
- `torch` - PyTorch for model training
- `datasets` - HuggingFace datasets library
- `beautifulsoup4` + `lxml` - XML parsing
- `sentencepiece` + `protobuf` - Tokenizer support
- `pytest` + `pytest-mock` - Testing (dev dependency)

## Quick Start

```python
from perscit_model.extraction.data_loader import ExtractionDataLoader, create_extraction_dataset
from perscit_model.extraction.model import create_model

# Load data
loader = ExtractionDataLoader()
dataset = create_extraction_dataset("cit_data/resolved.jsonl")

# Create model
model = create_model(loader.tokenizer)

# Train (coming soon - see implementation examples below)
```

## Overview

This project provides two complementary ML tasks for working with citations in TEI-encoded XML documents from the Perseus Digital Library:

1. **Tag Extraction**: Identify and extract citation tags (`<cit>`, `<quote>`, `<bibl>`) from plain text
2. **URN Resolution**: Map bibliographic references to Canonical Text Services (CTS) URNs

Both tasks share data pipelines and preprocessing infrastructure but use different model architectures appropriate to each problem.

## Task Definitions

### Task 1: Tag Extraction

**Input:** Plain text extracted from TEI XML documents
**Output:** Token-level tags identifying citation boundaries

**Target tags:**

- `<cit>` - Citation container
- `<quote>` - Quoted text
- `<bibl>` - Bibliographic reference

**Challenges:**

- Nested structures (e.g., `<cit>` contains `<quote>` and `<bibl>`)
- Variable citation formats
- Mixed languages (Greek, Latin, English)
- Context-dependent identification

### Task 2: URN Resolution

**Input:** Bibliographic reference text (e.g., "Hdt. 8.82")
**Output:** CTS URN (e.g., "urn:cts:greekLit:tlg0016.tlg001.perseus-grc2:8.82")

**Examples:**

- "Hom. Il. 7.268" → "urn:cts:greekLit:tlg0012.tlg001.perseus-grc2:7.268"
- "Thuc. 3.38" → "urn:cts:greekLit:tlg0003.tlg001.perseus-grc2:3.38"
- "Plat. Rep. 332D" → "urn:cts:greekLit:tlg0059.tlg030.perseus-grc2:332d"

**Challenges:**

- Abbreviated author names (Hdt., Hom., Thuc.)
- Work title variations and abbreviations
- Range notation (e.g., "7.268-272", "sqq.")
- Unresolvable references to modern scholarship (e.g., "ARV2, 987")
- Missing URNs for ~12% of citations

## Data Format

Training data is in JSONL format with two files:

**`resolved.jsonl` (~216K examples)** - Citations with URNs:

```json
{
  "bibl": "Hdt. 8.82",
  "quote": "",
  "xml_context": "...full XML context with tags...",
  "filename": "xml_files/viaf17286815.viaf001.xml",
  "urn": "urn:cts:greekLit:tlg0016.tlg001.perseus-grc2:8.82",
  "ref": "hdt. 8.82",
  "n_attrib": "Hdt. 8.82",
  "doc_cit_urn": ":citations-28.3"
}
```

**`unresolved.jsonl` (~30K examples)** - Citations without URNs:

```json
{
  "bibl": "FR, pl. 167,2",
  "quote": "",
  "xml_context": "...full XML context with tags...",
  "filename": "xml_files/viaf114145308.viaf001.xml",
  "urn": "",
  "ref": "fr pl. 167,2",
  "n_attrib": "",
  "doc_cit_urn": ":citations-24.1"
}
```

**Key fields:**

- `bibl`: Bibliographic reference text
- `quote`: Quoted text (often empty)
- `xml_context`: XML snippet with tags for tag extraction training
- `urn`: CTS URN (empty for unresolved citations)
- `ref`: Normalized reference text

---

# Task 1: Tag Extraction

## Approach 1: Transformer-based Token Classification

### Overview

Fine-tune a pre-trained transformer model (DeBERTa, RoBERTa, or BERT) for sequence labeling using BIO tagging.

### Architecture

```
Input Text → Tokenizer → Transformer Encoder → Linear Layer → Softmax → BIO Tags
```

### BIO Tagging Scheme

Each token is labeled with one of:

- `O` - Outside any citation tag
- `B-CIT` - Beginning of `<cit>` tag
- `I-CIT` - Inside `<cit>` tag
- `B-QUOTE` - Beginning of `<quote>` tag
- `I-QUOTE` - Inside `<quote>` tag
- `B-BIBL` - Beginning of `<bibl>` tag
- `I-BIBL` - Inside `<bibl>` tag

**Example:**

```
Text:     Hom.  Il.  7.268  -  272  :  "Ajax  hurled  a  rock"
Tags:     B-BIBL I-BIBL I-BIBL I-BIBL I-BIBL I-BIBL O B-QUOTE I-QUOTE I-QUOTE I-QUOTE
```

### Model Selection

**Recommended models (in order of preference):**

1. **`microsoft/deberta-v3-base`** - Best for this task due to:
   - Superior contextual understanding for nested structures
   - Better multilingual handling (Greek, Latin, English)
   - State-of-the-art performance on token classification
   - 1-3% F1 improvement over RoBERTa on similar tasks

2. **`roberta-base`** - Good alternative if:
   - Need faster inference (~10-15% faster than DeBERTa)
   - Memory constraints
   - Strong baseline performance

3. **`bert-base-uncased`** - Use only for:
   - Quick prototyping
   - Establishing baseline metrics

### Current Implementation (Special Tokens Approach)

**See actual implementation in `src/perscit_model/extraction/`**

```python
from perscit_model.extraction.data_loader import (
    ExtractionDataLoader,
    create_extraction_dataset,
    parse_xml_to_bio,
    generate_bio_labels
)
from perscit_model.extraction.model import create_model
from transformers import Trainer, TrainingArguments

# 1. Create data loader (adds special tokens to DeBERTa)
loader = ExtractionDataLoader()  # Automatically adds 6 special tokens

# 2. Create dataset from JSONL
#    - Parses XML, replaces tags with special tokens
#    - Tokenizes with DeBERTa
#    - Generates BIO labels from special token positions
dataset = create_extraction_dataset("cit_data/resolved.jsonl")

# 3. Create model (resizes embeddings for special tokens)
model = create_model(loader.tokenizer)

# 4. Training configuration
training_args = TrainingArguments(
    output_dir="./outputs/extraction/models",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# 5. Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    # eval_dataset=val_dataset,  # Create from separate file
)

trainer.train()
```

**Key differences from standard token classification:**

1. **No word alignment** - Special tokens handle tag boundaries
2. **Simpler labels** - Generated automatically from special token positions
3. **Malformed XML handling** - BeautifulSoup repairs broken tags
4. **Nested tags** - Inner tag takes precedence in BIO scheme

### Pros

- Simple to implement with HuggingFace
- Strong baseline performance
- Fast training with GPU
- Well-documented and widely used
- Transfer learning from pre-trained models

### Cons

- May produce invalid tag sequences (e.g., `I-QUOTE` without `B-QUOTE`)
- Doesn't explicitly model tag dependencies
- Treats each token prediction independently

---

## Approach 2: Transformer + CRF (Hybrid)

### Overview

Combines transformer contextual embeddings with a Conditional Random Field (CRF) layer to ensure valid tag sequences.

### Architecture

```
Input Text → Tokenizer → Transformer Encoder → Linear Layer → CRF Layer → BIO Tags
```

### Why Add CRF?

The CRF layer learns transition probabilities between tags, ensuring:

- Valid sequences (e.g., `I-QUOTE` must follow `B-QUOTE` or `I-QUOTE`)
- No impossible transitions (e.g., `B-CIT` → `I-QUOTE` directly)
- Global optimization across the entire sequence
- Structured prediction with dependencies

### Implementation

```python
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn
from torchcrf import CRF

class BertCRF(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        emissions = self.classifier(sequence_output)

        if labels is not None:
            # Training: compute CRF loss
            loss = -self.crf(emissions, labels, mask=attention_mask.byte(), reduction='mean')
            return loss
        else:
            # Inference: Viterbi decoding
            predictions = self.crf.decode(emissions, mask=attention_mask.byte())
            return predictions

# Initialize model
labels = ["O", "B-CIT", "I-CIT", "B-QUOTE", "I-QUOTE", "B-BIBL", "I-BIBL"]
model = BertCRF("bert-base-uncased", num_labels=len(labels))

# Training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

for epoch in range(num_epochs):
    model.train()
    for batch in train_dataloader:
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        loss = model(input_ids, attention_mask, labels)
        loss.backward()
        optimizer.step()

# Inference
model.eval()
with torch.no_grad():
    predictions = model(input_ids, attention_mask)
```

### CRF Layer Details

**What CRF learns:**

- Transition matrix: probability of tag A followed by tag B
- Start transitions: which tags can begin a sequence
- End transitions: which tags can end a sequence

**Example constraints:**

```
Valid:   B-QUOTE → I-QUOTE → I-QUOTE → O
Valid:   O → B-BIBL → I-BIBL → O
Invalid: O → I-QUOTE (can't start with I-)
Invalid: B-QUOTE → I-BIBL (can't switch tag types)
```

### Installation

```bash
pip install pytorch-crf
# or
pip install torchcrf
```

### Pros

- Guaranteed valid tag sequences
- Models dependencies between adjacent tags
- Often 1-3% better F1 score than plain transformers
- Theoretically sound structured prediction
- Better handling of rare tag transitions

### Cons

- More complex implementation
- Slightly slower training/inference
- Additional hyperparameters to tune
- Requires more memory (stores transition matrix)

---

## Comparison

| Feature               | Transformer Only             | Transformer + CRF        |
| --------------------- | ---------------------------- | ------------------------ |
| **Implementation**    | Simple (HuggingFace Trainer) | Moderate (custom model)  |
| **Training Speed**    | Fast                         | 10-20% slower            |
| **Sequence Validity** | No guarantee                 | Guaranteed               |
| **Performance**       | Good baseline                | Usually 1-3% better      |
| **Memory**            | Lower                        | Higher                   |
| **Best for**          | Quick experiments, baseline  | Production, max accuracy |

---

## Data Preparation Pipeline

**Actual implementation:** See `src/perscit_model/extraction/data_loader.py`

### Pipeline Overview

```
JSONL file → parse_xml_to_bio() → tokenize → generate_bio_labels() → Dataset
```

### Step 1: XML → Special Tokens

```python
from perscit_model.extraction.data_loader import parse_xml_to_bio

xml = '<bibl n="Hdt. 8.82">Hdt. 8.82</bibl> some context'
processed = parse_xml_to_bio(xml)
# Output: " [BIBL_START]  Hdt. 8.82  [BIBL_END]  some context"
```

**What it does:**

1. Parse XML with BeautifulSoup (repairs malformed XML)
2. Remove attributes from `<bibl>`, `<quote>`, `<cit>` tags
3. Replace tags with special tokens surrounded by spaces
4. Preserve other tags (`<title>`, `<author>`, etc.)

### Step 2: Tokenize with DeBERTa

```python
loader = ExtractionDataLoader()  # Adds special tokens to vocabulary
tokens = loader.tokenizer(processed)
# DeBERTa tokenizes, special tokens won't be split
```

### Step 3: Generate BIO Labels

```python
from perscit_model.extraction.data_loader import generate_bio_labels

labels = generate_bio_labels(tokens.input_ids[0], loader.tokenizer)
# State machine: [BIBL_START] triggers B-BIBL, subsequent tokens get I-BIBL
```

### Handling Nested Tags

For nested structures like `<cit><bibl>Hdt. 8.82</bibl></cit>`:

**Strategy: Inner tag takes precedence**

- Text inside `<bibl>` gets `B-BIBL`/`I-BIBL` labels
- `<cit>` markers present but tokens labeled by innermost tag
- Model learns nested structure from marker positions

### Dataset Splitting

```python
# TODO: Implement train/val/test split
# Recommended: Split by document to avoid data leakage
```

---

## Evaluation Metrics

```python
from seqeval.metrics import classification_report, f1_score

# seqeval properly handles BIO tagging
predictions = ["O", "B-BIBL", "I-BIBL", "O", "B-QUOTE", "I-QUOTE"]
ground_truth = ["O", "B-BIBL", "I-BIBL", "O", "B-QUOTE", "I-QUOTE"]

print(classification_report([ground_truth], [predictions]))
```

**Key metrics:**

- **Precision:** Of predicted tags, how many are correct?
- **Recall:** Of actual tags, how many did we find?
- **F1-score:** Harmonic mean of precision and recall
- **Entity-level F1:** Full span must match (stricter)

---

## Training Recommendations

### Hyperparameters

**Transformer only:**

- Learning rate: 2e-5 to 5e-5
- Batch size: 16-32
- Epochs: 3-5
- Warmup steps: 500
- Weight decay: 0.01

**Transformer + CRF:**

- Learning rate: 5e-5 (CRF may need slightly higher)
- Batch size: 16 (CRF uses more memory)
- Epochs: 3-5
- CRF learning rate: Can be different from BERT (e.g., 1e-3)

### Hardware Requirements

**Minimum:**

- GPU: 8GB VRAM (e.g., RTX 2070, T4)
- RAM: 16GB
- Storage: 10GB for model + data

**Recommended:**

- GPU: 16GB+ VRAM (e.g., V100, A100, RTX 3090)
- RAM: 32GB
- Storage: 50GB

### Training Time Estimates

With ~200MB JSONL data on single GPU:

- Data preparation: 1-2 hours
- Transformer training (3 epochs): 2-4 hours
- Transformer+CRF training (3 epochs): 3-5 hours

---

# Task 2: URN Resolution

## Overview

URN resolution maps bibliographic reference strings to Canonical Text Services (CTS) URNs. This is a different ML problem from tag extraction - it's a **structured prediction** or **sequence-to-sequence** task rather than token classification.

## Recommended Approaches

### Approach 1: Rule-based + Hierarchical Classification (Recommended)

**Strategy:** Use perseus-citation-processor for 87.6% of cases, hierarchical DNN classifiers for the remaining 12.4%.

**Advantages:**

- **Guaranteed valid URNs** - classification over known catalog, no hallucination
- **Interpretable** - see which stage (author/work) succeeded or failed
- **Efficient** - small vocabularies (~500 authors, ~50 works per author)
- **High precision** - rule-based handles well-formatted citations (87.6%)
- **Debuggable** - can inspect author confidence vs work confidence separately

**Architecture:**

```
Input: "Hdt. 8.82"
  ↓
[perseus-citation-processor] → 87.6% resolved directly
  ↓ (if unresolved or low confidence)
[Author Classifier (DNN)] → tlg0016 [conf: 0.95]
  ↓
[Work Classifier (DNN)] → tlg0016.tlg001 [conf: 0.92]
  ↓
[Passage Parser (rules)] → 8.82
  ↓
[Edition Selector (rules)] → perseus-grc2
  ↓
Assemble: "urn:cts:greekLit:tlg0016.tlg001.perseus-grc2:8.82"
```

**Components:**

1. **Rule-based baseline**: perseus-citation-processor (Go binary)
2. **Author classifier**: DeBERTa classification over ~500 Greek/Latin authors
3. **Work classifier**: DeBERTa classification over works (conditioned on author)
4. **Passage parser**: Rule-based extraction of passage references
5. **Edition selector**: Rule-based (Greek quote → grc, Latin quote → lat)
6. **Confidence scorer**: DeBERTa binary classifier for quality control

### Approach 2: Sequence-to-Sequence Model (NOT Recommended)

**Models:** T5, BART, ByT5, or encoder-decoder transformers

**Why NOT recommended:**

- **URNs are structured catalog lookups**, not creative text generation
- **Can hallucinate** invalid author/work combinations
- **Less interpretable** - black box generation
- **Wasteful** - learns URN syntax instead of just citation→URN mappings
- **No guaranteed validity** - requires complex constrained decoding

**When to consider:**

- If you need to handle completely new authors not in any catalog (unlikely for classical texts)
- If you want to experiment with end-to-end learning
- As a baseline to compare against hierarchical classification

**Architecture:**

```python
from transformers import T5ForConditionalGeneration, AutoTokenizer

# ByT5 for character-level handling of Greek/Latin
model = T5ForConditionalGeneration.from_pretrained("google/byt5-base")
tokenizer = AutoTokenizer.from_pretrained("google/byt5-base")

# Training format: "resolve citation: Hdt. 8.82" → "urn:cts:greekLit:..."
input_text = "resolve citation: Hdt. 8.82"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs)

# Major issue: May generate syntactically valid but semantically wrong URNs
# e.g., "urn:cts:greekLit:tlg9999.tlg999.perseus-grc2:8.82" (invalid author)
```

**Verdict:** Use hierarchical classification (Approach 1) instead.

### Approach 3: Retrieval-Augmented Generation

**Strategy:** Retrieve similar citations from database, use ML to rank/select.

**Advantages:**

- Leverages existing resolved citations
- Good for rare author/work combinations
- Can explain predictions via retrieved examples

**Architecture:**

```
Input: "Thuc. 3.38"
  ↓
Embedding model → Dense vector representation
  ↓
Vector DB → Retrieve top-K similar resolved citations
  ↓
Ranking model → Score candidates
  ↓
Output: Highest-scoring URN
```

---

## Implementation: Hybrid Rule-based + DNN System

**Recommended Approach:** Hierarchical Classification (NOT seq2seq)

This implementation uses:

1. **Rule-based baseline** (perseus-citation-processor) for 87.6% of citations
2. **Hierarchical DNN classifiers** for the remaining 12.4%
   - Author classifier (DeBERTa)
   - Work classifier (DeBERTa, conditioned on author)
3. **Confidence scorer** (DeBERTa) to validate rule-based outputs

**Why hierarchical classification over seq2seq?**

- Guaranteed valid URNs (no hallucination)
- Interpretable decisions (author vs work failures)
- Efficient (small vocabularies)
- Better uncertainty handling (top-k at each stage)

### Foundation: perseus-citation-processor

This project uses the **[perseus-citation-processor](https://github.com/andrewbird2/perseus-citation-processor)** as the rule-based foundation:

**Performance on 246K citations:**

- ✅ **87.6% resolution rate** (216K resolved)
- ❌ **12.4% unresolved** (30K citations)
- ⚡ Fast: 28 seconds for 125 XML files
- 📚 Comprehensive author/work mappings (Greek, Latin, Scholia)

The DNN components handle the remaining 12.4% plus add confidence scoring.

### Architecture Overview

```
Input Citation
       ↓
[perseus-citation-processor] ← Rule-based (87.6% coverage)
       ↓
   ┌───┴────┐
   ↓        ↓
Resolved  Unresolved
   ↓        ↓
[Confidence  [DNN Resolution
 Scorer]      Model]
   ↓        ↓
High conf?  URN + conf
   ↓        ↓
Yes→Output  Merge→Output
   No↘    ↗
    [DNN Model]
```

### DNN Component 1: Unresolved Citation Resolver

**Purpose:** Resolve the 30K citations that rule-based system couldn't handle.

**Approach:** Hierarchical Classification (NOT seq2seq generation)

**Why Classification over Seq2Seq:**

- **URNs are structured catalog lookups**, not creative text
- **Guaranteed valid outputs** - cannot hallucinate invalid author/work combinations
- **Interpretable** - see exactly which stage succeeded/failed
- **Efficient** - small vocabulary (~500 authors × ~50 works = 25K combinations)
- **Better uncertainty handling** - top-k predictions at each stage

**Architecture:** Multi-stage classification pipeline

```
"Hdt. 8.82"
    ↓
[Author Classifier] → tlg0016 (Herodotus) [confidence: 0.95]
    ↓
[Work Classifier] → tlg0016.tlg001 (Histories) [confidence: 0.92]
    ↓
[Edition Selector] → perseus-grc2 (rule-based: Greek text → grc)
    ↓
[Passage Parser] → 8.82 (rule-based extraction)
    ↓
Assemble: "urn:cts:greekLit:tlg0016.tlg001.perseus-grc2:8.82"
```

**Implementation:**

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

class AuthorClassifier:
    """Stage 1: Classify citation to author URN"""

    def __init__(self):
        # Load perseus-citation-processor author catalog
        self.author_catalog = self.load_author_catalog()  # ~500 Greek/Latin authors
        self.author_to_id = {urn: i for i, urn in enumerate(self.author_catalog)}
        self.id_to_author = {i: urn for urn, i in self.author_to_id.items()}

        # DeBERTa classifier over author vocabulary
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/deberta-v3-base",
            num_labels=len(self.author_catalog)
        )
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

    def predict(self, citation_text, context=""):
        """Predict top-k authors with confidence scores"""
        # Input format: "[citation] [SEP] [context]"
        text = f"{citation_text} [SEP] {context}"
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

        # Get predictions
        outputs = self.model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        top_k = torch.topk(probs[0], k=3)

        # Return [(author_urn, confidence), ...]
        return [(self.id_to_author[idx.item()], prob.item())
                for idx, prob in zip(top_k.indices, top_k.values)]


class WorkClassifier:
    """Stage 2: Classify to work URN (conditioned on author)"""

    def __init__(self):
        # Group works by author
        self.works_by_author = self.load_work_catalog()  # {tlg0016: [tlg001, tlg002, ...]}
        self.max_works = max(len(works) for works in self.works_by_author.values())

        # Classifier conditioned on author
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/deberta-v3-base",
            num_labels=self.max_works
        )
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

    def predict(self, citation_text, author_urn, context=""):
        """Predict work for given author"""
        # Get candidate works for this author
        candidates = self.works_by_author.get(author_urn, [])
        if not candidates:
            return []

        # Input format: "[author] [SEP] [citation] [SEP] [context]"
        text = f"{author_urn} [SEP] {citation_text} [SEP] {context}"
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

        outputs = self.model(**inputs)

        # Filter to only valid works for this author
        valid_logits = outputs.logits[0, :len(candidates)]
        probs = torch.softmax(valid_logits, dim=-1)

        # Return [(work_urn, confidence), ...] sorted by confidence
        results = [(candidates[i], probs[i].item()) for i in range(len(candidates))]
        return sorted(results, key=lambda x: x[1], reverse=True)


class HierarchicalURNResolver:
    """Complete hierarchical URN resolution system"""

    def __init__(self):
        self.author_classifier = AuthorClassifier()
        self.work_classifier = WorkClassifier()
        self.passage_parser = PassageParser()  # Rule-based
        self.edition_selector = EditionSelector()  # Rule-based

    def resolve(self, citation_text, context="", quote=""):
        """Resolve citation to URN with confidence score"""
        # Stage 1: Classify author
        author_candidates = self.author_classifier.predict(citation_text, context)

        # Stage 2: For each author candidate, classify work
        urn_candidates = []
        for author_urn, author_conf in author_candidates[:3]:  # Top-3 authors
            work_candidates = self.work_classifier.predict(
                citation_text, author_urn, context
            )

            for work_urn, work_conf in work_candidates[:3]:  # Top-3 works
                # Stage 3: Parse passage (rule-based)
                passage = self.passage_parser.extract(citation_text)

                # Stage 4: Select edition (rule-based)
                edition = self.edition_selector.select(author_urn, quote)

                # Stage 5: Assemble URN
                namespace = self.get_namespace(author_urn)  # greekLit or latinLit
                full_urn = f"urn:cts:{namespace}:{work_urn}.{edition}:{passage}"

                # Combined confidence (product of stages)
                confidence = author_conf * work_conf

                urn_candidates.append((full_urn, confidence, {
                    'author_conf': author_conf,
                    'work_conf': work_conf,
                    'author': author_urn,
                    'work': work_urn
                }))

        # Return best candidate
        if urn_candidates:
            best = max(urn_candidates, key=lambda x: x[1])
            return best[0], best[1], best[2]
        else:
            return None, 0.0, {}
```

**Training Data Format:**

```python
# Training examples from resolved.jsonl (216K examples)
# Split into author and work classification tasks

# Stage 1: Author Classification
author_examples = [
    {
        "citation": "Hdt. 8.82",
        "context": "",
        "label": "tlg0016"  # Herodotus
    },
    {
        "citation": "Soph. OT 151",
        "context": "τᾶς πολυχρύσου Πυθῶνος",
        "label": "tlg0011"  # Sophocles
    },
    {
        "citation": "Plat. Rep. 332D",
        "context": "",
        "label": "tlg0059"  # Plato
    }
]

# Stage 2: Work Classification (conditioned on author)
work_examples = [
    {
        "citation": "Hdt. 8.82",
        "author": "tlg0016",
        "context": "",
        "label": "tlg0016.tlg001"  # Histories
    },
    {
        "citation": "Soph. OT 151",
        "author": "tlg0011",
        "context": "τᾶς πολυχρύσου Πυθῶνος",
        "label": "tlg0011.tlg004"  # Oedipus Tyrannus
    },
    {
        "citation": "Hom. Il. 7.268",
        "author": "tlg0012",
        "context": "Ajax hurled a rock",
        "label": "tlg0012.tlg001"  # Iliad (not Odyssey)
    }
]
```

**Features Used:**

1. **Citation text** (required): "Hdt. 8.82"
2. **Context** (when available): Surrounding text or quote
3. **Author URN** (for work classification): Condition on stage 1 output
4. **Ground truth URN**: Parse author/work from `urn` field

### DNN Component 2: Confidence Scorer

**Purpose:** Identify incorrect resolutions from perseus-citation-processor.

**Problem:** From perseus-citation-processor README: "just because the processor resolves a citation doesn't mean that it resolves it correctly"

**Model:** DeBERTa-based Binary Classifier

```python
from transformers import AutoModel, AutoTokenizer
import torch.nn as nn

class URNConfidenceScorer(nn.Module):
    def __init__(self, model_name="microsoft/deberta-v3-base"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)

        hidden_size = self.encoder.config.hidden_size
        feature_size = 10  # Engineered features

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size + feature_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, citation_text, urn, context=""):
        # Encode: "[citation] <SEP> [URN] <SEP> [context]"
        text = f"{citation_text} [SEP] {urn} [SEP] {context}"
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)

        outputs = self.encoder(**inputs)
        pooled = outputs.last_hidden_state[:, 0, :]  # CLS token

        # Compute engineered features
        features = self.compute_features(citation_text, urn, context)

        combined = torch.cat([pooled, features], dim=1)
        confidence = self.classifier(combined)

        return confidence

    def compute_features(self, citation, urn, context):
        """Engineered features for confidence scoring"""
        return torch.tensor([
            self.citation_urn_match_score(citation, urn),
            self.has_ambiguous_abbrev(citation),
            self.context_language_match(context, urn),
            self.passage_validity(urn),
            self.author_frequency(urn),
            self.work_frequency(urn),
            self.quote_presence(context),
            self.citation_length(citation),
            self.urn_complexity(urn),
            self.catalog_match_count(citation)
        ])
```

**Training Data:**

```python
# Positive examples (confident resolutions)
positive_examples = [
    {
        "citation": "Hdt. 8.82",
        "urn": "urn:cts:greekLit:tlg0016.tlg001.perseus-grc2:8.82",
        "context": "Greek text context",
        "label": 1.0  # High confidence
    }
]

# Ambiguous cases (medium confidence)
ambiguous_examples = [
    {
        "citation": "Arist. Met.",  # Metaphysics or Meteorology?
        "urn": "urn:cts:greekLit:tlg0086.tlg025.perseus-grc2:",
        "context": "",
        "label": 0.6  # Medium confidence
    }
]

# Incorrect resolutions (low confidence)
negative_examples = [
    {
        "citation": "Hom. 7.268",  # Iliad or Odyssey?
        "urn": "urn:cts:greekLit:tlg0012.tlg002.perseus-grc2:7.268",  # Odyssey
        "context": "Ajax hurled a rock",  # Ajax is in Iliad!
        "label": 0.1  # Low confidence - context mismatch
    }
]
```

### DNN Component 3: Hybrid Orchestrator

**Purpose:** Combine rule-based and DNN intelligently.

```python
class HybridURNResolver:
    def __init__(self):
        self.rule_based = PerseusProcessorWrapper()  # Calls Go binary
        self.dnn_resolver = HierarchicalURNResolver()  # Hierarchical classifier
        self.confidence_scorer = URNConfidenceScorer()

        # Thresholds (tune on validation set)
        self.high_confidence_threshold = 0.85
        self.low_confidence_threshold = 0.50

    def resolve(self, citation_text, context="", quote=""):
        # Step 1: Try rule-based
        rule_urn, rule_status = self.rule_based.resolve(citation_text)

        # Step 2: Score rule-based result
        if rule_urn:
            confidence = self.confidence_scorer(citation_text, rule_urn, context)

            if confidence > self.high_confidence_threshold:
                return rule_urn, confidence, "rule-based"

            # Medium confidence - get DNN opinion
            elif confidence > self.low_confidence_threshold:
                dnn_urn, dnn_conf, dnn_details = self.dnn_resolver.resolve(
                    citation_text, context, quote
                )

                # Compare: rule-based vs DNN
                if dnn_conf > confidence:
                    return dnn_urn, dnn_conf, "dnn-override", dnn_details
                else:
                    return rule_urn, confidence, "rule-based-verified"

            # Low confidence - prefer DNN
            else:
                dnn_urn, dnn_conf, dnn_details = self.dnn_resolver.resolve(
                    citation_text, context, quote
                )
                return dnn_urn, dnn_conf, "dnn-preferred", dnn_details

        # Step 3: Rule-based failed, use DNN only
        else:
            dnn_urn, dnn_conf, dnn_details = self.dnn_resolver.resolve(
                citation_text, context, quote
            )
            return dnn_urn, dnn_conf, "dnn-only", dnn_details
```

### Training Strategy

**Phase 1: Train Author Classifier**

```python
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)
import json

# Step 1: Extract author labels from URNs
def parse_urn_to_author(urn):
    """Extract author URN from full CTS URN"""
    # urn:cts:greekLit:tlg0016.tlg001.perseus-grc2:8.82 → tlg0016
    try:
        parts = urn.split(":")
        author_work = parts[3]  # tlg0016.tlg001.perseus-grc2
        author = author_work.split(".")[0]  # tlg0016
        return author
    except:
        return None

# Step 2: Build author vocabulary from perseus-citation-processor data
author_catalog = set()
with open("cit_data/resolved.jsonl") as f:
    for line in f:
        item = json.loads(line)
        author = parse_urn_to_author(item['urn'])
        if author:
            author_catalog.add(author)

author_to_id = {author: i for i, author in enumerate(sorted(author_catalog))}
id_to_author = {i: author for author, i in author_to_id.items()}

# Step 3: Prepare training data
train_data = []
with open("cit_data/resolved.jsonl") as f:
    for line in f:
        item = json.loads(line)
        author = parse_urn_to_author(item['urn'])
        if author and author in author_to_id:
            train_data.append({
                "text": f"{item['bibl']} [SEP] {item.get('quote', '')}",
                "label": author_to_id[author]
            })

# Train/val split
dataset = Dataset.from_list(train_data)
dataset = dataset.train_test_split(test_size=0.1)

# Step 4: Train DeBERTa classifier
model = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/deberta-v3-base",
    num_labels=len(author_catalog)
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir="./outputs/resolution/models/author-classifier",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['test'],
)

trainer.train()
```

**Phase 2: Train Work Classifier**

```python
# Step 1: Parse work labels from URNs
def parse_urn_to_work(urn):
    """Extract work URN from full CTS URN"""
    # urn:cts:greekLit:tlg0016.tlg001.perseus-grc2:8.82 → tlg0016.tlg001
    try:
        parts = urn.split(":")
        author_work = parts[3]  # tlg0016.tlg001.perseus-grc2
        work = ".".join(author_work.split(".")[:2])  # tlg0016.tlg001
        return work
    except:
        return None

# Step 2: Build work vocabulary grouped by author
works_by_author = {}
with open("cit_data/resolved.jsonl") as f:
    for line in f:
        item = json.loads(line)
        author = parse_urn_to_author(item['urn'])
        work = parse_urn_to_work(item['urn'])
        if author and work:
            if author not in works_by_author:
                works_by_author[author] = set()
            works_by_author[author].add(work)

# Step 3: Prepare training data (conditioned on author)
train_data = []
with open("cit_data/resolved.jsonl") as f:
    for line in f:
        item = json.loads(line)
        author = parse_urn_to_author(item['urn'])
        work = parse_urn_to_work(item['urn'])

        if author and work:
            # Input includes author URN to condition on
            train_data.append({
                "text": f"{author} [SEP] {item['bibl']} [SEP] {item.get('quote', '')}",
                "author": author,
                "label": work
            })

# Create label mapping (per-author work indices)
# ... (similar training setup)
```

**Phase 3: Train Confidence Scorer**

- Curate labeled examples with confidence scores
- Use cross-validation on resolved.jsonl
- Add manually labeled ambiguous cases (~1K examples)
- Generate synthetic negative examples (incorrect author/work pairs)

**Phase 4: Evaluate Hierarchical Classifier**

- Test on held-out validation set (10% of resolved.jsonl)
- Measure component accuracy:
  - Author classification accuracy
  - Work classification accuracy (given correct author)
  - End-to-end URN exact match
- Test on 30K unresolved.jsonl examples
- Manual evaluation on random sample for accuracy

### Expected Performance

**Conservative estimates:**

| Component                                    | Coverage Improvement |
| -------------------------------------------- | -------------------- |
| Rule-based baseline                          | 87.6%                |
| DNN on unresolved (30-50% success)           | +4-6%                |
| Confidence filtering (catch incorrect rules) | +2-3%                |
| **Total hybrid coverage**                    | **~94-97%**          |

**Key insight:** Even resolving 1/3 of unresolved cases is significant improvement.

## Evaluation Metrics

**For URN Resolution:**

- **Exact match accuracy**: Percentage of perfectly resolved URNs
- **Component accuracy**: Separate metrics for author, work, passage
- **Coverage**: Percentage of citations with confident predictions
- **Precision@K**: Accuracy when allowing top-K predictions

## Data Split Considerations

Unlike tag extraction, URN resolution should be split by **unique citation patterns** rather than documents to test generalization:

- Test on unseen author abbreviations
- Test on unseen work titles
- Test on seen authors but unseen works

---

## Next Steps

### Tag Extraction

1. **Data exploration:** Analyze `resolved.jsonl` statistics
2. **Preprocessing:** Build data pipeline from JSONL to BIO format
3. **Baseline model:** Train DeBERTa token classifier
4. **Evaluation:** Measure performance on held-out test set
5. **Advanced model:** Implement DeBERTa+CRF if baseline is insufficient
6. **Error analysis:** Examine failure cases

### URN Resolution

**Phase 1: Rule-based Baseline (Week 1-2)**

1. **Integration:** Wrap perseus-citation-processor as Python callable
2. **Baseline metrics:** Establish 87.6% resolution rate on 216K citations
3. **Error analysis:** Analyze 30K unresolved.jsonl patterns
4. **Catalog extraction:** Load author/work URNs from perseus-citation-processor data

**Phase 2: Hierarchical Classifiers (Week 3-5)**

1. **Catalog extraction:** Extract author/work vocabularies from resolved.jsonl URNs
2. **Train author classifier:** DeBERTa classification over ~500 authors
   - Input: citation + context
   - Output: author URN (e.g., tlg0016)
3. **Train work classifier:** DeBERTa classification conditioned on author
   - Input: author + citation + context
   - Output: work URN (e.g., tlg0016.tlg001)
4. **Implement passage parser:** Rule-based extraction of passage references
5. **Evaluate components:** Measure author accuracy, work accuracy, end-to-end
6. **Test on unresolved:** Evaluate on 30K unresolved.jsonl citations

**Phase 3: Confidence Scorer (Week 5-6)**

1. **Training data curation:** Label confident vs ambiguous resolutions
2. **Train DeBERTa classifier:** (citation, URN, context) → confidence score
3. **Threshold tuning:** Calibrate confidence thresholds on validation set
4. **Cross-validation:** Test on ambiguous cases from resolved.jsonl

**Phase 4: Hybrid System (Week 7-8)**

1. **Orchestrator implementation:** Combine rule-based + DNN components
2. **End-to-end evaluation:** Measure coverage improvement (target: 94-97%)
3. **Component analysis:** Track rule-based vs DNN vs hybrid performance
4. **Error analysis:** Identify remaining failure modes

---

## Project Structure

```
perseus-citation-model/
│
├── cit_data/                          # Raw training data (not in repo)
│   ├── resolved.jsonl                 # 216K citations with URNs
│   └── unresolved.jsonl               # 30K citations without URNs
│
├── model_data/                        # Partitioned data
│   ├── extraction                     # Partitions for extraction task
│   └── resolution                     # Partitions for resolution task

├── outputs/                           # Fine-tuned model weights from training

├── src/                               # Source code
│   └── perscit_model/                 # Main package
│       ├── __init__.py
│       ├── shared/                    # Shared utilities across tasks
│       │   ├── __init__.py
│       │   ├── data_loader.py         # Base JSONL loader, tokenization
│       │   └── training_utils.py      # Training configuration utilities
│       ├── extraction/                # Task 1: Tag Extraction (BIO tagging)
│       │   ├── __init__.py
│       │   ├── data_loader.py         # XML → special tokens → BIO labels
│       │   └── model.py               # DeBERTa token classification model
│       └── resolution/                # Task 2: URN Resolution
│           ├── __init__.py
│           └── data_loader.py         # Citation data loading for resolution
│
├── configs/                           # Configuration files
│   └── extraction/
│       └── baseline.yaml              # Hyperparameters (model, max_length)
│
├── tests/                             # Test suite (98 tests)
│   ├── conftest.py                    # Shared fixtures (mock tokenizer)
│   ├── fixtures/                      # Test data
│   │   └── sample_extraction.jsonl   # 5 real citation examples
│   ├── unit/                          # Fast unit tests (88 tests, ~3s)
│   │   ├── test_extraction_dataset.py    # BIO label generation tests
│   │   ├── test_extraction_loader.py     # Data loader tests
│   │   ├── test_extraction_pipeline.py   # End-to-end pipeline tests
│   │   ├── test_resolution_loader.py     # Resolution data tests
│   │   └── test_shared_data_loader.py    # Shared utility tests
│   └── integration/                   # Slow integration tests (10 tests, ~8s)
│       └── test_extraction_model.py      # Real model loading/training tests
│
├── pyproject.toml                     # Project config, dependencies, test settings
├── .gitignore
├── .python-version                    # Python 3.13
└── README.md
```

**Key Implementation Details:**

### Extraction Data Pipeline (Special Tokens Approach)

Instead of word-level BIO tagging with complex alignment, we use **special tokens**:

1. **XML → Special Tokens**: Replace `<bibl>`, `<quote>`, `<cit>` tags with `[BIBL_START]`, `[BIBL_END]`, etc.
2. **Add to Vocabulary**: Special tokens added to DeBERTa tokenizer (won't be split)
3. **Tokenize**: DeBERTa tokenizes text with special tokens intact
4. **Generate BIO Labels**: State machine generates labels based on special token positions

**Example:**

```
XML:      <bibl>Hdt. 8.82</bibl> some context
↓
Special:  [BIBL_START] Hdt. 8.82 [BIBL_END] some context
↓
Tokens:   [CLS] [BIBL_START] Hdt . 8 . 82 [BIBL_END] some context [SEP]
↓
Labels:   -100  -100          B-  I- I- I- I- -100        O    O       -100
```

**Advantages over word-level alignment:**

- No complex subword↔word alignment logic
- Special tokens guaranteed not to split
- Simpler, more reliable label generation
- Handles malformed XML gracefully (BeautifulSoup repair)

### Model Initialization

**Embedding Resizing:**

- Base DeBERTa vocab: 128,000 tokens
- +6 special tokens = 128,006 tokens
- New embeddings initialized to **mean of existing embeddings** (training stability)

### Testing

```bash
# Fast unit tests only (default)
pytest                    # 88 tests in ~3s

# Integration tests (downloads real models)
pytest tests/integration  # 10 tests in ~8s

# All tests
pytest tests              # 98 tests in ~9s
```

---

## End-to-End Pipeline

The two tasks can be combined into a complete citation processing pipeline:

```
Raw Text → Tag Extraction → URN Resolution → Structured Citations
```

**Example workflow:**

1. **Input**: "Homer mentions this in Il. 7.268-272: 'Ajax hurled a rock'"
2. **Tag Extraction**: Identify `<bibl>Il. 7.268-272</bibl>` and `<quote>Ajax hurled a rock</quote>`
3. **URN Resolution**: Map "Il. 7.268-272" → "urn:cts:greekLit:tlg0012.tlg001.perseus-grc2:7.268"
4. **Output**: Structured citation with linked canonical reference

This enables:

- Automated citation extraction from plain text
- Linking to canonical text passages
- Cross-referencing across documents
- Building citation networks in classical scholarship

---

## Useful Links

**General:**

- TEI Guidelines: <https://tei-c.org/release/doc/tei-p5-doc/en/html/>
- CTS URN Specification: <http://cite-architecture.github.io/cts_spec/>
- Perseus Digital Library: <https://www.perseus.tufts.edu/>

**Tag Extraction:**

- HuggingFace Token Classification: <https://huggingface.co/docs/transformers/tasks/token_classification>
- pytorch-crf: <https://github.com/kmkurn/pytorch-crf>
- seqeval metrics: <https://github.com/chakki-works/seqeval>
- DeBERTa paper: <https://arxiv.org/abs/2006.03654>

**URN Resolution:**

- DeBERTa paper: <https://arxiv.org/abs/2006.03654>
- Hierarchical classification: <https://arxiv.org/abs/1904.02817>
- CTS URN Specification: <http://cite-architecture.github.io/cts_spec/>
- CTS API documentation: <http://cite-architecture.github.io/cts/>
- perseus-citation-processor: <https://github.com/andrewbird2/perseus-citation-processor>
- Fuzzy string matching: <https://github.com/seatgeek/fuzzywuzzy>
