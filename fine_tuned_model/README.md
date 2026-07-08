---
base_model: Mavkif/m2m100_rup_ur_to_rur
library_name: peft
tags:
- base_model:adapter:Mavkif/m2m100_rup_ur_to_rur
- lora
- transformers
- text2text-generation
- urdu
- roman-urdu
- transliteration
metrics:
- bleu
language:
- ur
---

# Model Card for M2M100 Roman Urdu Transliteration Adapter (v1.0 Baseline)

This model is a parameter-efficient fine-tuned version of `Mavkif/m2m100_rup_ur_to_rur` using LoRA adapters to optimize Urdu to Roman Urdu transliteration. It specifically targets domain-specific technical loanwords (e.g., "computer", "doctor") and introduces contextual structural adjustments for high-frequency conversational strings.

## Model Details

### Model Description

- **Developed by:** Maaz Ahmad (NUST)
- **At:** Planet Beyond
- **Shared by:** Maaz Ahmad
- **Model type:** PEFT (LoRA) Sequence-to-Sequence (Encoder-Decoder) Adapter
- **Language(s) (NLP):** Urdu (Native Nastaliq Script) to Roman Urdu (Latin Script Alphabet)
- **License:** MIT
- **Finetuned from model :** `Mavkif/m2m100_rup_ur_to_rur` (Based on `facebook/m2m100_418M`)

### Model Sources

- **Repository:** [GitHub - m2m100-rup-ur-to-rur-fine_tune](https://github.com/Maaz-x14/m2m100-rup-ur-to-rur-fine_tune)


## Uses

### Direct Use

This model is intended for direct programmatic use to transliterate native Urdu text strings into phonetically clean and optimized Roman Urdu text sequences.

### Downstream Use 

- **WhatsApp & Messaging Bot Automation:** Enhancing real-time communication modules where users prefer reading Roman Urdu over native Nastaliq script.
- **Data Pipeline Cleaners:** Building parallel text corpora automatically for training subsequent localized LLMs or text classifiers.
- **Search Engine Query Normalization:** Pre-processing user inputs in modern web applications to index Roman Urdu queries uniformly.

### Out-of-Scope Use

This model is strictly restricted to **transliteration** (script-to-script phonetic mapping) and is completely out of scope for semantic translation (converting the core meaning of Urdu to a separate language like English). It is also out of scope for general multi-turn conversational agents or zero-shot question answering.

## Bias, Risks, and Limitations

Because standard printed Urdu rarely incorporates short-vowel diacritics (*Zair/Pesh*), the native words for **اِس** ("this" / *iss*) and **اُس** ("that" / *uss*) appear identically as **اس**. 
- **Technical Limitation:** The core model adapter operates at roughly a **~50% baseline accuracy rate** on assigning the proper pronoun variants without explicit contextual or structural hints in the sentence.
- **Sociotechnical Risk:** Automated downstream systems utilizing these generations may encounter syntax flips or directional text distortion if output sequences are not strictly guarded.

### Recommendations

Users must deploy a rule-based postprocessor layer immediately following sequence generation to enforce pronoun sanity checks based on neighboring syntax tokens until deeper contextual structural data is introduced in subsequent versions.

## How to Get Started with the Model

Use the code below to get started with the model:

```python
import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from peft import PeftModel

base_model_path = "Mavkif/m2m100_rup_ur_to_rur"
adapter_path = "your-hf-username/m2m100-ur-to-roman-urdu"

# Load customized tokenizer and models
tokenizer = M2M100Tokenizer.from_pretrained("Mavkif/m2m100_rup_tokenizer_both")
base_model = M2M100ForConditionalGeneration.from_pretrained(base_model_path)
model = PeftModel.from_pretrained(base_model, adapter_path)
model.eval()

# Example conversational phrase with domain-specific terms
text = "ڈاکٹر نے کہا کمپیوٹر پر کام کرو اور ورزش کرو"
tokenizer.src_lang = "ur"
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    generated_tokens = model.generate(**inputs, forced_bos_token_id=128105)

print(tokenizer.decode(generated_tokens[0], skip_special_tokens=True)) 
```


## Training Details

### Training Data

Fine-tuned using an expanded parallel corpus of **2,501 curated text sequences** balancing complex domain-specific technical terminology alongside target contrastive pronoun alignments.  

### Training Procedure

#### Preprocessing 

Data tokenization splits inputs into custom localized boundaries (`__ur__`: id 128095 | `__roman-ur__`: id 128105) matching structural constraints within `Mavkif/m2m100_rup_tokenizer_both`.  

#### Training Hyperparameters

- **Training regime:** fp16 mixed precision  
- **Batch Size:** 16 (Effective batch size of 64 via Gradient Accumulation Steps = 4)  
- **Learning Rate:** 5e-4[cite: 1]  
- **LoRA Configurations:** Rank ($r$): 16, Alpha ($\alpha$): 32, Targets: `q_proj`, `k_proj`, `v_proj`, `out_proj`[cite: 1]  

#### Speeds, Sizes, Times

- **Hardware:** Single NVIDIA A5000 (24 GB VRAM)[cite: 1]  
- **Adapter Weights Size:** ~4.7 Million trainable parameters (~0.97% of base architecture)[cite: 1]  

---

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

Evaluated across a distinct 5% validation split comprising a 125 sequence holdout set[cite: 1].  

#### Factors

Evaluations are primarily disaggregated across conversational strings containing English tech loanwords versus pure localized native structures to evaluate code-switching resilience[cite: 1].

#### Metrics

- **BLEU:** SacreBLEU string evaluation metric used to benchmark validation set outputs against reference datasets[cite: 1].

### Results

| Metric | Value |
| :--- | :--- |
| **BLEU Score** | **67.36** |

### Summary

The validation configuration registers a clean 67.36 BLEU score across standard transliteration tasks, though it remains bottle-necked around the un-diacritized pronoun tasks[cite: 1].

---

## Model Examination 

Injected execution overrides onto `PatchedM2M100Model` resolve inherent `transformers 5.x` generational pipeline conflicts associated with handling underlying `decoder_inputs_embeds` arrays[cite: 1].

---

## Environmental Impact

Carbon emissions are calculated using the Machine Learning Impact calculator presented in Lacoste et al. (2019).

- **Hardware Type:** NVIDIA A5000[cite: 1]
- **Hours used:** ~4.5 Hours
- **Cloud Provider:** Local Infrastructure
- **Compute Region:** Islamabad, Pakistan
- **Carbon Emitted:** Minimal trace offset (~0.42 kg $\text{CO}_2\text{eq}$)

---

## Technical Specifications 

### Model Architecture and Objective

Sequence-to-Sequence conditional auto-regressive transformer model paired with an integrated low-rank adaptation matrix layer injected across internal encoder/decoder attention modules[cite: 1].

### Compute Infrastructure

#### Hardware

Enforced single-GPU locked configurations via specific execution scripts to guarantee runtime stability and eliminate catastrophic gradient dispersion under `fp16` parameters[cite: 1].

#### Software

PyTorch 2.1.0+, Transformers 5.0.0+, PEFT 0.19.1[cite: 1].

---

## Citation

If you use this model adapter layer or associated training architectures in your research:

**BibTeX:**

```bibtex
@software{maaz_m2m100_lora_2026,
  author       = {Maaz Ahmad},
  title        = {M2M100 Urdu to Roman Urdu Transliteration LoRA Adapter},
  month        = jul,
  year         = 2026,
  publisher    = {Hugging Face},
  version      = {v1.0},
  url          = {[https://github.com/Maaz-x14/m2m100-rup-ur-to-rur-fine_tune](https://github.com/Maaz-x14/m2m100-rup-ur-to-rur-fine_tune)}
}