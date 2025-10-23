#Urdu-Baat-Cheet

[Live demo](https://urdu-baat-cheet.streamlit.app/)

Urdu-Baat-Cheet is a compact Urdu dialogue model trained end-to-end with a custom BPE tokenizer and a lightweight Transformer (~8.4M params). We extract prompt–response pairs from final_main_dataset.tsv (tab-separated), clean/normalize Unicode and spacing, and train BPE on cleaned_urdu_text.txt to ~7.5k subword units to keep sequences short and reduce OOVs. Any tokenizer artifacts (e.g., </w>) are stripped at display/eval time.

The code supports optional span corruption (infilling): random spans are replaced by sentinel tokens (e.g., <SENT0>, <SENT1>) and the model learns to reconstruct them; typical settings are span_mask_prob≈0.15 and mean_span_len≈3–5. If you’re training pure supervised dialogue, keep masking off. Training uses AdamW with standard scheduling, early stopping on validation loss, and best-checkpoint saving.

We report BLEU/chrF/TER (with proper detokenization) and perplexity from validation loss; for Urdu, chrF is often the most reliable. A simple human evaluation rates Fluency, Relevance, Adequacy (1–5) on ~50 prompts by 3 raters. This repo is meant as a clean baseline you can scale by adding data, enabling span corruption, or swapping in a stronger multilingual backbone.
