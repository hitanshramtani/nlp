🧠 MBR-Based Neural Machine Translation

This project implements a neural machine translation (NMT) model with Minimum Bayes Risk (MBR) decoding, allowing for better translation selection by optimizing for similarity with multiple generated candidates. Evaluation metrics like BLEU and ROUGE are also used to assess model performance.

📂 Project Structure

model.py                # Transformer model architecture
mbr_decoding.py        # MBR decoding logic
evaluation.py          # BLEU & ROUGE evaluation functions
utils.py               # Tokenization and similarity functions
train.py               # Training loop and model fitting
inference.py           # Translation generation and MBR wrapper
README.txt             # This file

📦 Requirements

- Python ≥ 3.7
- TensorFlow ≥ 2.8
- NumPy
- NLTK
- scikit-learn
- tqdm

Install with:
pip install -r requirements.txt

🧪 How to Run

1. Train the Model:
   python train.py

2. Generate Translation with MBR:
   from inference import mbr_decode
   english_sentence = "I love languages"
   translation, candidates = mbr_decode(trained_model, english_sentence, n_samples=10)
   print("Candidates:")
   print("\n".join(candidates))
   print("\nFinal Selected Translation:", translation)

📊 Evaluation

BLEU:
evaluate_on_bleu(model, val_data, num_samples=100)

ROUGE-1:
evaluate_on_rouge(model, val_data, num_samples=100)

✅ Sample Results

Validation Accuracy: ~78.3%
Average ROUGE-1 Score: ~0.253
BLEU Score: (computed at evaluation time)

📚 Acknowledgements

- Based on TensorFlow's NMT transformer architecture
- Inspired by Minimum Bayes Risk Decoding in statistical and neural MT

📌 Future Work

- Add BLEU-4 and METEOR evaluations
- Support attention visualization
- Explore other decoding strategies (e.g., nucleus sampling)
