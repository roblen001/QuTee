# QuTee

## 🚀 A Hybrid Quantum-Classical GPT Framework

QuTee is a modular and extensible research framework designed to seamlessly integrate quantum components within classical Transformer architectures (GPT-style models). Its primary purpose is to explore whether quantum-enhanced modules—such as variational quantum circuits (VQCs) for feed-forward layers or quantum kernel-based attention mechanisms—can offer measurable improvements in generalization, efficiency, and representational power for natural language processing (NLP) tasks.


I added fine-tuning functionality and included some raw Shakespearean text to enable end-to-end model training. While this is not the primary focus of the framework at the moment, it was added for completeness. I also developed scripts that perform data distillation by generating fine-tuning datasets using OpenAI’s model APIs. As a result, the framework now supports full end-to-end GPT-style model development (on a small scale) for experimental purposes.

---

## 📖 Quick Start

**Installation:**

```bash
git clone https://github.com/roblen001/QuTee.git
cd QuTee
pip install -r requirements.txt
```

**Example of Classical feed-forward:**

```bash
from src.models.gpt import GPT
from src.model_components.feedfoward import ClassicalFeedForward

# these are currently set in configs/constants
CONTEXT_SIZE = 64
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
EPOCHS = 5000
MAX_NEW_TOKENS = 500
NUM_OF_BLOCKS = 4
NUM_HEADS = 4
EMBEDDING_DIM = 128 

model = GPT(
    tokenizer=tokenizer,
    context_size=CONTEXT_SIZE,
    num_heads=NUM_HEADS,
    n_layer=NUM_OF_BLOCKS,
    embedding_dim=EMBEDDING_DIM,
    feedforward_cls=ClassicalFeedForward,
    # I will decide what ff_kwargs should be added when I figure out what the quantum equivalents are for tunable parameters
    ff_kwargs={}
)
```

**Example of Quantum feed-forward:**

I currently have not implemented this part yet but this is the way I would like for this part to be called. You will currently get a not implemented error.

```bash
from src.models.gpt import GPT
from src.model_components.feedforward import QuantumFeedForward

# these are currently set in configs/constants
CONTEXT_SIZE = 64
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
EPOCHS = 5000
MAX_NEW_TOKENS = 500
NUM_OF_BLOCKS = 4
NUM_HEADS = 4
EMBEDDING_DIM = 128 

model = GPT(
    tokenizer=tokenizer,
    context_size=CONTEXT_SIZE,
    num_heads=NUM_HEADS,
    n_layer=NUM_OF_BLOCKS,
    embedding_dim=EMBEDDING_DIM,
    feedforward_cls=QuantumFeedForward,
    ff_kwargs={"n_qubits": 4, "n_layers": 6}
)
```

## 🎯 Objectives

- **Hybrid Experimentation:** Easily swap classical Transformer blocks with quantum-enhanced equivalents.
- **Performance Benchmarking:** Evaluate and compare performance metrics such as loss, perplexity, training speed, inference latency, and model complexity.
- **Quantum Component Identification:** Pinpoint quantum methods that offer tangible advantages in current (NISQ-era) quantum simulations and devices.

---

## 🧪 Synthetic Data Generation for Fine-tuning
You will probably need to run the script for multiple days to fill the final prompt answer file if it's not filled when you try running it.

1. **Set your API key** in `.env`:  
   ```bash
   OPENAI_API_KEY=your_api_token_here
   ```
2. **Install dependencies**:  
You can modify the parameters for the synthetic data generation in src/configs/synthetic_data_constants.py

3. **Install dependencies**:  
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the batch script:**:  
   ```bash
   python -m src.scripts.generate_data_for_finetuning
   ```
Once the CSV is fully populated, you’ll have the high-quality synthetic dataset you need to fine-tune your small Shakespeare GPT model. 

---

## 🧪 Run Fine-tuning
This script takes a long time to run on your cpus depending on your settings.

1. **Install dependencies**:  
   ```bash
   pip install -r requirements.txt
   ```

2. **Update parameters**:
   You can modify the parameters in /src/configs/finetuning_constants.py

3. **Run the finetuning script**:  
   ```bash
   python -m src.scripts.generate_data_for_finetuning
   ```
---

## 🧪 Run Inferencing

1. **Install dependencies**:  
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the finetuning script**:  
   ```bash
   python -m src.scripts.main_inferencing_scipt
   ```
---

## 🛠 Roadmap & Checklist

✅ **Phase 1: Classical Foundations**

- [x] Data loading & tokenization (Shakespeare corpus)
- [x] Implement baseline GPT model (classical Transformer)
- [x] Basic training loop & inference routines
- [x] Modularize Feedforward portion
- [x] Simple performance benchmarks module (loss, perplexity, runtime)

✅ **Phase 1.1: Classical Foundations**

At this stage we have a model that takes shakesperean text
and outputs cohesive words and sentences but it does not have any context to it. We want to fine tune the model so that when I ask it to write a story about a knight or something it will be able to spit out a cohesive story.
- [x] Add ability to save and load model checkpoints
- [x] Generate synthetic data for fine tuning using larger model
- [x] Finetune the small shakespear model
- [x] Implement method of data collection for comparison with quantum methods

Note the finetuning scripts all work but better hardware is needed (CPU is taking way too long to train)

Would ideally need to train a larger pretrained model too.

✅ **Phase 2: Quantum Integrations (Simulation)**

- [ ] Integrate PennyLane & PyTorch
- [ ] Quantum attention mechanisms (e.g., quantum kernels)
- [ ] Quantum-enhanced embeddings (amplitude or angle embeddings)
- [ ] Hybrid quantum-classical feed-forward layers (variational circuits)

✅ **Phase 3: Benchmark & Analysis**

- [ ] Side-by-side performance comparisons (classical vs. hybrid)
- [ ] Identify effective quantum modules based on empirical results
- [ ] Detailed documentation of experiments & findings

✅ **Phase 4: Future Expansions**

- [ ] Compatibility with real quantum hardware
- [ ] Integration with Hugging Face Transformers
- [ ] Extended benchmarks on larger NLP datasets

---

## 📚 Resources & References

- PennyLane Documentation: [pennylane.ai](https://pennylane.ai)
- Quantum NLP Examples: [Quantum Transformers Example](https://github.com/salcc/QuantumTransformers)
- Research Papers:
  - [Quantum Self-Attention Neural Networks](https://arxiv.org/abs/2205.05625)
  - [Quantum-Enhanced Attention Mechanism in NLP](https://arxiv.org/abs/2501.15630)

---

## 🤝 Contribution

Contributions to QuTee are welcome! Feel free to submit issues, feature requests, or pull requests. For major changes, please open an issue first to discuss your ideas.

---

## 📜 License

QuTee is open-source software licensed under the MIT license.
