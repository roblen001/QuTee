"""
Contains the main model architecture
"""

import torch
import torch.nn as nn
from torch.nn import functional
from torch import Tensor
import time
import os
import datetime

from src.utils.data_processing_tools import get_batch 
from src.model_components.attention import MultiHeadAttention
from src.model_components.feedforward import BaseFeedForward, ClassicalFeedForward, QuantumFeedForward
from src.monitoring.monitoring import TrainingMonitor
from src.utils.data_processing_tools import decode

import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    """
    A *single* decoder-style Transformer block (as used in GPT-like models).

    The block performs two consecutive steps on the input sequence:

    1. **Multi-Head Self-Attention** – tokens “talk” to every other token so the
       model can gather context.
    2. **Feed-Forward Network** – each token is processed independently through
       a small MLP to create richer representations.

    Each step is wrapped in:
      • **LayerNorm** (applied *before* the step, “Pre-Norm”) to stabilise
        training.  
      • **Residual / skip connection** so gradients flow easily through many
        stacked blocks.

    Parameters
    ----------
    embedding_dim : int
        Width of the model (size of each token’s embedding vector).
    num_heads : int
        Number of parallel attention heads. ``embedding_dim`` must be divisible by
        this value.
    feedforward_cls : type[BaseFeedForward]
        Class implementing the feed-forward interface to use (classical or quantum).
    ff_kwargs : dict
        Additional kwargs passed to the feedforward constructor.

    Notes
    -----
    *   Decoder blocks mask future positions inside the `MultiHeadAttention`
        module, ensuring auto-regressive (left-to-right) generation.
    *   Encoder blocks are identical except the attention sub-layer is
        *un-masked*.
    """

    def __init__(self,
                 embedding_dim: int,
                 num_heads: int,
                 feedforward_cls: type[BaseFeedForward] = ClassicalFeedForward,
                 **ff_kwargs) -> None:
        super().__init__()

        # attention layer is going to allow the model to have an understanding of tokens based on its surrounding context.
        # for example a human might not know the definition of 'mole' since it could be the animal mole, chemistry measurement unit mole, or mole in the medical sense
        # but if we see the word in a sentence like "The doctor examined my mole." We know defition of mole is being used. 
        # The attention mechanism gives the model the ability to understand language like that too. 
        self.self_attention = MultiHeadAttention(num_heads, embedding_dim)

        # the MLP portion is added to the architecture so the model can memorize facts about the text
        # thing like Lebron James is a basketball player...
        self.feed_forward  = feedforward_cls(embedding_dim, **ff_kwargs)

        # “Pre-Norm” formulation this is a deviation from the original 'Attention Is All You Need Paper'
        self.norm_attn = nn.LayerNorm(embedding_dim)
        self.norm_ffn  = nn.LayerNorm(embedding_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Apply the Transformer block.

        Parameters
        ----------
        input_ids : torch.Tensor
            Input of shape *(batch_size, seq_len, embed_dim)*.

        Returns
        -------
        torch.Tensor
            Output tensor of the same shape as ``x`` after attention and
            feed-forward transformations.
        """
        # 1) Communication — tokens exchange information
        input_ids = input_ids + self.self_attention(self.norm_attn(input_ids))
        # 2) Computation — per-token non-linear processing
        input_ids = input_ids + self.feed_forward(self.norm_ffn(input_ids))
        return input_ids

class GPT(nn.Module):
    """
    A simple neural language model that learns to predict the next token based solely 
    on the current token. It uses an embedding lookup table where each input token 
    maps directly to a vector of logits representing the probabilities of all possible 
    next tokens. This model captures basic token-to-token relationships (bigrams) 
    without using attention or larger context.
    """

    def __init__(self,
                 tokenizer: dict[str, list[str]],
                 context_size: int = 128,
                 n_layer: int = 4,
                 num_heads: int = 4,
                 embedding_dim: int = 328,
                 feedforward_cls: type[BaseFeedForward] = ClassicalFeedForward,
                 ff_kwargs: dict = {}):
        super().__init__() # calls constructor of parent (nn.Module) else you get an error, enables the pytorch stuff we need ( .parameters(), .cuda(), .train(), .eval()) and so on
        # A simple lookup table based on the vocab size x embedding dimension (you can choose this dimension)
        # for each encoding you now have a vector (randomly initalized) that will be improve each iteration of the model
        # the goal is to get the vector represensatations of the tokens (dense vectors) to be close together in the vector space
        # if they are related.
        token_dict_dim = len(tokenizer['encoder'])
        # TODO: currently not actually using gpu
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.token_embedding_table = nn.Embedding(token_dict_dim, embedding_dim) # I believe most models use a 128 dim embedding but that could be outdated by now
        self.context_size = context_size
        # without position embedding table the model has no way of knowing which token came first
        # this knowledge will help it make better predictions
        self.position_embedding_table = nn.Embedding(context_size, embedding_dim)
        self.tokenizer = tokenizer
        # linear layer turns a contextual vector into a token probability distribution since the position
        # embdding adds dimensionality to the data
        self.lm_head = nn.Linear(embedding_dim, token_dict_dim)
        
        # this function just makes it easier to add TransformerBlock layers for scaling
        self.blocks = nn.Sequential(*[TransformerBlock(
            embedding_dim,
            num_heads=num_heads,
            feedforward_cls=feedforward_cls,
            **ff_kwargs
        ) for _ in range(n_layer)])
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def _reshape_tensors(logits, targets):
        """
        Reshapes logits and targets for compatibility with PyTorch's cross-entropy loss.

        CrossEntropyLoss expects:
            - logits of shape (N, C): where N is the total number of predictions, and C is the number of classes
            - targets of shape (N,): where each entry is the correct class index

        This function flattens the batch and context dimensions into a single axis.

        Args:
            logits (Tensor): Tensor of shape (batch_size, context_size, vocab_size)
            targets (Tensor): Tensor of shape (batch_size, context_size)

        Returns:
            Tuple[Tensor, Tensor]: 
                - logits of shape (batch_size * context_size, vocab_size)
                - targets of shape (batch_size * context_size)
        """
        batch_size, context_size, vocab_size = logits.shape
        logits = logits.view(batch_size * context_size, vocab_size)
        targets = targets.view(batch_size * context_size)
        return logits, targets
    
    def forward(self, input_ids, targets=None):
        """
        Performs a forward pass of the BigramLanguageModel.

        During training, it computes the cross-entropy loss between predicted logits and target token indices.
        During inference (e.g., text generation), it returns only the logits.

        Args:
            input_ids (Tensor): Tensor of token indices with shape (batch_size, context_size)
            targets (Tensor, optional): Tensor of target token indices with shape (batch_size, context_size).
                                        If provided, loss is computed; if None, model runs in inference mode.

        Returns:
            logits (Tensor): Raw unnormalized scores for next-token prediction, shape (batch_size, context_size, vocab_size)
            loss (Tensor or None): Cross-entropy loss if targets are provided; otherwise None
        """
        batch_size, input_length = input_ids.shape
        
        # the shape of our input_ids will change once it is converted to the embeddings
        embeddings = self.token_embedding_table(input_ids)  # (batch_size, context_size, vocab_size)
        # truncate if the input_length is longer then the context size of the model
        positions = torch.arange(input_length)
        position_embeddings = self.position_embedding_table(positions)
        processed_embeddings = embeddings + position_embeddings 
        processed_embeddings = self.blocks(processed_embeddings)
        # linear layer turns a contextual vector into a token probability distribution since the position
        logits = self.lm_head(processed_embeddings)

        # no targets when inferencing, but not training
        if targets is None:
            loss = None
            return logits
        else:
            logits, targets = GPT._reshape_tensors(logits, targets)
            loss = functional.cross_entropy(logits, targets)
            return logits, loss

    def generate(self, input_ids: Tensor, max_new_tokens: int) -> Tensor:
        """
        Autoregressively generates tokens from input using the trained model.

        At each step, the model predicts the next token, samples from the probability
        distribution, and appends it to the input sequence. Repeats this process
        `max_new_tokens` times.

        Args:
            input_ids (Tensor): Initial token indices of shape (batch_size, context_size).
            max_new_tokens (int): Number of tokens to generate beyond the initial input.

        Returns:
            Tensor: Final sequence including generated tokens (batch_size, context_size + max_new_tokens).
        """
        i = 0
        output_ids = input_ids
        while i < max_new_tokens:
            # because positional embeddings relies on context_size we cannot
            # have input_ids with length larger than context_size
            output_ids_trimmed = output_ids[:, -self.context_size:]
            # Predict logits for the current sequence
            logits = self(output_ids_trimmed)  # (B, T, vocab_size)

            # Use only the last timestep to predict the next token
            logits = logits[:, -1, :]  # (B, vocab_size)

            # Convert logits to probabilities
            probs = functional.softmax(logits, dim=-1)  # (B, vocab_size)

            # Sample from the distribution (adds diversity)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # Append sampled token to the sequence
            output_ids = torch.cat((output_ids, next_token), dim=1)  # (B, T+1)

            i += 1

        return output_ids

    @torch.no_grad() # telling pytorch we wont be using backward method. This will save memory.
    def pooled_loss(self, data: dict[str, dict[str, torch.Tensor]], eval_iters, batch_size, context_size) -> dict[str, float]:
        """
        Computes the average loss over multiple random mini-batches for both training and test sets.

        Args:
            data (dict): A dictionary with 'train' and 'test' keys. Each maps to a sub-dict containing:
                - 'stream': 1D Tensor of token IDs
            eval_iters (int): Number of randomly sampled mini-batches to average the loss over.

        Returns:
            dict[str, float]: Dictionary with average losses for 'train' and 'test'.
        """
        self.eval() # stopping training behaviour like dropout, ...
        average_losses = {}

        for split in ['train', 'test']:
            losses = torch.zeros(eval_iters)
            for i in range(eval_iters):
                batch = get_batch({split: data[split]['stream']}, batch_size=batch_size, context_size=context_size)
                x = batch[split]['x'].to(self.device)
                y = batch[split]['y'].to(self.device)
                _, loss = self(x, y)
                losses[i] = loss.item()
            average_losses[split] = losses.mean().item()

        self.train() # reactivate the training behaviour
        return average_losses

    def training_loop(self, data: dict, optimizer, results_dir: str, epochs: int = 10000, batch_size: int = 32,) -> None:
        """
        Executes the training loop, periodically evaluating, printing, and logging losses,
        while recording metrics using TrainingMonitor.
        """
        train_dataset = {'train': data['train']['stream']}
        monitor = TrainingMonitor()

        # Update monitor to save plots to the correct subfolder
        monitor.results_dir = results_dir

        # Set up training log
        log_file_path = os.path.join(results_dir, "training_log.txt")
        with open(log_file_path, "w", encoding="utf-8") as log_file:
            for step in range(epochs):
                epoch_start_time = time.time()

                if step % 100 == 0:
                    losses = self.pooled_loss(data, 100, batch_size, self.context_size)
                    epoch_time = time.time() - epoch_start_time
                    monitor.record(train_loss=losses['train'], val_loss=losses['test'], epoch_time=epoch_time)

                    log_message = (
                        f"\n========= TRAINING LOSS AT STEP {step} =========\n"
                        f"Validation Loss: {losses['test']:.6f}\n"
                        f"Training Loss: {losses['train']:.6f}\n"
                        f"Epoch Time: {epoch_time:.2f} seconds\n"
                    )
                    print(log_message.strip())
                    log_file.write(log_message)
                    log_file.flush()

                    # Save checkpoint after every 500 steps
                    checkpoint_path = os.path.join(results_dir, f"checkpoint_step_{step}.pt")
                    torch.save({
                        'step': step,
                        'model_state_dict': self.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': losses['train'],
                        'val_loss': losses['test']
                    }, checkpoint_path)
                    print(f"Saved checkpoint to {checkpoint_path}")

                    # Save text generation sample after every 200 steps
                    sample_output = self.generate(
                        torch.zeros((1, 1), dtype=torch.int64).to(self.device),
                        max_new_tokens=200
                    )
                    sample_text = decode(sample_output[0].tolist(), self.tokenizer)
                    sample_path = os.path.join(results_dir, f"sample_step_{step}.txt")
                    with open(sample_path, "w", encoding="utf-8") as sample_file:
                        sample_file.write(f"=== Generated text at step {step} ===\n\n")
                        sample_file.write(sample_text)
                    print(f"Saved sample text to {sample_path}")

                batch = get_batch(train_dataset, batch_size, self.context_size)
                x = batch['train']['x'].to(self.device)
                y = batch['train']['y'].to(self.device)

                logits, loss = self(x, y)

                # resets gradient of all model before backward pass or else you will have
                # accumulated gradients across batches
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            final_loss_message = (
                "\n========= FINAL TRAINING LOSS =========\n"
                f"Final Training Loss: {loss.item():.6f}\n"
            )
            print(final_loss_message.strip())
            log_file.write(final_loss_message)
            log_file.flush()

            # Save final model checkpoint
            final_model_path = os.path.join(results_dir, "model_final.pt")
            torch.save({
                'model_state_dict': self.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'final_loss': loss.item()
            }, final_model_path)
            print(f"Saved final model to {final_model_path}")

            # Export full model + tokenizer so you can load & generate without rebuilding
            export_path = os.path.join(results_dir, "model_full.pt")
            # attach tokenizer into the model object
            self.tokenizer = self.tokenizer
            torch.save(self, export_path)
            print(f"Exported full model to {export_path}")

        # Save plots after training
        monitor.plot_losses()
        monitor.plot_epoch_times()

        print(f"\nTraining log, checkpoints, text samples, and plots saved to {results_dir}")

