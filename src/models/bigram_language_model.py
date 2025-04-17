
import torch
import torch.nn as nn
from torch.nn import functional
from torch import Tensor

from src.utils.data_processing_tools import get_batch 

# simple bigrammodel
# just predicts the next token based on the current token
# linear, no larger context
class BigramLanguageModel(nn.Module):
    """
    A simple neural language model that learns to predict the next token based solely 
    on the current token. It uses an embedding lookup table where each input token 
    maps directly to a vector of logits representing the probabilities of all possible 
    next tokens. This model captures basic token-to-token relationships (bigrams) 
    without using attention or larger context.
    """

    def __init__(self, tokenizer: dict[str, list[str]]):
        super().__init__() # calls constructor of parent (nn.Module) else you get an error, enables the pytorch stuff we need ( .parameters(), .cuda(), .train(), .eval()) and so on
        # A simple lookup table based on the vocab size x embedding dimension (you can choose this dimension)
        # for each encoding you now have a vector (randomly initalized) that will be improve each iteration of the model
        # the goal is to get the vector represensatations of the tokens (dense vectors) to be close together in the vector space
        # if they are related.
        token_dict_dim = len(tokenizer['encoder'])
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.token_embedding_table = nn.Embedding(token_dict_dim, token_dict_dim).to(self.device) # I believe most models use a 128 dim embedding but that could be outdated by now

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
        # the shape of our input_ids will change once it is converted to the embeddings
        logits = self.token_embedding_table(input_ids)  # (batch_size, context_size, vocab_size)
        # no targets when inferencing, but not training
        if targets is None:
            loss = None
            return logits
        else:
            logits, targets = BigramLanguageModel._reshape_tensors(logits, targets)
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
        while i <= max_new_tokens:
            # Predict logits for the current sequence
            logits = self(output_ids)  # (B, T, vocab_size)

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


    def training_loop(self, data, optimizer, epochs: int = 10000, batch_size: int = 32, context_size: int = 8):
        train_dataset = {'train': data['train']['stream']}
        for step in range(epochs):
            if step % 500 == 0:
                losses = self.pooled_loss(data, 500, batch_size, context_size)
                print(f"=========TRAINING LOSS AT STEP {step}===================")
                print(f"validation: {losses['test']}")
                print(f"training: {losses['train']}")
            
            batch = get_batch(train_dataset, batch_size, context_size)
            x = batch['train']['x'].to(self.device)
            y = batch['train']['y'].to(self.device)

            logits, loss = self(x, y)
            # resets gradient of all model before backward pass or else you will have
            # accumulated gradients across batches
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        print("=========FINAL TRAINING LOSS=============")
        print(loss.item())