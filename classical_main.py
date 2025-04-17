"""
Building the simplest Transformer model possible.
"""
import torch
import torch.nn as nn
from torch.nn import functional
from torch import Tensor

from src.data_processing.data_processing import prepare_data
from src.utils.data_processing_tools import decode

batched_data, tokenizer = prepare_data(
    path="data/raw/shakespeare.txt",
    batch_size=32,
    context_size=16
)


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
        self.token_embedding_table = nn.Embedding(token_dict_dim, 64) # I believe most models use a 128 dim embedding but that could be outdated by now

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

model = BigramLanguageModel(tokenizer=tokenizer)
# logits, loss = model(batched_data['train']['x'], batched_data['train']['y'])
print(model.generate(torch.zeros((1, 1), dtype=torch.int64), max_new_tokens=10)) 
