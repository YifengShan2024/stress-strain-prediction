import torch
import torch.nn as nn


class HybridAttention(nn.Module):
    """
    Hybrid Attention Module combining self-attention and channel attention.
    """

    def __init__(self, input_dim, attention_dim):
        """
        Initialize the Hybrid Attention Module.
        Args:
            input_dim (int): Input dimension of features.
            attention_dim (int): Dimension for attention transformations.
        """
        super(HybridAttention, self).__init__()

        # Self-attention components
        self.query = nn.Linear(input_dim, attention_dim)
        self.key = nn.Linear(input_dim, attention_dim)
        self.value = nn.Linear(input_dim, attention_dim)
        self.softmax = nn.Softmax(dim=-1)

        # Channel attention components
        self.channel_fc1 = nn.Linear(input_dim, input_dim // 2)
        self.channel_fc2 = nn.Linear(input_dim // 2, input_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass of the Hybrid Attention Module.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim).
        Returns:
            torch.Tensor: Attention-weighted feature representation.
            torch.Tensor: Attention weights from the self-attention module.
        """
        # Self-Attention
        q = self.query(x)  # Query
        k = self.key(x)  # Key
        v = self.value(x)  # Value
        attention_weights = self.softmax(torch.matmul(q, k.transpose(-2, -1)) / (q.size(-1) ** 0.5))
        self_attention_output = torch.matmul(attention_weights, v)

        # Channel Attention
        channel_avg = torch.mean(x, dim=1)  # Average pooling along sequence length
        channel_weights = self.sigmoid(self.channel_fc2(self.channel_fc1(channel_avg)))
        channel_attention_output = x * channel_weights.unsqueeze(1)

        # Combine self-attention and channel attention
        hybrid_output = self_attention_output + channel_attention_output

        return hybrid_output, attention_weights


if __name__ == "__main__":
    # Example input
    batch_size = 32
    seq_len = 10
    input_dim = 64
    attention_dim = 32

    example_input = torch.rand((batch_size, seq_len, input_dim))

    # Initialize attention module
    attention = HybridAttention(input_dim, attention_dim)

    # Forward pass
    output, attn_weights = attention(example_input)

    print(f"Hybrid Attention Output shape: {output.shape}")
    print(f"Attention Weights shape: {attn_weights.shape}")
