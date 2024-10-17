import numpy as np

class PositionalEncoding:
    """
    This class generates positional encodings for input sequences to inject
    information about the relative position of tokens.
    """

    def __init__(self, d_model, max_len=5000):
        """
        Initialize the positional encoding table.

        Parameters:
        - d_model (int): Dimensionality of the model's embeddings.
        - max_len (int): Maximum length of the input sequence.
        """
        self.d_model = d_model

        # Create a matrix of shape (max_len, d_model) where each row corresponds to the positional encoding.
        self.encoding = np.zeros((max_len, d_model))
        position = np.arange(0, max_len).reshape(-1, 1)
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        self.encoding[:, 0::2] = np.sin(position * div_term)
        self.encoding[:, 1::2] = np.cos(position * div_term)

    def get_positional_encoding(self, seq_len):
        """
        Returns the positional encoding for a sequence of length 'seq_len'.

        Parameters:
        - seq_len (int): Length of the sequence.

        Returns:
        - numpy array of shape (seq_len, d_model) containing positional encodings.
        """
        return self.encoding[:seq_len]


class ScaledDotProductAttention:
    """
    This class implements the scaled dot-product attention mechanism.
    """

    def __init__(self):
        pass

    def attention(self, Q, K, V, mask=None):
        """
        Compute the scaled dot-product attention.

        Parameters:
        - Q (numpy array): Query matrix of shape (batch_size, num_heads, seq_len, d_k).
        - K (numpy array): Key matrix of shape (batch_size, num_heads, seq_len, d_k).
        - V (numpy array): Value matrix of shape (batch_size, num_heads, seq_len, d_v).
        - mask (numpy array): Optional masking array for preventing attention to certain positions.

        Returns:
        - Attention output (numpy array) and attention weights.
        """
        d_k = Q.shape[-1]
        
        # Compute the dot product between queries and keys
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
        
        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)
        
        # Apply softmax to get attention weights
        attention_weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
        
        # Multiply the attention weights by the value matrix
        attention_output = np.matmul(attention_weights, V)
        
        return attention_output, attention_weights


class MultiHeadAttention:
    """
    This class implements multi-head attention, which runs scaled dot-product attention
    multiple times in parallel and projects the results.
    """

    def __init__(self, d_model, num_heads):
        """
        Initialize the multi-head attention layer.

        Parameters:
        - d_model (int): Dimensionality of the model's embeddings.
        - num_heads (int): Number of attention heads.
        """
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads

        # Linear projections for queries, keys, values, and the final output projection
        self.W_q = np.random.randn(d_model, d_model)
        self.W_k = np.random.randn(d_model, d_model)
        self.W_v = np.random.randn(d_model, d_model)
        self.W_o = np.random.randn(d_model, d_model)

        self.attention = ScaledDotProductAttention()

    def split_heads(self, X):
        """
        Splits the last dimension into (num_heads, depth).
        
        Parameters:
        - X (numpy array): Input of shape (batch_size, seq_len, d_model).

        Returns:
        - numpy array of shape (batch_size, num_heads, seq_len, depth).
        """
        batch_size, seq_len, _ = X.shape
        return X.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)

    def combine_heads(self, X):
        """
        Combines the multi-head output back into a single vector.
        
        Parameters:
        - X (numpy array): Input of shape (batch_size, num_heads, seq_len, depth).

        Returns:
        - numpy array of shape (batch_size, seq_len, d_model).
        """
        batch_size, num_heads, seq_len, depth = X.shape
        return X.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)

    def forward(self, Q, K, V, mask=None):
        """
        Computes the multi-head attention mechanism.

        Parameters:
        - Q, K, V (numpy arrays): Query, Key, and Value matrices of shape (batch_size, seq_len, d_model).
        - mask (numpy array): Optional mask for attention.

        Returns:
        - Multi-head attention output of shape (batch_size, seq_len, d_model).
        """
        # Linear projections
        Q = np.matmul(Q, self.W_q)
        K = np.matmul(K, self.W_k)
        V = np.matmul(V, self.W_v)

        # Split into multiple heads
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # Apply scaled dot-product attention
        attention_output, _ = self.attention.attention(Q, K, V, mask)

        # Combine the heads
        attention_output = self.combine_heads(attention_output)

        # Final linear projection
        output = np.matmul(attention_output, self.W_o)
        return output


class FeedForwardNetwork:
    """
    Implements a simple position-wise feed-forward neural network.
    """

    def __init__(self, d_model, d_ff):
        """
        Initialize the feed-forward layer.
        
        Parameters:
        - d_model (int): Dimensionality of the model.
        - d_ff (int): Dimensionality of the hidden layer.
        """
        self.W1 = np.random.randn(d_model, d_ff)
        self.b1 = np.zeros((1, d_ff))
        self.W2 = np.random.randn(d_ff, d_model)
        self.b2 = np.zeros((1, d_model))

    def forward(self, X):
        """
        Forward pass through the feed-forward network.
        
        Parameters:
        - X (numpy array): Input of shape (batch_size, seq_len, d_model).

        Returns:
        - Output of shape (batch_size, seq_len, d_model).
        """
        hidden = np.maximum(0, np.dot(X, self.W1) + self.b1)  # ReLU activation
        output = np.dot(hidden, self.W2) + self.b2
        return output


class LayerNorm:
    """
    Applies layer normalization over the last dimension of the input.
    """

    def __init__(self, d_model, epsilon=1e-6):
        """
        Initialize the layer normalization.

        Parameters:
        - d_model (int): Dimensionality of the model's embeddings.
        - epsilon (float): Small value to prevent division by zero.
        """
        self.gamma = np.ones((1, d_model))
        self.beta = np.zeros((1, d_model))
        self.epsilon = epsilon

    def forward(self, X):
        """
        Applies layer normalization to the input.
        
        Parameters:
        - X (numpy array): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
        - Normalized output tensor of shape (batch_size, seq_len, d_model).
        """
        mean = np.mean(X, axis=-1, keepdims=True)
        variance = np.var(X, axis=-1, keepdims=True)
        X_norm = (X - mean) / np.sqrt(variance + self.epsilon)
        return self.gamma * X_norm + self.beta


class EncoderLayer:
    """
    Represents a single layer of the Transformer encoder, consisting of multi-head attention,
    feed-forward network, and layer normalization.
    """

    def __init__(self, d_model, num_heads, d_ff):
        self.multi_head_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForwardNetwork(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

    def forward(self, X, mask=None):
        """
        Forward pass through the encoder layer.
        
        Parameters:
        - X (numpy array): Input tensor of shape (batch_size, seq_len, d_model).
        - mask (numpy array): Optional attention mask.

        Returns:
        - Output of shape (batch_size, seq_len, d_model).
        """
        # Normalize first, then apply attention
        X_norm1 = self.norm1.forward(X)
        attention_output = self.multi_head_attention.forward(X_norm1, X_norm1, X_norm1, mask)
        attention_output += X  # Residual connection

        # Normalize first, then apply feed-forward network
        X_norm2 = self.norm2.forward(attention_output)
        feed_forward_output = self.feed_forward.forward(X_norm2)
        return feed_forward_output + attention_output  # Residual connection


class TransformerEncoder:
    """
    Transformer encoder consisting of a stack of identical encoder layers.
    """

    def __init__(self, d_model, num_heads, d_ff, num_layers):
        """
        Initialize the encoder with multiple encoder layers.
        
        Parameters:
        - d_model (int): Dimensionality of the model's embeddings.
        - num_heads (int): Number of attention heads.
        - d_ff (int): Dimensionality of the feed-forward network.
        - num_layers (int): Number of encoder layers.
        """
        self.layers = [EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)]

    def forward(self, X, mask=None):
        """
        Forward pass through the encoder stack.
        
        Parameters:
        - X (numpy array): Input tensor of shape (batch_size, seq_len, d_model).
        - mask (numpy array): Optional attention mask.

        Returns:
        - Output of shape (batch_size, seq_len, d_model).
        """
        for layer in self.layers:
            X = layer.forward(X, mask)
        return X


# Example usage:
# transformer = TransformerEncoder(d_model=512, num_heads=8, d_ff=2048, num_layers=6)
# input_seq = np.random.randn(32, 100, 512)  # Batch size = 32, Sequence length = 100, Model dimensionality = 512
# output = transformer.forward(input_seq)