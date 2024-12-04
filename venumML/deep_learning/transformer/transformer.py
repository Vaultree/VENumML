from venumML.venumpy import small_glwe as vp
import numpy as np
import math
from venumML.venum_tools import *
from venumML.approx_functions import * 

class Embeddings:
    """
    Handles embedding lookup for input tokens, returning embeddings with a specified dimension.

    Attributes
    ----------
    embedding_matrix : np.ndarray
        Custom embedding matrix, where each row corresponds to the embedding vector for a token.
    d_model : int
        Dimensionality of each embedding vector.
    """

    def __init__(self, custom_embeddings):
        """
        Initialises the Embeddings class with a custom embedding matrix.

        Parameters
        ----------
        custom_embeddings : np.ndarray
            Pre-trained embedding matrix, shape (vocab_size, d_model).
        """
        self._embedding_matrix = custom_embeddings
        self._d_model = custom_embeddings.shape[1]

    def forward(self, x, batch_size, max_seq_len):
        """
        Computes embeddings for a batch of input token sequences.

        Parameters
        ----------
        x : np.ndarray
            Array of token indices with shape [batch_size, seq_length].
        batch_size : int
            The number of sequences in the batch.
        max_seq_len : int
            Maximum sequence length.

        Returns
        -------
        np.ndarray
            Array of embeddings with shape (batch_size, seq_length, d_model).
        """

        # x is assumed to be of shape [batch_size, seq_length]
        batch_size, seq_length = batch_size,max_seq_len
        embeddings = np.zeros((batch_size, seq_length, self._d_model))

        for i in range(batch_size):
            for j in range(seq_length):
                embeddings[i, j] = self._embedding_matrix[x[i][j]]
        return embeddings 


def positional_encoding(max_seq_len, d_model):
    """
    Generates positional encoding for a sequence of given length and embedding dimension.

    Parameters
    ----------
    max_seq_len : int
        Maximum sequence length for which positional encoding is generated.
    d_model : int
        Dimensionality of each embedding vector.

    Returns
    -------
    np.ndarray
        Array of shape (max_seq_len, d_model) with positional encodings for each position.
    """

    PE = np.zeros((max_seq_len, d_model))
    position = np.arange(0, max_seq_len).reshape(-1, 1).astype(float)
    div_term = np.exp(np.arange(0, d_model, 2).astype(float) * (-math.log(10000.0) / d_model))

    PE[:, 0::2] = np.sin(position * div_term)
    PE[:, 1::2] = np.cos(position * div_term)

    return PE


def scaled_dot_product_attention(Q, K, V, ctx):
    """
    Computes scaled dot-product attention with encrypted attention weights.

    Parameters
    ----------
    Q : np.ndarray
        Query matrix of shape (batch_size, num_heads, seq_length, d_k).
    K : np.ndarray
        Key matrix of shape (batch_size, num_heads, seq_length, d_k).
    V : np.ndarray
        Value matrix of shape (batch_size, num_heads, seq_length, d_v).
    ctx : EncryptionContext
        The encryption context used to encrypt the attention scores.

    Returns
    -------
    output : np.ndarray
        Output after applying attention weights to the values, shape (batch_size, num_heads, seq_length, d_v).
    attention_weights : np.ndarray
        Encrypted attention weights applied to each value.
    """

    d_k = K.shape[-1]
    scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(d_k)
    attention_weights = softmax_approximation(ctx,scores,d_k)
    output = np.matmul(attention_weights, V)
    return output, attention_weights

class MultiHeadAttention:
    """
    Implements multi-head attention mechanism with separate attention heads and output projection.

    Attributes
    ----------
    num_heads : int
        Number of attention heads.
    W_Qs : list
        List of query weight matrices, one per head.
    b_Qs : list
        List of query bias vectors, one per head.
    W_Ks : list
        List of key weight matrices, one per head.
    b_Ks : list
        List of key bias vectors, one per head.
    W_Vs : list
        List of value weight matrices, one per head.
    b_Vs : list
        List of value bias vectors, one per head.
    W_O : np.ndarray
        Output weight matrix applied after concatenating head outputs.
    b_O : np.ndarray
        Output bias vector applied after concatenating head outputs.
    """
    
    def __init__(self, num_heads):
        """
        Initialises the MultiHeadAttention class with the specified number of heads.

        Parameters
        ----------
        num_heads : int
            Number of attention heads.
        """

        self._num_heads = num_heads
        self._W_Qs = [None] * num_heads
        self._b_Qs = [None] * num_heads
        self._W_Ks = [None] * num_heads
        self._b_Ks = [None] * num_heads
        self._W_Vs = [None] * num_heads
        self._b_Vs = [None] * num_heads
        self._W_O = None
        self._b_O = None

    def set_head_weights(self, head_index, head_weights):
        """
        Sets the weights and biases for a specific attention head.

        Parameters
        ----------
        head_index : int
            Index of the attention head.
        head_weights : dict
            Dictionary containing weights and biases for the specified head.
        """
        self._W_Qs[head_index] = head_weights[f'W_Q_{head_index}']
        self._b_Qs[head_index] = head_weights[f'b_Q_{head_index}']
        self._W_Ks[head_index] = head_weights[f'W_K_{head_index}']
        self._b_Ks[head_index] = head_weights[f'b_K_{head_index}']
        self._W_Vs[head_index] = head_weights[f'W_V_{head_index}']
        self._b_Vs[head_index] = head_weights[f'b_V_{head_index}']

    def set_output_weights(self, W_O, b_O):
        """
        Sets the weights and biases for the output layer.

        Parameters
        ----------
        W_O : np.ndarray
            Output weight matrix.
        b_O : np.ndarray
            Output bias vector.
        """
        self._W_O = W_O
        self._b_O = b_O


    
    def multi_head_attention(self, Q, K, V, ctx, d_model):
        """
        Computes multi-head attention for the given query, key, and value matrices.

        Parameters
        ----------
        Q : np.ndarray
            Query matrix of shape (batch_size, seq_length, d_model).
        K : np.ndarray
            Key matrix of shape (batch_size, seq_length, d_model).
        V : np.ndarray
            Value matrix of shape (batch_size, seq_length, d_model).
        ctx : EncryptionContext
            The encryption context used to encrypt attention scores.
        d_model : int
            Dimensionality of the model.

        Returns
        -------
        np.ndarray
            The output of multi-head attention, shape (batch_size, seq_length, d_model).
        """

        d_k = d_model // self._num_heads
        d_v = d_model // self._num_heads
        heads_outputs = []

        for i in range(self._num_heads):
            # Projecting Q, K, V for each head and adding bias
            Q_proj = np.dot(Q, self._W_Qs[i]) + self._b_Qs[i]
            K_proj = np.dot(K, self._W_Ks[i]) + self._b_Ks[i]
            V_proj = np.dot(V, self._W_Vs[i]) + self._b_Vs[i]

            # Apply scaled dot-product attention for each head
            head_output, _ = scaled_dot_product_attention(Q_proj, K_proj, V_proj, ctx)
            heads_outputs.append(head_output)

        # Concatenating outputs from each head
        concatenated = np.concatenate(heads_outputs, axis=-1)

        # Applying the output linear layer with bias
        output = np.dot(concatenated, self._W_O) + self._b_O
        return output

class PositionwiseFeedForwardNetwork:
    """
    Implements a position-wise feed-forward network with two linear transformations 
    and an activation function.

    Attributes
    ----------
    d_model : int
        Dimensionality of the input.
    d_ff : int
        Dimensionality of the hidden layer.
    W_1 : np.ndarray
        Weight matrix for the first linear layer.
    b_1 : np.ndarray
        Bias vector for the first linear layer.
    W_2 : np.ndarray
        Weight matrix for the second linear layer.
    b_2 : np.ndarray
        Bias vector for the second linear layer.
    """
    
    def __init__(self, d_model, d_ff):
        """
        Initialises the PositionwiseFeedForwardNetwork.

        Parameters
        ----------
        d_model : int
            Dimensionality of the input.
        d_ff : int
            Dimensionality of the hidden layer.
        """
        self._d_model = d_model
        self._d_ff = d_ff

        # Initialize weights and biases for the two linear layers using Xavier initialization
        self._W_1 = np.random.randn(d_model, d_ff) * np.sqrt(2.0 / d_model)
        self._b_1 = np.zeros(d_ff)
        self._W_2 = np.random.randn(d_ff, d_model) * np.sqrt(2.0 / d_ff)
        self._b_2 = np.zeros(d_model)

    def set_weights(self, W_1, b_1, W_2, b_2):
        """
        Sets the weights and biases for the feed-forward network.

        Parameters
        ----------
        W_1 : np.ndarray
            Weight matrix for the first linear layer.
        b_1 : np.ndarray
            Bias vector for the first linear layer.
        W_2 : np.ndarray
            Weight matrix for the second linear layer.
        b_2 : np.ndarray
            Bias vector for the second linear layer.
        """

        self._W_1 = W_1
        self._b_1 = b_1
        self._W_2 = W_2
        self._b_2 = b_2

    def forward(self, x, ctx):
        """
        Forward pass through the feed-forward network.

        Parameters
        ----------
        x : np.ndarray
            Input array with shape [batch_size, seq_length, d_model].
        ctx : EncryptionContext
            Encryption context used for any approximations in activation.

        Returns
        -------
        np.ndarray
            Output of the feed-forward network.
        """

        # First linear layer with ReLU activation
        x = np.dot(x, self._W_1) + self._b_1
        x = sigmoid_approximation(ctx, x)  # ReLU activation

      

        # Second linear layer
        x = np.dot(x, self._W_2) + self._b_2

        return x
        
def output_linear_layer(x, W, b):
    """
    Implements the output linear layer for a Transformer model.

    Parameters
    ----------
    x : np.ndarray
        Input array representing the output of the last Transformer layer.
    W : np.ndarray
        Weight matrix for the linear transformation.
    b : np.ndarray
        Bias vector for the linear transformation.

    Returns
    -------
    np.ndarray
        The output vector for each token in the vocabulary.
    """
    # Linear transformation
    linear_output = np.matmul(x, W) + b


    return linear_output 


class TransformerModule:
    """
    A simplified Transformer module implementing embedding, multi-head attention, 
    positional encoding, and a feed-forward network.

    Attributes
    ----------
    max_seq_len : int
        Maximum sequence length.
    num_heads : int
        Number of attention heads.
    d_model : int
        Dimensionality of embeddings.
    MHA : MultiHeadAttention
        Multi-head attention layer.
    positional_encoding : np.ndarray
        Positional encoding matrix.
    p_ffn : PositionwiseFeedForwardNetwork
        Feed-forward network.
    output_w : np.ndarray
        Weight matrix for the output layer.
    output_b : np.ndarray
        Bias vector for the output layer.
    """

    def __init__(self, encrypted_state_dict, max_seq_len, d_model, num_heads, d_ff, vocab_size):
        """
        Initialises the TransformerModule with necessary layers and weights.

        Parameters
        ----------
        encrypted_state_dict : dict
            Dictionary containing pre-trained weights in encrypted form.
        max_seq_len : int
            Maximum length of input sequences.
        d_model : int
            Dimensionality of embeddings.
        num_heads : int
            Number of attention heads.
        d_ff : int
            Dimensionality of the feed-forward network's hidden layer.
        vocab_size : int
            Size of the vocabulary.
        """

        #mha weights
        mha_weights = {}
        for i in range(num_heads):
            mha_weights[f'W_Q_{i}'] = encrypted_state_dict[f'mha.W_Qs.{i}.weight']
            mha_weights[f'b_Q_{i}'] = encrypted_state_dict[f'mha.W_Qs.{i}.bias']
            mha_weights[f'W_K_{i}'] = encrypted_state_dict[f'mha.W_Ks.{i}.weight']
            mha_weights[f'b_K_{i}'] = encrypted_state_dict[f'mha.W_Ks.{i}.bias']
            mha_weights[f'W_V_{i}'] = encrypted_state_dict[f'mha.W_Vs.{i}.weight']
            mha_weights[f'b_V_{i}'] = encrypted_state_dict[f'mha.W_Vs.{i}.bias']
        mha_output_weights = encrypted_state_dict['mha.W_O.weight']
        mha_output_bias = encrypted_state_dict['mha.W_O.bias']

        #feed forward weights
        pffnw1 = encrypted_state_dict['ffn.linear1.weight'] 
        pffnw2 = encrypted_state_dict['ffn.linear2.weight'] 
        pffnb1 = encrypted_state_dict['ffn.linear1.bias'] 
        pffnb2 = encrypted_state_dict['ffn.linear2.bias'] 
        
        # Initialize weights for multi-head attention
        self._max_seq_len = max_seq_len
        self._num_heads = num_heads
        self._d_model = d_model
        self._MHA = MultiHeadAttention(num_heads=num_heads)
        for i in range(num_heads):
            self._MHA.set_head_weights(i,mha_weights)
        self._MHA.set_output_weights(mha_output_weights,mha_output_bias)
        
        self._positional_encoding = positional_encoding(max_seq_len, d_model)  # Assuming positional_encoding is defined

        # Initialize weights for feed forward network
        
        self._p_ffn = PositionwiseFeedForwardNetwork(d_model,d_ff)
        self._p_ffn.set_weights(pffnw1,pffnb1,pffnw2,pffnb2)


        self._output_w = encrypted_state_dict['output_linear.weight'] 
        self._output_b = encrypted_state_dict['output_linear.bias'] 

   
    def forward(self, embeddings, ctx, batch_size):
        """
        Processes the input embeddings through the Transformer.

        Parameters
        ----------
        embeddings : np.ndarray
            Input embeddings with shape [batch_size, seq_length, d_model].
        ctx : EncryptionContext
            Encryption context used for any approximations in activation.
        batch_size : int
            Number of input sequences in the batch.

        Returns
        -------
        np.ndarray
            Output vector after embedding, positional encoding, attention, and the feed-forward network.
        """
 
        #  # Add positional encoding (up to max_len)
        pos_encoding = self._positional_encoding

       #  # Repeat the positional encoding for each sequence in the batch
        pos_encoding_batch = np.repeat(pos_encoding[np.newaxis, :, :], batch_size, axis=0)
        pos_encoding_batch= encrypt_array(pos_encoding_batch,ctx)

        encoded_vector = embeddings + pos_encoding_batch
       # #  # Apply multi-headed attention 
        attention_output = self._MHA.multi_head_attention(encoded_vector, encoded_vector, encoded_vector, ctx, self._d_model)

       # # #residual connection
        attention_output = attention_output + encoded_vector
       #   #positionwise feed forward network
        ff_output = self._p_ffn.forward(attention_output, ctx)
       #  # #residual connection


        ff_output = ff_output + attention_output

       #    # # Pass through the final linear layer
       # #  # Apply the output linear layer
        cls_output = ff_output[:, 0, :]
        output = output_linear_layer(cls_output,self._output_w,self._output_b) 
        return output