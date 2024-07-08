import venumpy
import numpy as np
import math
from venumMLlib.venum_tools import *
from venumMLlib.approx_functions import * 

class Embeddings:
    def __init__(self, custom_embeddings):
        self.embedding_matrix = custom_embeddings
        self.d_model = custom_embeddings.shape[1]

    def forward(self, x,batch_size,max_seq_len):
        # x is assumed to be of shape [batch_size, seq_length]
        batch_size, seq_length = batch_size,max_seq_len
        embeddings = np.zeros((batch_size, seq_length, self.d_model))

        for i in range(batch_size):
            for j in range(seq_length):
                embeddings[i, j] = self.embedding_matrix[x[i][j]]

        return embeddings 


def positional_encoding(max_seq_len, d_model):
    PE = np.zeros((max_seq_len, d_model))
    position = np.arange(0, max_seq_len).reshape(-1, 1).astype(float)
    div_term = np.exp(np.arange(0, d_model, 2).astype(float) * (-math.log(10000.0) / d_model))

    PE[:, 0::2] = np.sin(position * div_term)
    PE[:, 1::2] = np.cos(position * div_term)

    return PE


def scaled_dot_product_attention(Q, K, V, ctx):
    d_k = K.shape[-1]
    scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(d_k)
    attention_weights = softmax_approximation(ctx,scores,d_k)
    output = np.matmul(attention_weights, V)
    return output, attention_weights

class MultiHeadAttention:
    def __init__(self, num_heads):
        self.num_heads = num_heads
        self.W_Qs = [None] * num_heads
        self.b_Qs = [None] * num_heads
        self.W_Ks = [None] * num_heads
        self.b_Ks = [None] * num_heads
        self.W_Vs = [None] * num_heads
        self.b_Vs = [None] * num_heads
        self.W_O = None
        self.b_O = None

    def set_head_weights(self, head_index, head_weights):
        self.W_Qs[head_index] = head_weights[f'W_Q_{head_index}']
        self.b_Qs[head_index] = head_weights[f'b_Q_{head_index}']
        self.W_Ks[head_index] = head_weights[f'W_K_{head_index}']
        self.b_Ks[head_index] = head_weights[f'b_K_{head_index}']
        self.W_Vs[head_index] = head_weights[f'W_V_{head_index}']
        self.b_Vs[head_index] = head_weights[f'b_V_{head_index}']
    def set_output_weights(self, W_O, b_O):
        self.W_O = W_O
        self.b_O = b_O


    
    def multi_head_attention(self, Q, K, V, ctx, d_model):
        d_k = d_model // self.num_heads
        d_v = d_model // self.num_heads
        heads_outputs = []

        for i in range(self.num_heads):
            # Projecting Q, K, V for each head and adding bias
            Q_proj = np.dot(Q, self.W_Qs[i]) + self.b_Qs[i]
            K_proj = np.dot(K, self.W_Ks[i]) + self.b_Ks[i]
            V_proj = np.dot(V, self.W_Vs[i]) + self.b_Vs[i]

            # Apply scaled dot-product attention for each head
            head_output, _ = scaled_dot_product_attention(Q_proj, K_proj, V_proj, ctx)
            heads_outputs.append(head_output)

        # Concatenating outputs from each head
        concatenated = np.concatenate(heads_outputs, axis=-1)

        # Applying the output linear layer with bias
        output = np.dot(concatenated, self.W_O) + self.b_O
        return output

class PositionwiseFeedForwardNetwork:
    def __init__(self, d_model, d_ff):
        """
        Initializes the PositionwiseFeedForwardNetwork.

        Args:
        - d_model (int): The dimensionality of the input.
        - d_ff (int): The dimensionality of the hidden layer.
        - dropout_rate (float): The dropout rate.
        """
        self.d_model = d_model
        self.d_ff = d_ff

        # Initialize weights and biases for the two linear layers using Xavier initialization
        self.W_1 = np.random.randn(d_model, d_ff) * np.sqrt(2.0 / d_model)
        self.b_1 = np.zeros(d_ff)
        self.W_2 = np.random.randn(d_ff, d_model) * np.sqrt(2.0 / d_ff)
        self.b_2 = np.zeros(d_model)

    def set_weights(self, W_1, b_1, W_2, b_2):
        """
        Set the weights and biases for the feed-forward network.

        Args:
        - W_1, b_1: Weights and biases for the first linear layer.
        - W_2, b_2: Weights and biases for the second linear layer.
        """
        self.W_1 = W_1
        self.b_1 = b_1
        self.W_2 = W_2
        self.b_2 = b_2

    def forward(self, x, ctx):
        """
        Forward pass through the feed-forward network.

        Args:
        - x (np.array): Input array with shape [batch_size, seq_length, d_model]

        Returns:
        - np.array: Output of the feed-forward network.
        """
        # First linear layer with ReLU activation
        x = np.dot(x, self.W_1) + self.b_1
        x = sigmoid_approximation(ctx, x)  # ReLU activation

      

        # Second linear layer
        x = np.dot(x, self.W_2) + self.b_2

        return x
        
def output_linear_layer(x, W, b):
    """
    Implements the output linear layer and softmax for a Transformer model.

    Args:
    - x (np.array): The input to the layer (output of the last Transformer layer).
    - W (np.array): The weight matrix for the linear transformation.
    - b (np.array): The bias vector for the linear transformation.

    Returns:
    - np.array: The output probabilities for each token in the vocabulary.
    """
    # Linear transformation
    linear_output = np.matmul(x, W) + b


    return linear_output 


class TransformerModule:
    def __init__(self, encrypted_state_dict, max_seq_len, d_model, num_heads, d_ff, vocab_size):
        """
        Initializes the SimpleTransformer.
        
        Args:
        - embedding_matrix (np.array): Pre-trained embedding matrix.
        - max_seq_len (int): Maximum length of the input sequences.
        - d_model (int): The dimensionality of the embeddings.
        - num_heads (int): The number of heads in the multi-head attention.
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
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.d_model = d_model
        self.MHA = MultiHeadAttention(num_heads=num_heads)
        for i in range(num_heads):
            self.MHA.set_head_weights(i,mha_weights)
        self.MHA.set_output_weights(mha_output_weights,mha_output_bias)
        
        self.positional_encoding = positional_encoding(max_seq_len, d_model)  # Assuming positional_encoding is defined

        # Initialize weights for feed forward network

        
        self.p_ffn = PositionwiseFeedForwardNetwork(d_model,d_ff)
        self.p_ffn.set_weights(pffnw1,pffnb1,pffnw2,pffnb2)


        self.output_w = encrypted_state_dict['output_linear.weight'] 
        self.output_b = encrypted_state_dict['output_linear.bias'] 

   
    def forward(self, embeddings, ctx, batch_size):
        """
        Process the input texts through the Transformer.

        Args:
        - embeddings (list of str): The input text embeddings
        - ctx: venumpy context
        - batch_size: number of inputs

        Returns:
        - np.array: The output vector after embedding, positional encoding, and attention.
        """
 
        #  # Add positional encoding (up to max_len)
        pos_encoding = self.positional_encoding

       #  # Repeat the positional encoding for each sequence in the batch
        pos_encoding_batch = np.repeat(pos_encoding[np.newaxis, :, :], batch_size, axis=0)
        pos_encoding_batch= encrypt_array(pos_encoding_batch,ctx)

        encoded_vector = embeddings + pos_encoding_batch
       # #  # Apply multi-headed attention 
        attention_output = self.MHA.multi_head_attention(encoded_vector, encoded_vector, encoded_vector, ctx, self.d_model)

       # # #residual connection
        attention_output = attention_output + encoded_vector
       #   #positionwise feed forward network
        ff_output = self.p_ffn.forward(attention_output, ctx)
       #  # #residual connection


        ff_output = ff_output + attention_output

       #    # # Pass through the final linear layer
       # #  # Apply the output linear and softmax layer
        cls_output = ff_output[:, 0, :]
        output = output_linear_layer(cls_output,self.output_w,self.output_b) 
        return output