
# transformer_module.py
import sys
sys.path.append('../..')
import numpy as np
import torch
from transformers import BertTokenizerFast
from venumML.venumpy import small_glwe as vp
from tqdm import tqdm
from venumML.venum_tools import *
from venumML.approx_functions import *
from venumML.deep_learning.transformer.transformer import *



# Define a function to initialize the encryption context
def initialize_encryption_context(precision=6):
    ctx = vp.SecretContext()
    ctx.precision = precision
    return ctx

# Tokenizer setup
def load_tokenizer():
    return BertTokenizerFast.from_pretrained("bert-base-uncased")

# Tokenization helper function
def tokenize_input(sentence, tokenizer, max_seq_len):
    inputs = tokenizer.encode_plus(
        sentence,
        add_special_tokens=True,
        max_length=max_seq_len,
        padding='max_length',
        return_tensors='pt',
        truncation=True
    )
    return inputs['input_ids'].squeeze().numpy()

# Load and encrypt model weights
def encrypt_weights(state_dict, ctx):
    encrypted_state_dict = {}
    for k in tqdm(state_dict.keys(), desc="Encrypting weights"):
        weight = state_dict[k].T.numpy()
        encrypted_state_dict[k] = encrypt_array(weight, ctx)
    return encrypted_state_dict


def log_softmax(x):
    # Subtract the max for numerical stability
    x_max = np.max(x, axis=-1, keepdims=True)
    
    # Log of the sum of exponentials of the input elements
    log_sum_exp = np.log(np.sum(np.exp(x - x_max), axis=-1, keepdims=True))
    
    return x - x_max - log_sum_exp

def softmax(logits):
    """
    Compute the softmax probabilities for the input logits.
    
    Args:
        logits (array-like): Logits (unnormalized scores) for each class.

    Returns:
        numpy.ndarray: Probabilities for each class.
    """
    exp_logits = np.exp(logits - np.max(logits))  # Subtract max for numerical stability
    return exp_logits / np.sum(exp_logits)

def to_classes(outputs):
    # Convert logits to log-probabilities using log_softmax
    log_probs = log_softmax(outputs)
    # Convert log-probabilities to probabilities
    probabilities = np.exp(log_probs)
    # Choose the class with the highest probability
    predicted_classes = np.argmax(probabilities, axis=1)
    return probabilities, predicted_classes

# Embedding and batching utilities
def texts_to_batch_indices(texts, tokenizer, max_seq_len):
    batch_indices = [tokenize_input(text, tokenizer, max_seq_len) for text in texts]
    return batch_indices

def decrypt_array(encrypted_array):
    return np.array([element.decrypt() for element in encrypted_array])


def encrypt_user_summary( user_summary,embeddings,tokenizer,max_seq_len,ctx):

    encrypted_user_summary = {}
    for user in tqdm(user_summary.keys()):
        
        text = [user_summary[user]]
        batch_indices = texts_to_batch_indices(text, tokenizer, max_seq_len)
        batch_indices = texts_to_batch_indices(text,tokenizer, max_seq_len)
        embedding_output = embeddings.forward(batch_indices, batch_size=1,max_seq_len= max_seq_len)
        encrypted_user_summary[user] = encrypt_array(embedding_output, ctx)
    return encrypted_user_summary


# Transformer Inference class
class TransformerInference:
    def __init__(self, model_weights_path, tokenizer, encryption_context, max_seq_len, d_model, num_heads, d_ff, vocab_size, class_size):
        self.ctx = encryption_context
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.vocab_size = vocab_size
        self.class_size = class_size

        # Load model weights
        self.state_dict = torch.load(model_weights_path, map_location=torch.device('cpu'))
        self.encrypted_state_dict = encrypt_weights(self.state_dict, self.ctx)

    def predict(self, user_summary):
        transformer = TransformerModule(self.encrypted_state_dict, max_seq_len=self.max_seq_len, d_model=self.d_model, num_heads=self.num_heads, d_ff=self.d_ff, vocab_size=self.vocab_size)
        encrypted_classifications = {}
        for user in tqdm(user_summary.keys()):

            output_linear = transformer.forward(user_summary[user], self.ctx, 1)
            encrypted_classifications[user] = output_linear
        return encrypted_classifications