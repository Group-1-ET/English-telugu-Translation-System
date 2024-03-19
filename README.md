# English to Telugu Translation System

## Aim

This project aims to create a proficient English to Telugu translation system using transformer architectures in deep learning. By bridging language gaps, it enhances communication and accessibility, offering an advanced tool for seamless translation between English and Telugu.

## Progress of the project

<img width="893" alt="image" src="https://github.com/Group-1-ET/English-telugu-Translation-System/assets/82363361/618d09c7-5078-49bb-ba02-d403c1d1a093">

## Steps 

### 1. Dataset

https://github.com/himanshudce/Indian-Language-Dataset  (It has huge data for low resourced languages like Tamil, Malayalam, Telugu or Bengali)

### 2. Datacleaning

1. Converted all contracted words like to wont,cant to natural form (would not, can not)
2. Converted all the text into lower letters in english input as for example word Pineapple and pineapple mean the same
3. Removed these characters: {'$', ')', '?', '"', '’', '.',  '°', '!', ';', '/', "'", '€', '%', ':', ',', '('}
4. Removed all the spacial characters: except space ' '

### 3. Creating Vocab file from our corpus
### 4. Custom tokenizer (BERT in our case)
#### Bert Tokeniser ( Bidirectional Encoder Representations )

Traditional language models process text sequentially, either from left to right or right to left. This method limits the model’s awareness to the immediate context preceding the target word. BERT uses a bi-directional approach considering both the left and right context of words in a sentence, instead of analyzing the text sequentially, BERT looks at all the words in a sentence simultaneously

```python
class CustomTokenizer(tf.Module):
  def __init__(self, reserved_tokens, vocab_path):
    self.tokenizer = text.BertTokenizer(vocab_path, lower_case=True)
    self._reserved_tokens = reserved_tokens
    self._vocab_path = tf.saved_model.Asset(vocab_path)

    vocab = pathlib.Path(vocab_path).read_text(encoding="utf8").splitlines()
    self.vocab = tf.Variable(vocab)

    ## Create the signatures for export:   

    # Include a tokenize signature for a batch of strings. 
    self.tokenize.get_concrete_function(
        tf.TensorSpec(shape=[None], dtype=tf.string))

    # Include `detokenize` and `lookup` signatures for:
    #   * `Tensors` with shapes [tokens] and [batch, tokens]
    #   * `RaggedTensors` with shape [batch, tokens]
    self.detokenize.get_concrete_function(
        tf.TensorSpec(shape=[None, None], dtype=tf.int64))
    self.detokenize.get_concrete_function(
          tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64))

    self.lookup.get_concrete_function(
        tf.TensorSpec(shape=[None, None], dtype=tf.int64))
    self.lookup.get_concrete_function(
          tf.RaggedTensorSpec(shape=[None, None], dtype=tf.int64))

    # These `get_*` methods take no arguments
    self.get_vocab_size.get_concrete_function()
    self.get_vocab_path.get_concrete_function()
    self.get_reserved_tokens.get_concrete_function()

  @tf.function
  def tokenize(self, strings):
    enc = self.tokenizer.tokenize(strings)
    # Merge the `word` and `word-piece` axes.
    enc = enc.merge_dims(-2,-1)
    enc = add_start_end(enc)
    return enc

  @tf.function
  def detokenize(self, tokenized):
    words = self.tokenizer.detokenize(tokenized)
    return cleanup_text(self._reserved_tokens, words)

  @tf.function
  def lookup(self, token_ids):
    return tf.gather(self.vocab, token_ids)

  @tf.function
  def get_vocab_size(self):
    return tf.shape(self.vocab)[0]

  @tf.function
  def get_vocab_path(self):
    return self._vocab_path

  @tf.function
  def get_reserved_tokens(self):
    return tf.constant(self._reserved_tokens)
```

**Initialization**: The __init__ method initializes the custom tokenizer. It uses a BERT tokenizer provided by TensorFlow to tokenize text. This tokenizer is initialized with a vocabulary file (vocab_path) and a parameter for whether to convert text to lowercase (lower_case=True). Additionally, the method stores the reserved tokens and the path to the vocabulary file as attributes.

**Tokenization**: The tokenize method tokenizes input strings using the BERT tokenizer. It takes a batch of strings as input and returns the tokenized sequences. After tokenization, the method merges the axes representing individual tokens and word-pieces into a single axis. It also adds start and end tokens to the tokenized sequences, which are important for various natural language processing tasks.

**Detokenization**: The detokenize method reverses the tokenization process. It takes tokenized sequences as input and converts them back into strings of text. Additionally, the method cleans up the text by removing any reserved tokens that were added during tokenization.

**Vocabulary Lookup**: The lookup method retrieves the actual tokens from the vocabulary based on their token IDs. Given a set of token IDs, it returns the corresponding tokens from the vocabulary.

**Helper Function**s: Lastly, the code includes helper functions to retrieve metadata about the tokenizer. These functions allow users to obtain information such as the size of the vocabulary, the path to the vocabulary file, and the reserved tokens.

### 5. Transformer model 
![image](https://github.com/Group-1-ET/English-telugu-Translation-System/assets/82363361/f276b721-93f5-4252-b97a-723dfa5825a1)
#### Positional Encoding
Positional encoding is a technique used in Transformer-based models, such as the Transformer architecture used in BERT, to inject information about the positions of words or tokens into the input embeddings. This is crucial because these models do not inherently understand the order or position of words in a sequence since they process tokens in parallel rather than sequentially.
<img width="893" alt="image" src="https://github.com/Group-1-ET/English-telugu-Translation-System/assets/82363361/b4b84ba6-0ca6-472f-a16c-d22400d67de1">
```python
def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates

def positional_encoding(position, d_model):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)

  # apply sin to even indices in the array; 2i
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

  # apply cos to odd indices in the array; 2i+1
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

  pos_encoding = angle_rads[np.newaxis, ...]

  return tf.cast(pos_encoding, dtype=tf.float32)

n, d = 2048, 512
pos_encoding = positional_encoding(n, d)
print(pos_encoding.shape)
pos_encoding = pos_encoding[0]

# Juggle the dimensions for the plot
pos_encoding = tf.reshape(pos_encoding, (n, d//2, 2))
pos_encoding = tf.transpose(pos_encoding, (2, 1, 0))
pos_encoding = tf.reshape(pos_encoding, (d, n))

plt.pcolormesh(pos_encoding, cmap='RdBu')
plt.ylabel('Depth')
plt.xlabel('Position')
plt.colorbar()
plt.show()

```
![image](https://github.com/Group-1-ET/English-telugu-Translation-System/assets/82363361/d5a65bfa-d6e5-4b36-99bf-d936402ddf3d)


### 6. Training and testing
 


