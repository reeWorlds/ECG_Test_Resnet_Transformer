import tensorflow as tf
from tensorflow import keras
from keras import layers

class TokenizationBlock(layers.Layer):
	def __init__(self, n_embd):
		super(TokenizationBlock, self).__init__()

		self.n_embd = n_embd
		
		self.conv1 = layers.Conv1D(n_embd // 2, kernel_size = 5, padding='same', activation='relu',
							 kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0))
		self.batch1 = layers.BatchNormalization()
		self.maxpool1 = layers.MaxPooling1D(pool_size = 5)

		self.conv2 = layers.Conv1D(n_embd, kernel_size = 3, padding='same', activation='relu',
							 kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0))
		self.batch2 = layers.BatchNormalization()
		self.maxpool2 = layers.MaxPooling1D(pool_size = 3)

		self.pos_emdb = layers.Embedding(input_dim=100, output_dim=n_embd, 
							embeddings_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0))

		self.drop = layers.Dropout(0.5)

	def call(self, inputs):
		x = self.conv1(inputs)
		x = self.batch1(x)
		x = self.maxpool1(x)

		x = self.conv2(x)
		x = self.batch2(x)
		x = self.maxpool2(x)

		x = x + self.pos_emdb(tf.range(100))

		return self.drop(x)

	def get_config(self):
		config = super().get_config()
		config.update({ "n_embd": self.n_embd })
		return config


class MultiHeadSelfAttentionBlock(layers.Layer):
	def __init__(self, n_embd, n_heads):
		super(MultiHeadSelfAttentionBlock, self).__init__()
		
		self.n_embd = n_embd
		self.n_heads = n_heads
		self.head_embd = n_embd // n_heads
		
		# Note that trick is used here
		# In reality keyW/queryW/valueW consist of n_heads heads each with head_embd dimension
		# however heads are emulated with single dense matrix instead of n_heads dense matrices
		self.keyW = layers.Dense(n_embd,
						  kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1))
		self.queryW = layers.Dense(n_embd,
							 kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1))
		self.valueW = layers.Dense(n_embd,
							 kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.1))
		
		self.projW = layers.Dense(n_embd,
							kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0))
		
		self.drop1 = layers.Dropout(0.5)
		self.drop2 = layers.Dropout(0.5)

	def reshapeForHeads(self, x, batch_size):
		x = tf.reshape(x, (batch_size, -1, self.n_heads, self.head_embd)) # (B, T, n_embd) -> (B, T, n_heads, head_embd)
		return tf.transpose(x, perm = [0, 2, 1, 3]) # (B, T, n_heads, head_embd) -> (B, n_heads, T, head_embd)

	def call(self, x):
		batch_size = tf.shape(x)[0] # (B, T, n_embd), where B is batch size, T is number of tokens in sequence
		
		# calculate k, q, v
		k = self.keyW(x) # (B, T, N)
		q = self.keyW(x) # (B, T, N)
		v = self.keyW(x) # (B, T, N)
		
		# reshape k, q, v so each head will be processed separately (B, n_heads, T, head_embd)
		k = self.reshapeForHeads(k, batch_size)
		q = self.reshapeForHeads(q, batch_size)
		v = self.reshapeForHeads(v, batch_size)
		
		# find attention weights - how well key of each token correponds to query of each token
		# (B, n_heads, T, head_embd) * (B, n_heads, head_embd, T) -> (B, n_heads, T, T)
		weights = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(tf.cast(self.head_embd, tf.float32))
		weights = tf.nn.softmax(weights, axis=-1)
		weights = self.drop1(weights)
		
		# for each token query we now have weights of keys of other tokens
		# we now multiply these weights with Values of all tokens for each dimension of Values
		# summary: [for each token query][for each Value dimension] find dot product over all corresponding tokens
		# (B, n_heads, T, T) * (B, n_heads, T, head_embd) -> (B, n_heads, T, head_embd)
		output = tf.matmul(weights, v)
		output = tf.transpose(output, perm = [0, 2, 1, 3]) # (B, n_heads, T, head_embd) -> (B, T, n_heads, head_embd)
		output = tf.reshape(output, (batch_size, -1, self.n_embd)) # (B, T, n_heads, head_embd) -> (B, T, n_embd)
		
		# apply final projection to combine information from different heads
		return self.drop2(self.projW(output))

	def get_config(self):
		config = super().get_config()
		config.update({ "n_embd": self.n_embd, "n_heads": self.n_heads })
		return config


class TransformerEncoderBlock(layers.Layer):
	def __init__(self, n_embd, n_heads, scale):
		super(TransformerEncoderBlock, self).__init__()

		self.n_embd = n_embd
		self.n_heads = n_heads
		self.scale = scale
		
		self.norm1 = layers.LayerNormalization()
		self.mhsa = MultiHeadSelfAttentionBlock(n_embd, n_heads)
		
		self.norm2 = layers.LayerNormalization()
		self.ff1 = layers.Dense(n_embd * scale, activation='relu',
						  kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0))
		self.ff2 = layers.Dense(n_embd,
						  kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0))
		self.drop = layers.Dropout(0.5)

	def call(self, x):
		x = x + self.mhsa(self.norm1(x))
		x = x + self.drop(self.ff2(self.ff1(self.norm2(x))))
		return x

	def get_config(self):
		config = super().get_config()
		config.update({ "n_embd": self.n_embd, "n_heads": self.n_heads, "scale": self.scale })
		return config


class SequencePoolingBlock(layers.Layer):
	def __init__(self):
		super(SequencePoolingBlock, self).__init__()
		
		self.norm = layers.LayerNormalization()
		self.ll = layers.Dense(1, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0))
		self.drop = layers.Dropout(0.5)

	def call(self, x):
		input_shape = tf.shape(x)
		
		x = self.norm(x)
		
		y = tf.reshape(self.ll(x), (input_shape[0], 1, -1))
		y = tf.nn.softmax(y)
		
		output = tf.matmul(y, x)
		output = tf.squeeze(output, axis=1)
		
		return self.drop(output)


class Conv1DCompactTransformer(layers.Layer):
	def __init__(self, n_blocks, n_embd, n_heads, scale, **kwargs):
		super(Conv1DCompactTransformer, self).__init__(**kwargs)

		self.n_blocks = n_blocks
		self.n_embd = n_embd
		self.n_heads = n_heads
		self.scale = scale
		
		self.tokenizer = TokenizationBlock(n_embd)
		self.blocks = keras.Sequential([TransformerEncoderBlock(n_embd, n_heads, scale) for i in range(n_blocks)])
		self.seqpool = SequencePoolingBlock()
		
		self.head_p1 = layers.Dense(n_embd // 2, activation='relu',
							  kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0))
		self.head_p2 = layers.Dense(3, activation = 'softmax',
							  kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0))

	def call(self, x):
		x = self.tokenizer(x)
		x = self.blocks(x)
		x = self.seqpool(x)
		x = self.head_p2(self.head_p1(x))
		
		return x

	def get_config(self):
		config = super().get_config()
		config.update({ "n_blocks": self.n_blocks, "n_embd": self.n_embd, "n_heads": self.n_heads, "scale": self.scale })
		return config