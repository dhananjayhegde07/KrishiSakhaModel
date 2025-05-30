import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Input, Dropout, LayerNormalization, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras.layers import GlobalAveragePooling1D 
from tensorflow.keras.layers import Embedding

def gelu(x):
    """Gaussian Error Linear Unit activation function."""
    return 0.5 * x * (1 + tf.math.erf(x / tf.sqrt(2.0)))

class PatchExtractor(Layer):
    def __init__(self, patch_size, **kwargs):
        super(PatchExtractor, self).__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        num_patches = (images.shape[1] // self.patch_size) * (images.shape[2] // self.patch_size)
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [-1, num_patches, patch_dims])
        return patches

    def get_config(self):
        config = super(PatchExtractor, self).get_config()
        config.update({"patch_size": self.patch_size})
        return config

class PatchEmbedding(Layer):
    def __init__(self, projection_dim, **kwargs):
        super(PatchEmbedding, self).__init__(**kwargs)
        self.projection = Dense(units=projection_dim)

    def call(self, patches):
        return self.projection(patches)

    def get_config(self):
        config = super(PatchEmbedding, self).get_config()
        config.update({"projection_dim": self.projection.units})
        return config

class PositionalEmbedding(Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super(PositionalEmbedding, self).__init__(**kwargs)
        self.pos_embedding = Embedding(input_dim=num_patches, output_dim=projection_dim)

    def call(self, x):
        positions = tf.range(start=0, limit=tf.shape(x)[1], delta=1)
        pos_emb = self.pos_embedding(positions)
        return x + pos_emb

    def get_config(self):
        config = super(PositionalEmbedding, self).get_config()
        config.update({
            "num_patches": self.pos_embedding.input_dim,
            "projection_dim": self.pos_embedding.output_dim
        })
        return config

class TransformerEncoder(Layer):
    def __init__(self, num_heads, projection_dim, transformer_units, dropout_rate=0.1, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim)
        self.dropout1 = Dropout(dropout_rate)
        self.norm1 = LayerNormalization(epsilon=1e-6)
        
        self.dense_proj = tf.keras.Sequential([
            Dense(transformer_units, activation=gelu),
            Dense(projection_dim)
        ])
        self.dropout2 = Dropout(dropout_rate)
        self.norm2 = LayerNormalization(epsilon=1e-6)

    def call(self, inputs, training=False):
        attn_output = self.attention(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.norm1(inputs + attn_output)
        
        ffn_output = self.dense_proj(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.norm2(out1 + ffn_output)

    def get_config(self):
        config = super(TransformerEncoder, self).get_config()
        config.update({
            "num_heads": self.attention.num_heads,
            "projection_dim": self.attention.key_dim,
            "transformer_units": self.dense_proj.layers[0].units,
            "dropout_rate": self.dropout1.rate
        })
        return config

class GlobalAveragePoolingLayer(Layer):
    def call(self, inputs):
        return tf.reduce_mean(inputs, axis=1)

    def get_config(self):
        return super(GlobalAveragePoolingLayer, self).get_config()
    

@tf.keras.utils.register_keras_serializable()
def focal_loss(gamma=2., alpha=0.25):
    """
    Focal Loss for addressing class imbalance.
    Args:
        gamma (float): Focusing parameter that down-weights easy examples.
        alpha (float): Balancing factor for class imbalance.
    Returns:
        loss (function): Focal loss function.
    """
    @tf.keras.utils.register_keras_serializable()
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        
        # Cross entropy
        cross_entropy = -y_true * tf.math.log(y_pred)
        
        # Focal loss component
        loss = alpha * tf.pow(1 - y_pred, gamma) * cross_entropy
        return tf.reduce_mean(loss, axis=-1)
    
    return focal_loss_fixed