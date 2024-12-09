
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Input, Dropout, LayerNormalization, Embedding
from tensorflow.keras.models import Model

# Set hyperparameters
IMG_HEIGHT = 224
IMG_WIDTH = 224
NUM_CLASSES = 27
NUM_PATCHES = 64
PROJECTION_DIM = 64
NUM_HEADS = 4
TRANSFORMER_UNITS = [64, 32]
EPOCHS = 20

import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Input, Dropout, LayerNormalization, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam


import tensorflow as tf

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


# ------------- Custom GELU Activation Function -------------
def gelu(x):
    """Gaussian Error Linear Unit activation function."""
    return 0.5 * x * (1 + tf.math.erf(x / tf.sqrt(2.0)))

# ------------- Custom Layers for ViT -------------

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
# ------------- Vision Transformer (ViT) Model -------------

def create_vit_classifier(input_shape, num_classes, num_patches=64, projection_dim=64, transformer_units=2048, num_heads=4, num_layers=12):
    """
    Creates a Vision Transformer (ViT) classifier model.

    Args:
        input_shape (tuple): Shape of the input images (height, width, channels).
        num_classes (int): Number of output classes.
        num_patches (int): Number of patches to divide the image into.
        projection_dim (int): Dimension to project each patch.
        transformer_units (int): Number of units in the transformer feed-forward network.
        num_heads (int): Number of attention heads.
        num_layers (int): Number of transformer encoder layers.

    Returns:
        Model: Compiled ViT model.
    """
    inputs = Input(shape=input_shape)
    
    # Calculate patch size
    patch_size = input_shape[0] // int(num_patches**0.5)  # Assuming square patches
    patches = PatchExtractor(patch_size=patch_size)(inputs)
    
    # Patch embedding
    x = PatchEmbedding(projection_dim)(patches)
    
    # Positional embedding
    x = PositionalEmbedding(num_patches, projection_dim)(x)
    
    # Transformer blocks
    for _ in range(num_layers):
        x = TransformerEncoder(num_heads=num_heads, projection_dim=projection_dim, transformer_units=transformer_units)(x)
    
    # Global average pooling
    x = GlobalAveragePoolingLayer()(x)
    x = Dropout(0.5)(x)
    
    # Classification head
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model

# ------------- Data Augmentation and Generators -------------

# Path to your dataset directory
dataset_dir = "plantvillage_dataset/color"  # Replace with your dataset path

# ImageDataGenerator for data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1.0 / 127.5,  # Scale images to [-1, 1]
    validation_split=0.2,  # 80% training, 20% validation
    horizontal_flip=True,
    vertical_flip=True,
    zoom_range=0.2,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

# Training data generator
train_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=32,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

# Validation data generator
val_generator = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# Number of classes based on the dataset
num_classes = train_generator.num_classes

# ------------- Instantiate and Compile the ViT Model -------------

# Create the Vision Transformer model
vit_model = create_vit_classifier(
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
    num_classes=num_classes,
    num_patches=64,         # 8x8 patches for 224x224 images
    projection_dim=64,      # Embedding dimension
    transformer_units=2048, # FFN dimension
    num_heads=4,            # Number of attention heads
    num_layers=12           # Number of transformer layers
)

# Compile the model
vit_model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss=focal_loss(gamma=2., alpha=0.25),
    metrics=['accuracy','Precision', 'Recall', 'AUC']
)

# Save the model



# Display the model architecture
vit_model.summary()

# ------------- Define Callbacks -------------

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint('best_vit_modelxxx.keras', save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
]

# ------------- Train the Model -------------

history = vit_model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# ------------- Evaluate the Model -------------

val_loss, val_accuracy = vit_model.evaluate(val_generator)
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

# ------------- Save the Final Model -------------

vit_model.save('vit_model_finalxxx.keras')
print("Model saved successfully.")