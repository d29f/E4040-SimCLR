import tensorflow as tf
import yaml
from model import ResNetSimCLR
from augmentation import augment_image

def load_dataset(file_pattern, batch_size):
    # Load dataset from file pattern and apply data augmentation

    files = tf.data.Dataset.list_files(file_pattern)
    dataset = files.interleave(tf.data.TFRecordDataset, cycle_length=tf.data.experimental.AUTOTUNE)

    dataset = dataset.map(augment_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset

def contrastive_loss(batch_size, temperature=0.1):
    # Define contrastive loss function (NT-Xent loss)
    def loss_fn(z_i, z_j):
        """
        Calculate the NT-Xent loss.

        Parameters:
        - z_i, z_j: Outputs from the two augmented views of the images,
                    with shapes (batch_size, feature_dim).

        Returns:
        - Scalar loss value.
        """

        # Concatenate the projections for positive and negative pairs
        z = tf.concat([z_i, z_j], axis=0)

        # Normalize the projections to unit vectors
        z = tf.math.l2_normalize(z, axis=1)

        # Compute cosine similarity as dot product of normalized vectors
        similarity_matrix = tf.matmul(z, z, transpose_b=True)

        # Scale similarity by temperature
        similarity_matrix = similarity_matrix / temperature

        # Create labels for positive pairs (matching augmented images)
        labels = tf.range(batch_size)
        labels = tf.concat([labels, labels], axis=0)

        # Create a mask to exclude self-comparisons (diagonal elements)
        mask = tf.one_hot(labels, 2 * batch_size)
        logits_mask = tf.logical_not(tf.cast(mask, dtype=tf.bool))
        masked_similarity_matrix = tf.boolean_mask(similarity_matrix, logits_mask)

        # Reshape logits for cross-entropy calculation
        masked_similarity_matrix = tf.reshape(masked_similarity_matrix, (2 * batch_size, -1))
        labels = tf.repeat(labels, batch_size * 2 - 1)

        # Compute cross-entropy loss between similarities and labels
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=masked_similarity_matrix, labels=labels)

        # Average the loss across the batch
        loss = tf.reduce_mean(loss)

        return loss
    return loss_fn


# Load configuration from YAML file
config = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)

# Prepare the training dataset
train_dataset = load_dataset(config['data_path'], batch_size=config['batch_size'])

# Initialize the SimCLR model with specified input and output dimensions
model = ResNetSimCLR(config['input_size'], config['output_size'])

# Initialize the contrastive loss function with model and temperature
loss_fn = contrastive_loss(model, temperature=config['temperature'])

# Set optimizer for training
optimizer = tf.keras.optimizers.Adam(learning_rate=config['learning_rate'])

# Training loop setup
epochs = config['epochs']    

for epoch in range(epochs):
    total_loss = 0
    num_batches = 0

    for images, _ in train_dataset:
        with tf.GradientTape() as tape:
            # Forward pass through the model for both sets of augmented images
            _, proj1 = model(images[0], training=True)
            _, proj2 = model(images[1], training=True)

            # Calculate loss
            loss = loss_fn(proj1, proj2)

        # Compute and apply gradients
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # Accumulate loss for reporting
        total_loss += loss
        num_batches += 1

    # Calculate and display average loss for the epoch
    avg_loss = total_loss / num_batches
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

# Save the trained model
model.save('./saved_models')