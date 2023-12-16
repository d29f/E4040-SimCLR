import tensorflow as tf

def crop_and_resize(image, target_height, target_width, bbox=None, random_crop=False, prob=1.0):
    '''
    Randomly crops and resizes an image if random_crop is True, or resizes within a specified bounding box. 
    If bbox is None and random_crop is False, it resizes the whole image.

    Inputs:
    - image: A tensor representing the image.
    - target_height: The height to resize the image to after cropping.
    - target_width: The width to resize the image to after cropping.
    - bbox: A list of floats representing the bounding box [ymin, xmin, ymax, xmax] for cropping.
    - random_crop: A boolean indicating whether to apply random cropping or not.
    - prob: The probability with which to apply the crop and resize.

    Outputs:
    - The transformed image tensor after cropping and resizing.
    '''
    if tf.random.uniform([]) < prob:
        # Only apply cropping if random_crop is True or bbox is provided
        if random_crop:
            shape = tf.shape(image)
            # Ensure the random crop is within the image boundaries
            max_ymin = tf.maximum(0, shape[0] - target_height)
            max_xmin = tf.maximum(0, shape[1] - target_width)
            random_ymin = tf.random.uniform([], 0, max_ymin, dtype=tf.int32)
            random_xmin = tf.random.uniform([], 0, max_xmin, dtype=tf.int32)

            ymin, xmin = tf.cast(random_ymin, tf.float32) / tf.cast(shape[0], tf.float32), tf.cast(random_xmin, tf.float32) / tf.cast(shape[1], tf.float32)
            ymax, xmax = tf.cast(random_ymin + target_height, tf.float32) / tf.cast(shape[0], tf.float32), tf.cast(random_xmin + target_width, tf.float32) / tf.cast(shape[1], tf.float32)
        elif bbox is not None:
            # If a bounding box is provided, use it for cropping
            ymin, xmin, ymax, xmax = bbox
        else:
            # If no bbox or random_crop, use the entire image
            bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32)
            ymin, xmin, ymax, xmax = bbox

        # Calculate the height and width of the cropped area
        crop_height = tf.cast((ymax - ymin) * tf.cast(shape[0], tf.float32), tf.int32)
        crop_width = tf.cast((xmax - xmin) * tf.cast(shape[1], tf.float32), tf.int32)

        # Calculate the starting points of the crop
        ymin = tf.cast(ymin * tf.cast(shape[0], tf.float32), tf.int32)
        xmin = tf.cast(xmin * tf.cast(shape[1], tf.float32), tf.int32)

        # Crop and then resize the image
        image = tf.image.crop_to_bounding_box(image, ymin, xmin, crop_height, crop_width)
        image = tf.image.resize(image, [target_height, target_width])

    return image  

def crop_and_resize(image, target_height, target_width, bbox=None, random_crop=False, prob=0.5):
    """
    Conditionally crops and resizes an image. If random_crop is True, it randomly crops the image.
    If bbox is provided, it crops the image to the specified bounding box.
    The cropped image is then resized to target_height and target_width.
    The operation is applied with a certain probability defined by prob.

    Inputs:
    - image: A tensor representing the image.
    - target_height: The height to resize the image to.
    - target_width: The width to resize the image to.
    - bbox: An optional list of floats representing the bounding box [ymin, xmin, ymax, xmax] for cropping.
    - random_crop: A boolean flag that indicates whether to apply random cropping.
    - prob: The probability of applying the crop and resize operation.

    Outputs:
    - The image tensor after cropping and resizing, or the original image tensor if the operation was not applied.
    """
    if tf.random.uniform([]) < prob:
        if random_crop:
            shape = tf.shape(image)
            # Ensure the random crop does not exceed the image boundaries
            max_ymin = tf.maximum(0, shape[0] - target_height)
            max_xmin = tf.maximum(0, shape[1] - target_width)
            random_ymin = tf.random.uniform([], 0, max_ymin, dtype=tf.int32)
            random_xmin = tf.random.uniform([], 0, max_xmin, dtype=tf.int32)

            # Calculate the normalized coordinates of the crop
            ymin, xmin = tf.cast(random_ymin, tf.float32) / tf.cast(shape[0], tf.float32), tf.cast(random_xmin, tf.float32) / tf.cast(shape[1], tf.float32)
            ymax, xmax = tf.cast(random_ymin + target_height, tf.float32) / tf.cast(shape[0], tf.float32), tf.cast(random_xmin + target_width, tf.float32) / tf.cast(shape[1], tf.float32)
        elif bbox is not None:
            # Use the provided bounding box for cropping
            ymin, xmin, ymax, xmax = bbox
        else:
            # Use the entire image if no bounding box or random crop is specified
            bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32)
            ymin, xmin, ymax, xmax = bbox

        # Compute the actual dimensions of the crop
        crop_height = tf.cast((ymax - ymin) * tf.cast(shape[0], tf.float32), tf.int32)
        crop_width = tf.cast((xmax - xmin) * tf.cast(shape[1], tf.float32), tf.int32)

        # Crop to the bounding box and resize to the target dimensions
        image = tf.image.crop_to_bounding_box(image, tf.cast(ymin * tf.cast(shape[0], tf.float32), tf.int32), tf.cast(xmin * tf.cast(shape[1], tf.float32), tf.int32), crop_height, crop_width)
        image = tf.image.resize(image, [target_height, target_width])

    return image


def color_distort_drop(image):
    '''
    Apply color drop effect to the image.
    Notice this is written based on the pseudo-code provided in the paper.

    Inputss:
    - image: An image tensor.

    Outpuss:
    - Grayscale image tensor.
    '''
    # Apply the color drop effect
    grayscale_image = tf.image.rgb_to_grayscale(image)
    grayscale_image = tf.tile(grayscale_image, [1, 1, 3])
    return grayscale_image

def color_distort_jitter(image, strength=1):
    '''
    Apply color jitter effect to the image.
    Notice this is written based on the pseudo-code provided in the paper.

    Inputs:
    - image: A Tensor representing an image.
    - strength: The strength of the color jitter.

    Outputs:
    - Color jittered image tensor.
    '''
    # set parameters
    brightness_delta = 0.8 * strength
    contrast_lower, contrast_upper = 1 - 0.8 * strength, 1 + 0.8 * strength
    saturation_lower, saturation_upper = 1 - 0.8 * strength, 1 + 0.8 * strength
    hue_delta = 0.2 * strength

    # Apply the jitter effect
    image = tf.image.random_brightness(image, max_delta=brightness_delta)
    image = tf.image.random_contrast(image, lower=contrast_lower, upper=contrast_upper)
    image = tf.image.random_saturation(image, lower=saturation_lower, upper=saturation_upper)
    image = tf.image.random_hue(image, max_delta=hue_delta)

    return tf.clip_by_value(image, 0, 1)

def color_distort(image, prob=0.5):
    '''
    Applies color distortion to the image with a probability prob.
    Notice this is written based on the pseudo-code provided in the paper.

    Inputs:
    - image: An image tensor.
    - prob: The probability of applying the color distortion.

    Outputs:
    - Color distorted (both dropped and jittered) image tensor.
    '''
    if tf.random.uniform([]) < prob:
        # Apply the color drop and jitter effects
        distorted_image = color_distort_drop(image)
        distorted_image = color_distort_jitter(distorted_image)
        return distorted_image
    else:
        return image

def rotate(image, prob=0.5):
    '''
    Rotates the image by a random angle, chosen from 90, 180, or 270 degrees.

    Inputs:
    - image: An image tensor.
    - prob: The probability of applying the rotation.

    Outputs:
    - A Tensor of the rotated image.
    '''
    if tf.random.uniform([]) < prob:
        rotation_counts = [1, 2, 3]
        rotation_choice= tf.random.shuffle(rotation_counts)[0]

        # Rotate the image
        rotated_image = tf.image.rot90(image, k=rotation_choice)
        return rotated_image
    else:
        return image

def cutout(image, patch_size, prob=0.5):
    '''
    Applies the cutout augmentation.

    Inputs:
    - image: An image tensor.
    - patch_size: The size of the square patch to be masked.
    - prob: The probability of applying the cutout.

    Outputs:
    - Image tensor with a patch masked.
    '''
    if tf.random.uniform([]) < prob:
        # Get the dimensions of the image.
        height, width, channels = image.shape

        # Choose a random location for the patch.
        center_height = tf.random.uniform([], minval=0, maxval=height, dtype=tf.int32)
        center_width = tf.random.uniform([], minval=0, maxval=width, dtype=tf.int32)

        # Determine the bounds of the patch.
        half_patch = patch_size // 2
        start_height = tf.clip_by_value(center_height - half_patch, 0, height)
        start_width = tf.clip_by_value(center_width - half_patch, 0, width)
        end_height = tf.clip_by_value(center_height + half_patch, 0, height)
        end_width = tf.clip_by_value(center_width + half_patch, 0, width)

        # Create the mask.
        mask = tf.ones((height, width, channels), dtype=image.dtype)

        # Calculate the indices to update.
        indices = tf.reshape(tf.stack(tf.meshgrid(
            tf.range(start_height, end_height),
            tf.range(start_width, end_width),
            indexing='ij'
        )), (-1, 2))

        # Create the updates (the values to insert).
        updates = tf.zeros((indices.shape[0], channels), dtype=image.dtype)

        # Perform the tensor update.
        mask = tf.tensor_scatter_nd_update(mask, indices, updates)

        # Apply the mask to the image.
        image = image * mask

        return image
    else:
        return image

def gaussian_noise(image, noise_std=0.1, prob=0.5):
    '''
    Adds Gaussian noise to the image with specified standard deviation.

    Inputs:
    - image: An image tensor.
    - noise_std: The standard deviation of the Gaussian noise.
    - prob: The probability of applying the noise.

    Outputs:
    - The image tensor with Gaussian noise added.
    '''
    if tf.random.uniform([]) < prob:
        # Generate Gaussian noise
        noise = tf.random.normal(tf.shape(image), stddev=noise_std)

        # Add the Gaussian noise to the image
        noisy_image = tf.clip_by_value(image + noise, 0, 1)

        return noisy_image
    else:
        return image

def gaussian_blur(image, prob=0.5):
    '''
    Applies Gaussian blur to the image with a probability prob. 
    The sigma value for the Gaussian kernel is randomly chosen between 0.1 and 2.0, and the kernel 
    size is set to 10% of the image's height or width as paper suggested.

    Inputs:
    - image: An image tensor.
    - prob: The probability of applying the gaussian blur.

    Outputss:
    - Blurred image tensor.
    '''
    if tf.random.uniform([]) < prob:
        # Randomly choose the sigma value
        sigma = tf.random.uniform([], minval=0.1, maxval=2.0)

        # Calculate the kernel size
        kernel_size = tf.cast(tf.maximum(tf.shape(image)[0], tf.shape(image)[1]) * 0.1, tf.int32)

        # Generate Gaussian kernel
        kernel = tf.cast(tf.math.exp(-(tf.range(kernel_size) - kernel_size // 2)**2 / (2 * sigma**2)), tf.float32)
        kernel = kernel / tf.reduce_sum(kernel)

        # Create and apply blur filters
        kernel_x = tf.reshape(kernel, [kernel_size, 1, 1, 1])
        kernel_y = tf.reshape(kernel, [1, kernel_size, 1, 1])
        blur_filter = tf.cast(tf.matmul(kernel_x, kernel_y), tf.float32)
        blurred_image = tf.nn.depthwise_conv2d(image[None], blur_filter, [1, 1, 1, 1], 'SAME')[0]

        return blurred_image
    else:
        return image

def sobel_filter(image, prob=0.5):
    '''
    Applies Sobel filtering.

    Inputs:
    - image: An image tensor.
    - prob: The probability of applying the Sobel filter.

    Outputs:
    - Image tensor after sobel filtering.
    '''
    if tf.random.uniform([]) < prob:
        # Sobel filtering
        sobel_image = tf.image.sobel_edges(image)

        # Calculate the magnitude of the edges
        sobel_image = tf.sqrt(tf.reduce_sum(tf.square(sobel_image), axis=-1))
        sobel_image = tf.reduce_sum(sobel_image, axis=-1)
        sobel_image = tf.clip_by_value(sobel_image, 0, 1)

        return sobel_image
    else:
        return image