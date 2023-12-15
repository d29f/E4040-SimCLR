import tensorflow as tf

def crop_and_resize(image, target_height, target_width, prob=0.5):
    '''
    Randomly crops and resizes an image.
    Used the paper suggested sample_distorted_bounding_box function.

    Inputs:
    - image: An image tensor.
    - target_height: The height to resize the image to.
    - target_width: The width to resize the image to.
    - prob: The probability of applying the crop and resize.

    Outputs:
    - Transformed image tensor.
    '''
    if tf.random.uniform([]) < prob:
        # Crop the image
        ratio = target_width / target_height
        cropped_image = tf.image.sample_distorted_bounding_box(
            tf.shape(image),
            bounding_boxes=tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4]),
            aspect_ratio_range=(0.75 * ratio, 4.0/3 * ratio),
            area_range=(0.08, 1.0))[0]
        
        # Resize the image
        resized_image = tf.image.resize(cropped_image, [target_height, target_width])
        return resized_image
    else:
        return image

def crop_and_resize_and_flip(image, target_height, target_width, prob=0.5):
    """
    Randomly crops, resizes, and flips an image.
    Used the paper suggested sample_distorted_bounding_box function.

    Inputs:
    - image: An image tensor.
    - target_height: The height to resize the image to.
    - target_width: The width to resize the image to.
    - prob: The probability of applying the crop and resize and flip.

    Outputs:
    - Transformed image tensor.
    """
    if tf.random.uniform([]) < prob:
        # Crop the image
        ratio = target_width / target_height
        cropped_image = tf.image.sample_distorted_bounding_box(
            tf.shape(image),
            bounding_boxes=tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4]),
            aspect_ratio_range=(0.75 * ratio, 4.0/3 * ratio),
            area_range=(0.08, 1.0))[0]
        
        # Resize the image
        resized_image = tf.image.resize(cropped_image, [target_height, target_width])

        # Flip the image
        flipped_image = tf.image.random_flip_left_right(resized_image)

        return flipped_image
    else:
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
        height, width, channels = image.shape

        # Choose the center of the patch
        center_height = tf.random.uniform([], minval=0, maxval=height, dtype=tf.int32)
        center_width = tf.random.uniform([], minval=0, maxval=width, dtype=tf.int32)

        # Calculate the lower and upper bounds of the patch
        lower_height = tf.maximum(0, center_height - patch_size // 2)
        upper_height = tf.minimum(height, center_height + patch_size // 2)
        lower_width = tf.maximum(0, center_width - patch_size // 2)
        upper_width = tf.minimum(width, center_width + patch_size // 2)

        # Create a mask
        mask = tf.ones_like(image)
        patch = tf.zeros((upper_height - lower_height, upper_width - lower_width, channels), dtype=image.dtype)
        mask = tf.tensor_scatter_nd_update(mask, [[lower_height, lower_width, 0]], [patch])

        # Apply the mask
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