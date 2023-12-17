import tensorflow as tf


def center_crop(image, height, width):
    """
    Crops the central part of an image to the specified height and width.

    Inputs:
    - image: A tensor representing the image to be cropped.
    - height: The height of the output image after cropping.
    - width: The width of the output image after cropping.

    Outputs:
    - The cropped image tensor with the specified height and width.
    """
    # Calculate the dimensions of the image to determine the scaling factor for cropping
    shape = tf.shape(image)
    image_height, image_width = shape[0], shape[1]

    # Determine the scaling factor based on the target dimensions
    scale = tf.minimum(image_width / width, image_height / height)
    new_height = tf.cast(height * scale, tf.int32)
    new_width = tf.cast(width * scale, tf.int32)

    # Resize the image using the scaling factor and maintain aspect ratio
    image = tf.image.resize(image, [new_height, new_width], method='nearest')

    # Calculate the offsets for cropping to get the central region
    offset_height = (new_height - height) // 2
    offset_width = (new_width - width) // 2

    # Crop the resized image to the specified height and width
    image = tf.image.crop_to_bounding_box(image, offset_height, offset_width, height, width)
    return image



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


def crop_and_resize_and_flip(image, target_height, target_width, prob=1.0):
    """
    Randomly crops, resizes, and flips an image. The cropping is performed using a distorted bounding box approach which is beneficial for training robust neural networks. The image is then resized to the target dimensions and flipped horizontally with a specified probability.

    Inputs:
    - image: A tensor of shape (height, width, channels) representing the image.
    - target_height: The desired height of the image after resizing.
    - target_width: The desired width of the image after resizing.
    - prob: The probability with which to apply the crop, resize, and flip operations. 

    Outputs:
    - The image tensor after potentially cropping, resizing, and flipping. If the random number exceeds the specified probability, the original image tensor is returned.
    """
    if tf.random.uniform([]) < prob:
        # Calculate the aspect ratio of the target image
        ratio = target_width / target_height
        # Use sample_distorted_bounding_box to get the parameters for cropping
        begin, size, _ = tf.image.sample_distorted_bounding_box(
            tf.shape(image),
            bounding_boxes=tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4]),
            aspect_ratio_range=(0.75 * ratio, 4.0/3 * ratio),
            area_range=(0.08, 1.0))
        
        # Crop the image according to the parameters
        cropped_image = tf.slice(image, begin, size)
        # Reshape to ensure the image has three dimensions
        cropped_image = tf.reshape(cropped_image, [size[0], size[1], -1])
        
        # Resize the cropped image to the target dimensions
        resized_image = tf.image.resize(cropped_image, [target_height, target_width])

        # Flip the image horizontally
        flipped_image = tf.image.random_flip_left_right(resized_image)

        return flipped_image
    else:
        # Return the original image if the random number does not fall below prob
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
    
def cutout(image, pad_size, replace=0):
    """Apply cutout (https://arxiv.org/abs/1708.04552) to image.

    This operation applies a (2*pad_size x 2*pad_size) mask of zeros to
    a random location within `img`. The pixel values filled in will be of the
    value `replace`. The located where the mask will be applied is randomly
    chosen uniformly over the whole image.

    Args:
      image: An image Tensor of type uint8.
      pad_size: Specifies how big the zero mask that will be generated is that
        is applied to the image. The mask will be of size
        (2*pad_size x 2*pad_size).
      replace: What pixel value to fill in the image in the area that has
        the cutout mask applied to it.

    Returns:
      An image Tensor that is of type uint8.
    """
    image_height = tf.shape(image)[0]
    image_width = tf.shape(image)[1]

    # Sample the center location in the image where the zero mask will be applied.
    cutout_center_height = tf.random.uniform(
        shape=[], minval=0, maxval=image_height,
        dtype=tf.int32)

    cutout_center_width = tf.random.uniform(
        shape=[], minval=0, maxval=image_width,
        dtype=tf.int32)

    lower_pad = tf.maximum(0, cutout_center_height - pad_size)
    upper_pad = tf.maximum(0, image_height - cutout_center_height - pad_size)
    left_pad = tf.maximum(0, cutout_center_width - pad_size)
    right_pad = tf.maximum(0, image_width - cutout_center_width - pad_size)

    cutout_shape = [image_height - (lower_pad + upper_pad),
                    image_width - (left_pad + right_pad)]
    padding_dims = [[lower_pad, upper_pad], [left_pad, right_pad]]
    mask = tf.pad(
        tf.zeros(cutout_shape, dtype=image.dtype),
        padding_dims, constant_values=1)
    mask = tf.expand_dims(mask, -1)
    mask = tf.tile(mask, [1, 1, 3])
    image = tf.where(
        tf.equal(mask, 0),
        tf.ones_like(image, dtype=image.dtype) * replace,
        image)
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