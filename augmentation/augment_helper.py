import tensorflow as tf

def _compute_crop_shape(
    image_height, image_width, aspect_ratio, crop_proportion):
  """Compute aspect ratio-preserving shape for central crop.
  The resulting shape retains `crop_proportion` along one side and a proportion
  less than or equal to `crop_proportion` along the other side.
  Args:
    image_height: Height of image to be cropped.
    image_width: Width of image to be cropped.
    aspect_ratio: Desired aspect ratio (width / height) of output.
    crop_proportion: Proportion of image to retain along the less-cropped side.
  Returns:
    crop_height: Height of image after cropping.
    crop_width: Width of image after cropping.
  """
  image_width_float = tf.cast(image_width, tf.float32)
  image_height_float = tf.cast(image_height, tf.float32)

  def _requested_aspect_ratio_wider_than_image():
    crop_height = tf.cast(tf.rint(
        crop_proportion / aspect_ratio * image_width_float), tf.int32)
    crop_width = tf.cast(tf.rint(
        crop_proportion * image_width_float), tf.int32)
    return crop_height, crop_width

  def _image_wider_than_requested_aspect_ratio():
    crop_height = tf.cast(
        tf.rint(crop_proportion * image_height_float), tf.int32)
    crop_width = tf.cast(tf.rint(
        crop_proportion * aspect_ratio *
        image_height_float), tf.int32)
    return crop_height, crop_width

  return tf.cond(
      aspect_ratio > image_width_float / image_height_float,
      _requested_aspect_ratio_wider_than_image,
      _image_wider_than_requested_aspect_ratio)


def center_crop(image, height, width, crop_proportion):
  """Crops to center of image and rescales to desired size.
  Args:
    image: Image Tensor to crop.
    height: Height of image to be cropped.
    width: Width of image to be cropped.
    crop_proportion: Proportion of image to retain along the less-cropped side.
  Returns:
    A `height` x `width` x channels Tensor holding a central crop of `image`.
  """
  shape = tf.shape(image)
  image_height = shape[0]
  image_width = shape[1]
  crop_height, crop_width = _compute_crop_shape(
      image_height, image_width, height / width, crop_proportion)
  offset_height = ((image_height - crop_height) + 1) // 2
  offset_width = ((image_width - crop_width) + 1) // 2
  image = tf.image.crop_to_bounding_box(
      image, offset_height, offset_width, crop_height, crop_width)

  image = tf.image.resize_bicubic([image], [height, width])[0]

  return image


def distorted_bounding_box_crop(image,
                                bbox,
                                min_object_covered=0.1,
                                aspect_ratio_range=(0.75, 1.33),
                                area_range=(0.05, 1.0),
                                max_attempts=100,
                                scope=None):
    """Generates cropped_image using one of the bboxes randomly distorted.
    See `tf.image.sample_distorted_bounding_box` for more documentation.
    """
    with tf.name_scope(scope or 'distorted_bounding_box_crop'):
        shape = tf.shape(image)
        sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
            shape,
            bounding_boxes=bbox,
            min_object_covered=min_object_covered,
            aspect_ratio_range=aspect_ratio_range,
            area_range=area_range,
            max_attempts=max_attempts,
            use_image_if_no_bounding_boxes=True)
        bbox_begin, bbox_size, _ = sample_distorted_bounding_box

        # Crop the image to the specified bounding box.
        offset_y, offset_x, _ = tf.unstack(bbox_begin)
        target_height, target_width, _ = tf.unstack(bbox_size)
        cropped_image = tf.image.crop_to_bounding_box(
            image, offset_y, offset_x, target_height, target_width)

        return cropped_image


def crop_and_resize(image, height, width):
  """Make a random crop and resize it to height `height` and width `width`.
  Args:
    image: Tensor representing the image.
    height: Desired image height.
    width: Desired image width.
  Returns:
    A `height` x `width` x channels Tensor holding a random crop of `image`.
  """
  bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
  aspect_ratio = width / height
  image = distorted_bounding_box_crop(
      image,
      bbox,
      min_object_covered=0.1,
      aspect_ratio_range=(3. / 4 * aspect_ratio, 4. / 3. * aspect_ratio),
      area_range=(0.08, 1.0),
      max_attempts=100,
      scope=None)
  resized_image = tf.image.resize(image, [height, width], method=tf.image.ResizeMethod.BICUBIC)
  return resized_image

def crop_and_resize_and_flip(image, target_height, target_width, flip_probability=0.5):
    """
    Randomly crops, resizes, and conditionally flips an image horizontally.

    Inputs:
    - image: Tensor representing the image.
    - target_height: Desired height to resize the image.
    - target_width: Desired width to resize the image.
    - flip_probability: Probability of flipping the image horizontally (default 0.5).

    Outputs:
    - Transformed image tensor.
    """
    if tf.random.uniform([]) < flip_probability:
        # Crop and resize the image
        cropped_and_resized_image = crop_and_resize(image, target_height, target_width)

        # Flip the image
        flipped_image = tf.image.random_flip_left_right(cropped_and_resized_image)

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