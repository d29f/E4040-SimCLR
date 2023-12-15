import tensorflow as tf

def random_brightness(image, max_delta):
    random_factor = 1.0 + tf.random.uniform([], -max_delta, max_delta)
    image = image * random_factor
    # to ensure the values are within a reasonable range
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image


def random_color_jitter(image, strength=1):
    # Apply random adjustments to the image's brightness, contrast, saturation, and hue
    image = tf.image.random_brightness(image, max_delta=0.8*strength)
    image = tf.image.random_contrast(image, lower=1-0.8*strength, upper=1+0.8*strength)
    image = tf.image.random_saturation(image, lower=1-0.8*strength, upper=1+0.8*strength)
    image = tf.image.random_hue(image, max_delta=0.2*strength)

    # Clip values to stay within the valid range
    image = tf.clip_by_value(image, 0, 1)
    return image

def center_crop(image, height, width):
    # Calculate the dimensions of the image and determine the scaling factor for cropping
    shape = tf.shape(image)
    image_height, image_width = shape[0], shape[1]

    scale = tf.minimum(image_width / width, image_height / height)
    new_height = tf.cast(height * scale, tf.int32)
    new_width = tf.cast(width * scale, tf.int32)

    # Resize and then crop the image to the specified dimensions
    image = tf.image.resize(image, [new_height, new_width], method='nearest')
    offset_height = (new_height - height) // 2
    offset_width = (new_width - width) // 2

    image = tf.image.crop_to_bounding_box(image, offset_height, offset_width, height, width)
    return image


def crop_and_resize(image, target_height, target_width, bbox=None, random_crop=False):
    if random_crop:
        shape = tf.shape(image)
        random_ymin = tf.random.uniform([], 0, shape[0] - target_height, dtype=tf.int32)
        random_xmin = tf.random.uniform([], 0, shape[1] - target_width, dtype=tf.int32)

        ymin, xmin = tf.cast(random_ymin, tf.float32) / tf.cast(shape[0], tf.float32), tf.cast(random_xmin, tf.float32) / tf.cast(shape[1], tf.float32)
        ymax, xmax = tf.cast(random_ymin + target_height, tf.float32) / tf.cast(shape[0], tf.float32), tf.cast(random_xmin + target_width, tf.float32) / tf.cast(shape[1], tf.float32)
    elif bbox is not None:
        ymin, xmin, ymax, xmax = bbox
    else:
        bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32)
        ymin, xmin, ymax, xmax = bbox

    crop_height = tf.cast((ymax - ymin) * tf.cast(tf.shape(image)[0], tf.float32), tf.int32)
    crop_width = tf.cast((xmax - xmin) * tf.cast(tf.shape(image)[1], tf.float32), tf.int32)

    ymin = tf.cast(ymin * tf.cast(tf.shape(image)[0], tf.float32), tf.int32)
    xmin = tf.cast(xmin * tf.cast(tf.shape(image)[1], tf.float32), tf.int32)


    image = tf.image.crop_to_bounding_box(image, ymin, xmin, crop_height, crop_width)
    image = tf.image.resize(image, [target_height, target_width])

    return image

def color_drop(image):
    image = tf.image.rgb_to_grayscale(image)  
    image = tf.image.grayscale_to_rgb(image)  
    
    return image