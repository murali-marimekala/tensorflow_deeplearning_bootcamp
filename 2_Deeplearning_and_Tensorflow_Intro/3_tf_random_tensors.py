import tensorflow as tf

# Create two random (but the same) tensors
random_1 = tf.random.Generator.from_seed(7)  # Set seed for reproducibility
random_1 = random_1.normal(shape=(3, 2))
print(f"Random 1 tensor is ", random_1)
random_2 = tf.random.Generator.from_seed(7)
random_2 = random_2.normal(shape=(3, 2))
print(f"Random 2 tensor is ", random_2)

# Are they equal?
print(random_1, random_2, random_1 == random_2)