import tensorflow as tf

#Create shuffle tensor
not_shuffled = tf.constant([10,7],
                          [3,4],
                          [2,5])
print(not_shuffled.dim())
shuffled = tf.random.shuffle(not_shuffled)
print(shuffled)