import numpy as np
import tensorflow as tf


vocab_size = 10000
embedding_size = 100
batch_size = 2**14

with np.load("sgns_samples.npz") as data:
    words = data["pairs"][:,0]
    contexts = data["pairs"][:,1]
    labels = data["targets"].astype(np.int32)

num_examples = words.shape[0]
num_batches = num_examples // batch_size


words_placeholder = tf.placeholder(words.dtype,name="words_placeholder")
contexts_placeholder = tf.placeholder(contexts.dtype,name="contexts_placeholder")
labels_placeholder = tf.placeholder(labels.dtype,name="labels_placeholder")

dataset = tf.data.Dataset.from_tensor_slices((words_placeholder, contexts_placeholder, labels_placeholder))
dataset = dataset.shuffle(1000).repeat().batch(batch_size,drop_remainder=True)

iterator = dataset.make_initializable_iterator()

u, v, label = iterator.get_next()

word_embeddings = tf.get_variable("word_embeddings",
                                  [vocab_size, embedding_size])

context_embeddings = tf.get_variable("context_embeddings",
                                  [vocab_size, embedding_size])


embedded_word_ids = tf.nn.embedding_lookup(word_embeddings, u)
embedded_context_ids = tf.nn.embedding_lookup(context_embeddings, v)

z = tf.reduce_sum(tf.multiply(embedded_word_ids, embedded_context_ids),axis=1)
z = tf.cast(tf.pow(-1,label),tf.float32) * z
loss = -(tf.log(tf.sigmoid(z)))
optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)
batch_loss = tf.reduce_mean(loss)

init = tf.initializers.global_variables()

epochs = 10

with tf.Session() as sess:
    sess.run(init)
    sess.run(iterator.initializer, feed_dict={words_placeholder: words,
                                              contexts_placeholder: contexts,
                                              labels_placeholder: labels})

    for _ in range(epochs):
        total_loss = 0
        progbar = tf.keras.utils.Progbar(num_batches, 20, stateful_metrics="loss")
        for i in range(num_batches):
            partial_loss, _ = sess.run([batch_loss, optimizer])
            total_loss += partial_loss / float(num_batches)
            progbar.update(i, values=[("loss", partial_loss)])
        print(" total_loss: {}".format(total_loss))
