"""
Wrap the sgns graph from sgns.py into a tf.Estimator
Adapted from: https://github.com/tensorflow/models/blob/master/samples/cookbook/regression/custom_regression.py
"""
import parser

import numpy as np
import tensorflow as tf
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=16384, type=int, help='batch size')
parser.add_argument('--embedding_size', default=100, type=int, help='Dimensions of embedding')
parser.add_argument('--vocab_size', default=10000, type=int, help='Size of vocab')
parser.add_argument('--train_steps', default=5000, type=int,
                    help='number of training steps')
parser.add_argument('--learning_rate', default=0.001, type=float,
                    help='learning rate')


def sgns_fn(features, labels, mode, params):
    """
    Model function implementing Skip Gram with Negative Sampling
    """

    # u = tf.get_variable("words_embedding_input",(None, params['embedding_size']))
    # v = tf.get_variable("contexts_embedding_input",(None, params['embedding_size']))

    e_sz = params['embedding_size']
    v_sz = params['vocab_size']

    # u = tf.placeholder(tf.int64,(None,))
    # v = tf.placeholder(tf.int64,(None,))

    #
    # uv_mapping = { params['feature_columns'][0]:u,
    #                params['feature_columns'][1]:v
    #                }

    u = tf.feature_column.input_layer(features,[params['feature_columns'][0] ])
    v = tf.feature_column.input_layer(features, [params['feature_columns'][1] ])
    # u = features["words"]
    # v = features["contexts"]

    z = tf.reduce_sum(tf.multiply(u, v), axis=1)

    # word_embeddings = tf.get_variable("word_embeddings",
    #                                   [v_sz, e_sz])
    #
    # context_embeddings = tf.get_variable("context_embeddings",
    #                                      [v_sz, e_sz])
    #
    # u_embed = tf.nn.embedding_lookup(word_embeddings, u)
    # v_embed = tf.nn.embedding_lookup(context_embeddings, v)


    # ones_ =tf.ones((2, params['embedding_size']))
    #
    # x = tf.feature_column.input_layer(features, params['feature_columns'])
    # z = tf.reduce_sum(tf.matmul(x,ones_),axis=1)

    # z = tf.reduce_sum(tf.multiply(u_embed, v_embed), axis=1)

    predictions = tf.sigmoid(z)
    # z = tf.cast(tf.pow(-1,label),tf.float32) * z
    # loss = -(tf.log(tf.sigmoid(z)))

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.EstimatorSpec(
            mode=mode,
            predicts={"in_context": predictions}
        )

    total_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,logits=z)
    average_loss = tf.reduce_mean(total_loss)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = params.get("optimizer", tf.train.AdamOptimizer)
        optimizer = optimizer(params.get("learning_rate", 1e-3))
        train_op = optimizer.minimize(
            loss=average_loss, global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(
            mode=mode, loss=average_loss, train_op=train_op)

    assert mode == tf.estimator.ModeKeys.EVAL

    # Calculate predictions
    print(labels)
    print(predictions)

    return tf.estimator.EstimatorSpec(
        mode=mode,
        # Report sum of error for compatibility with pre-made estimators
        loss=average_loss)


def make_dataset(batch_sz, w,c , y=None, shuffle=False, shuffle_buffer_size=1000):
    """Create a slice Dataset from a pandas DataFrame and labels"""
    features = {'words': w, 'contexts': c}

    def input_fn():
        if y is not None:
            dataset = tf.data.Dataset.from_tensor_slices((features , y))
        else:
            dataset = tf.data.Dataset.from_tensor_slices((features, ))
        if shuffle:
            dataset = dataset.shuffle(shuffle_buffer_size).batch(batch_sz, drop_remainder=True).repeat()
        else:
            dataset = dataset.batch(batch_sz, drop_remainder=True)
        return dataset.make_one_shot_iterator().get_next()

    return input_fn


def main(argv):
    args = parser.parse_args(argv[1:])


    with np.load("sgns_samples.npz") as data:
        words = data["pairs"][:, 0]
        contexts = data["pairs"][:, 1]
        # pairs = data["pairs"]
        labels = data["targets"].astype(np.float32)

    train_input_fn = make_dataset(args.batch_size, words, contexts, labels,shuffle=True)

    # words_feat = tf.feature_column.numeric_column(key='words', dtype=tf.int64)
    # context_feat = tf.feature_column.numeric_column(key='contexts', dtype=tf.int64)
    #
    # feature_columns = [words_feat, context_feat]

    # words_feat = tf.feature_column.categorical_column_with_vocabulary_file(
    #     key='words',
    #     vocabulary_file="vocab.10k.txt",
    #     vocabulary_size=args.vocab_size,
    #     default_value=2,
    #     dtype=tf.string
    # )
    #
    # context_feat = tf.feature_column.categorical_column_with_vocabulary_file(
    #     key='contexts',
    #     vocabulary_file="vocab.10k.txt",
    #     vocabulary_size=args.vocab_size,
    #     default_value=2,
    #     dtype=tf.string
    # )

    words_feat = tf.feature_column.categorical_column_with_identity(key="words",
                                                                    num_buckets=args.vocab_size)
    context_feat = tf.feature_column.categorical_column_with_identity(key="contexts",
                                                                    num_buckets=args.vocab_size)

    words_embed = tf.feature_column.embedding_column(words_feat,
                                                     dimension=args.embedding_size,
                                                     trainable=True)

    context_embed = tf.feature_column.embedding_column(context_feat,
                                                     dimension=args.embedding_size,
                                                     trainable=True)

    feature_columns = [
        words_embed,
        context_embed
    ]

    p = {
        "feature_columns" : feature_columns,
        "learning_rate" : args.learning_rate,
        "optimizer" : tf.train.AdamOptimizer,
        "embedding_size" : args.embedding_size,
        "vocab_size" : args.vocab_size
    }

    config = tf.estimator.RunConfig(
        model_dir='./estimator_model',
        tf_random_seed=42
    )
    model = tf.estimator.Estimator(
        model_fn = sgns_fn,
        params = p,
        config=config
    )

    model.train(input_fn=train_input_fn, steps=args.train_steps)


if __name__ == "__main__":
    # The Estimator periodically generates "INFO" logs; make these logs visible.
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)
