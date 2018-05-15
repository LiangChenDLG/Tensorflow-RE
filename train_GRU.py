import network
import tensorflow as tf
import numpy as np
import time
import datetime
import os
from tensorflow.contrib.tensorboard.plugins import projector
from sklearn.metrics import average_precision_score

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('summary_dir', '.', 'path to store summary')

# change the name to who you want to send
# tf.app.flags.DEFINE_string('wechat_name', 'Tang-24-0325','the user you want to send info to')
tf.app.flags.DEFINE_string('wechat_name', 'filehelper', 'the user you want to send info to')

def main(_):
    # the path to save models
    save_path = './model/'

    print('reading wordembedding')
    wordembedding = np.load('./data/vec.npy')

    print('reading training data')
    train_y = np.load('./data/small_y.npy')
    train_word = np.load('./data/small_word.npy')
    train_pos1 = np.load('./data/small_pos1.npy')
    train_pos2 = np.load('./data/small_pos2.npy')
    train_pos2 = np.load('./data/small_pos2.npy')

    settings = network.Settings()
    settings.vocab_size = len(wordembedding)
    settings.num_classes = len(train_y[0])

    # ATTENTION: change pathname before you load your model
    pathname = "./model/ATT_GRU_model-"

    test_settings = network.Settings()
    test_settings.vocab_size = 114044
    test_settings.num_classes = 53
    test_settings.big_num = 50
    big_num_test = test_settings.big_num

    big_num = settings.big_num

    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():
            initializer = tf.contrib.layers.xavier_initializer()
            with tf.variable_scope("model", reuse=None, initializer=initializer):
                m = network.GRU(is_training=True, word_embeddings=wordembedding, settings=settings)
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(0.001)

            # train_op=optimizer.minimize(m.total_loss,global_step=global_step)
            train_op = optimizer.minimize(m.final_loss, global_step=global_step)
            sess.run(tf.initialize_all_variables())
            saver = tf.train.Saver(max_to_keep=None)

            merged_summary = tf.summary.merge_all()
            # merged_summary = tf.merge_all_summaries()
            summary_writer = tf.summary.FileWriter(FLAGS.summary_dir + '/train_loss', sess.graph)

            # summary for embedding
            # it's not available in tf 0.11,(because there is no embedding panel in 0.11's tensorboard) so I delete it =.=
            # you can try it on 0.12 or higher versions but maybe you should change some function name at first.

            # summary_embed_writer = tf.train.SummaryWriter('./model',sess.graph)
            # config = projector.ProjectorConfig()
            # embedding_conf = config.embedding.add()
            # embedding_conf.tensor_name = 'word_embedding'
            # embedding_conf.metadata_path = './data/metadata.tsv'
            # projector.visualize_embeddings(summary_embed_writer, config)

            def train_step(word_batch, pos1_batch, pos2_batch, y_batch, big_num):

                feed_dict = {}
                total_shape = []
                total_num = 0
                total_word = []
                total_pos1 = []
                total_pos2 = []
                for i in range(len(word_batch)):
                    total_shape.append(total_num)
                    total_num += len(word_batch[i])
                    for word in word_batch[i]:
                        total_word.append(word)
                    for pos1 in pos1_batch[i]:
                        total_pos1.append(pos1)
                    for pos2 in pos2_batch[i]:
                        total_pos2.append(pos2)
                total_shape.append(total_num)
                total_shape = np.array(total_shape)
                total_word = np.array(total_word)
                total_pos1 = np.array(total_pos1)
                total_pos2 = np.array(total_pos2)

                feed_dict[m.total_shape] = total_shape
                feed_dict[m.input_word] = total_word
                feed_dict[m.input_pos1] = total_pos1
                feed_dict[m.input_pos2] = total_pos2
                feed_dict[m.input_y] = y_batch
                feed_dict[m.keep_prob] = 0.5

                temp, step, loss, accuracy, summary, l2_loss, final_loss, debug_output_forward = sess.run(
                    [train_op, global_step, m.total_loss, m.accuracy, merged_summary, m.l2_loss, m.final_loss, m.output_forward],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                accuracy = np.reshape(np.array(accuracy), (big_num))
                acc = np.mean(accuracy)
                summary_writer.add_summary(summary, step)

                if step % 50 == 0:
                    tempstr = "{}: step {}, softmax_loss {:g}, acc {:g}".format(time_str, step, loss, acc)
                    print(debug_output_forward[0][1])
                    print('\n')
                    print(tempstr + '\n')
                    label_not_NA_num = 0
                    for i in y_batch :
                        if i[0] != 1 :
                            label_not_NA_num += 1
                    print('not NA num : '+ str(label_not_NA_num) + '\n') 
                    

                if step >= 10000:
                    print('finished 10000 trains')

            def test_step(word_batch, pos1_batch, pos2_batch, y_batch):
                feed_dict = {}
                total_shape = []
                total_num = 0
                total_word = []
                total_pos1 = []
                total_pos2 = []
                for i in range(len(word_batch)):
                    total_shape.append(total_num)
                    total_num += len(word_batch[i])
                    for word in word_batch[i]:
                        total_word.append(word)
                    for pos1 in pos1_batch[i]:
                        total_pos1.append(pos1)
                    for pos2 in pos2_batch[i]:
                        total_pos2.append(pos2)
                total_shape.append(total_num)
                total_shape = np.array(total_shape)
                total_word = np.array(total_word)
                total_pos1 = np.array(total_pos1)
                total_pos2 = np.array(total_pos2)
                feed_dict[m.total_shape] = total_shape
                feed_dict[m.input_word] = total_word
                feed_dict[m.input_pos1] = total_pos1
                feed_dict[m.input_pos2] = total_pos2
                feed_dict[m.input_y] = y_batch
                feed_dict[m.keep_prob] = 1
                loss, accuracy, prob = sess.run(
                    [m.loss, m.accuracy, m.prob], feed_dict)
                return prob, accuracy

            # evaluate p@n
            def eval_pn(test_y, test_word, test_pos1, test_pos2, test_settings):
                allprob = []
                acc = []
                for i in range(int(len(test_word) / float(test_settings.big_num))):
                    prob, accuracy = test_step(
                        test_word[i * test_settings.big_num:(i + 1) * test_settings.big_num],
                        test_pos1[i * test_settings.big_num:(i + 1) * test_settings.big_num],
                        test_pos2[i * test_settings.big_num:(i + 1) * test_settings.big_num],
                        test_y[i * test_settings.big_num:(i + 1) * test_settings.big_num])
                    acc.append(np.mean(np.reshape(np.array(accuracy), (test_settings.big_num))))
                    prob = np.reshape(np.array(prob), (test_settings.big_num, test_settings.num_classes))
                    for single_prob in prob:
                        allprob.append(single_prob[1:])
                allprob = np.reshape(np.array(allprob), (-1))
                eval_y = []
                for i in test_y:
                    eval_y.append(i[1:])
                allans = np.reshape(eval_y, (-1))
                order = np.argsort(-allprob)
                print('mytest')
                print(allprob)
                print(allans)
                print('P@100:')
                top100 = order[:100]
                correct_num_100 = 0.0
                for i in top100:
                    if allans[i] == 1:
                        correct_num_100 += 1.0
                print(correct_num_100 / 100)
                print('P@200:')
                top200 = order[:200]
                correct_num_200 = 0.0
                for i in top200:
                    if allans[i] == 1:
                        correct_num_200 += 1.0
                print(correct_num_200 / 200)
                print('P@300:')
                top300 = order[:300]
                correct_num_300 = 0.0
                for i in top300:
                    if allans[i] == 1:
                        correct_num_300 += 1.0
                print(correct_num_300 / 300)

        for one_epoch in range(settings.num_epochs):

            temp_order = list(range(len(train_word)))
            np.random.shuffle(temp_order)
            for i in range(int(len(temp_order) / float(settings.big_num))):

                temp_word = []
                temp_pos1 = []
                temp_pos2 = []
                temp_y = []

                temp_input = temp_order[i * settings.big_num:(i + 1) * settings.big_num]
                for k in temp_input:
                    temp_word.append(train_word[k])
                    temp_pos1.append(train_pos1[k])
                    temp_pos2.append(train_pos2[k])
                    temp_y.append(train_y[k])
                num = 0
                for single_word in temp_word:
                    num += len(single_word)

                if num > 1500:
                    print('out of range')
                    continue

                temp_word = np.array(temp_word)
                temp_pos1 = np.array(temp_pos1)
                temp_pos2 = np.array(temp_pos2)
                temp_y = np.array(temp_y)

                train_step(temp_word, temp_pos1, temp_pos2, temp_y, settings.big_num)

                current_step = tf.train.global_step(sess, global_step)
                if current_step > 9000 and current_step % 500 == 0:
                    # if current_step == 50:
                    print('saving model')
                    path = saver.save(sess, save_path + 'ATT_GRU_model', global_step=current_step)
                    tempstr = 'have saved model to ' + path
                    print(tempstr)
                if current_step >= 10000:
                    break

        # TEST
        for i in [0]:

            print('Evaluating P@N for one')

            test_y = np.load('./data/pone_test_y.npy')
            test_word = np.load('./data/pone_test_word.npy')
            test_pos1 = np.load('./data/pone_test_pos1.npy')
            test_pos2 = np.load('./data/pone_test_pos2.npy')
            eval_pn(test_y, test_word, test_pos1, test_pos2, test_settings)

            print('Evaluating P@N for two')
            test_y = np.load('./data/ptwo_test_y.npy')
            test_word = np.load('./data/ptwo_test_word.npy')
            test_pos1 = np.load('./data/ptwo_test_pos1.npy')
            test_pos2 = np.load('./data/ptwo_test_pos2.npy')
            eval_pn(test_y, test_word, test_pos1, test_pos2, test_settings)

            print('Evaluating P@N for all')
            test_y = np.load('./data/pall_test_y.npy')
            test_word = np.load('./data/pall_test_word.npy')
            test_pos1 = np.load('./data/pall_test_pos1.npy')
            test_pos2 = np.load('./data/pall_test_pos2.npy')
            eval_pn(test_y, test_word, test_pos1, test_pos2, test_settings)

            time_str = datetime.datetime.now().isoformat()
            print(time_str)
            print('Evaluating all test data and save data for PR curve')

            test_y = np.load('./data/testall_y.npy')
            test_word = np.load('./data/testall_word.npy')
            test_pos1 = np.load('./data/testall_pos1.npy')
            test_pos2 = np.load('./data/testall_pos2.npy')
            allprob = []
            acc = []
            for i in range(int(len(test_word) / float(test_settings.big_num))):
                prob, accuracy = test_step(test_word[i * test_settings.big_num:(i + 1) * test_settings.big_num],
                                           test_pos1[i * test_settings.big_num:(i + 1) * test_settings.big_num],
                                           test_pos2[i * test_settings.big_num:(i + 1) * test_settings.big_num],
                                           test_y[i * test_settings.big_num:(i + 1) * test_settings.big_num])
                acc.append(np.mean(np.reshape(np.array(accuracy), (test_settings.big_num))))
                prob = np.reshape(np.array(prob), (test_settings.big_num, test_settings.num_classes))
                for single_prob in prob:
                    allprob.append(single_prob[1:])
            allprob = np.reshape(np.array(allprob), (-1))
            order = np.argsort(-allprob)

            print('saving all test result...')
            current_step = 'TEST'

            # ATTENTION: change the save path before you save your result !!
            np.save('./out/allprob_iter_' + str(current_step) + '.npy', allprob)
            allans = np.load('./data/allans.npy')

            # caculate the pr curve area
            average_precision = average_precision_score(allans, allprob)
            print('PR curve area:' + str(average_precision))

            time_str = datetime.datetime.now().isoformat()
            print(time_str)
            print('P@N for all test data:')
            print('P@100:')
            top100 = order[:100]
            correct_num_100 = 0.0
            for i in top100:
                if allans[i] == 1:
                    correct_num_100 += 1.0
            print(correct_num_100 / 100)

            print('P@200:')
            top200 = order[:200]
            correct_num_200 = 0.0
            for i in top200:
                if allans[i] == 1:
                    correct_num_200 += 1.0
            print(correct_num_200 / 200)

            print('P@300:')
            top300 = order[:300]
            correct_num_300 = 0.0
            for i in top300:
                if allans[i] == 1:
                    correct_num_300 += 1.0
            print(correct_num_300 / 300)


if __name__ == "__main__":
    tf.app.run()
