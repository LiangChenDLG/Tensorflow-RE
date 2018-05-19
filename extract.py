from pprint import pprint

import tensorflow as tf
import numpy as np
import time
import datetime
import os
import network
import utils
import sys
import json

FLAGS = tf.app.flags.FLAGS

# use the legacy tensorflow 0.x model checkpoint file provided by THUNLP
# USE_LEGACY = 0


def main(_):
    fout = open('extracted.json', 'w')
    data_out = {}
    # ATTENTION: change pathname before you load your model
    pathname = "./model/kbp/ATT_GRU_model-"
    test_model_id = int(sys.argv[1])

    none_ind = utils.get_none_id('./origin_data/KBP/relation2id.txt')
    print("None index: ", none_ind)

    wordembedding = np.load('./data/KBP/vec.npy')

    test_y = np.load('./data/KBP/testall_y.npy')
    test_word = np.load('./data/KBP/testall_word.npy')
    test_pos1 = np.load('./data/KBP/testall_pos1.npy')
    test_pos2 = np.load('./data/KBP/testall_pos2.npy')
    test_o = np.load('./data/KBP/testall_o.npy')

    print(test_y[0])
    test_settings = network.Settings()
    test_settings.vocab_size = len(wordembedding)
    test_settings.num_classes = len(test_y[0])
    test_settings.big_num = 179

    with tf.Graph().as_default():

        sess = tf.Session()
        with sess.as_default():

            def test_step(word_batch, pos1_batch, pos2_batch, y_batch, o_batch):

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

                feed_dict[mtest.total_shape] = total_shape
                feed_dict[mtest.input_word] = total_word
                feed_dict[mtest.input_pos1] = total_pos1
                feed_dict[mtest.input_pos2] = total_pos2
                feed_dict[mtest.input_y] = y_batch

                loss, accuracy, predictions, word_attention, sentence_attention, prob= sess.run(
                    [mtest.loss, mtest.accuracy, mtest.predictions, mtest.word_attention, mtest.sentence_attention, mtest.prob], feed_dict)

                for i in range(len(word_batch)) :
                    new_out = {}
                    pos1 = 0
                    pos2 = 0
                    for pos_ind in range(len(total_pos1[total_shape[i]])):
                        if total_pos1[total_shape[i]][pos_ind] == 61:
                            pos1 = pos_ind
                            break
                    for pos_ind in range(len(total_pos2[total_shape[i]])):
                        if total_pos2[total_shape[i]][pos_ind] == 61:
                            pos2 = pos_ind
                            break
                    entities_pair = o_batch[i][0][pos1] + ' ' + o_batch[i][0][pos2]
                    print(entities_pair)
                    # p: predictions
                    # a: accuracy
                    # w: word_attention
                    # s: sentence_attention
                    # o: original sentence
                    # t: true relation
                    new_out['p'] = predictions[i].item()
                    new_out['a'] = prob[i][predictions[i]].item()
                    new_out['w'] = word_attention[total_shape[i] : total_shape[i+1]].tolist()
                    new_out['s'] = sentence_attention[i][0].tolist()
                    new_out['o'] = o_batch[i]
                    new_out['t'] = np.argmax(y_batch[i], 0).item()
                    data_out[entities_pair] = new_out
                return predictions, accuracy

            with tf.variable_scope("model"):
                mtest = network.GRU(is_training=False, word_embeddings=wordembedding, settings=test_settings)

            saver = tf.train.Saver()

            # ATTENTION: change the list to the iters you want to test !!
            # testlist = range(9025,14000,25)
            testlist = [test_model_id]
            for model_iter in testlist:
                saver.restore(sess, pathname + str(model_iter))

                time_str = datetime.datetime.now().isoformat()
                print(time_str)

                all_pred = []
                all_true = []
                all_accuracy = []

                for i in range(int(len(test_word) / float(test_settings.big_num))):
                    predictions, accuracy = test_step(test_word[i * test_settings.big_num:(i + 1) * test_settings.big_num],
                                               test_pos1[i * test_settings.big_num:(i + 1) * test_settings.big_num],
                                               test_pos2[i * test_settings.big_num:(i + 1) * test_settings.big_num],
                                               test_y[i * test_settings.big_num:(i + 1) * test_settings.big_num],
                                               test_o[i * test_settings.big_num:(i + 1) * test_settings.big_num])
                    pred = np.array(predictions)
                    all_pred.append(pred)
                    all_true.append(test_y[i * test_settings.big_num:(i + 1) * test_settings.big_num])
                    all_accuracy.append(accuracy)
                all_pred = np.concatenate(all_pred, axis=0)
                all_true = np.concatenate(all_true, axis=0)
                accu = float(np.mean(all_accuracy))
                all_true_inds = np.argmax(all_true, 1)
                precision, recall, f1 = utils.evaluate_rm_neg(all_pred, all_true_inds, none_ind)
                print('Accu = %.4f, F1 = %.4f, recall = %.4f, precision = %.4f)' %
                      (accu,
                       f1,
                       recall,
                       precision))
            fout.write(json.dumps(data_out))


if __name__ == "__main__":
    tf.app.run()
