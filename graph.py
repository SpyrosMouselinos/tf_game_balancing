import tensorflow as tf
import numpy as np


class BalanceModel(object):

    def __init__(self, train_data, validation_data,batch_size=16):
        self.train_data = train_data
        self.validation_data = validation_data
        self.batch_size = batch_size

    """
        Setters
    """

    def load_train_data(self, data):
        self.train_data = data

    def load_validation_data(self, data):
        self.validation_data = data

    def load_batch_size(self, batch_size):
        self.batch_size = batch_size

    """
        Getters
    """

    def get_train_data(self):
        return self.train_data

    def get_validation_data(self):
        return self.validation_data

    def get_batch_size(self):
        return self.batch_size

    """
        Static Methods
    """

    @staticmethod
    def _input_fn_(features, labels, batch_size):
        dataset = tf.data.Dataset.from_tensor_slices((features, labels.reshape(-1, 1)))
        dataset = dataset.batch(batch_size).repeat().make_one_shot_iterator().get_next()
        return dataset

    @staticmethod
    def _eval_fn_(features, labels):
        dataset = tf.data.Dataset.from_tensor_slices((features, labels.reshape(-1, 1)))
        dataset = dataset.batch(np.shape(features)[0]).repeat().make_one_shot_iterator().get_next()
        return dataset

    @staticmethod
    def create_model(fighter1, fighter2, reuse=True, training=True):
        with tf.variable_scope('BNet', reuse=reuse):
            # Branch 1 #
            # ---------------------------------------#
            layer1 = tf.layers.dense(inputs=fighter1, units=16, activation='relu')
            dropout1 = tf.layers.dropout(inputs=layer1, rate=0.3, training=training)

            # Branch 2 #
            layer2 = tf.layers.dense(inputs=fighter2, units=16, activation='relu')
            dropout2 = tf.layers.dropout(inputs=layer2, rate=0.3, training=training)

            # Common Layers #
            merged = tf.concat([dropout1, dropout2], axis=0)
            layer3 = tf.layers.dense(inputs=merged, units=32, activation='relu')
            dropout3 = tf.layers.dropout(inputs=layer3, rate=0.3, training=training)
            layer4 = tf.layers.dense(inputs=dropout3, units=32, activation='relu')
            dropout4 = tf.layers.dropout(inputs=layer4, rate=0.3, training=training)
            final_out = tf.layers.dense(inputs=dropout4, units=1, name='model_out')
            # ---------------------------------------#
        return final_out

    """
        Private Variables
    """
    __x, __y = _input_fn_(get_train_data()[0], get_train_data()[1], get_batch_size())
    __x_eval, __y_eval = _eval_fn_(get_validation_data()[0], get_validation_data()[1])


    final_out = create_model()
    final_out_eval = create_model()



    """
        Losses
    """

    train_loss1 = tf.reduce_mean(tf.square(tf.cast(__y, tf.float32) - final_out))
    train_loss2 = tf.reduce_mean(tf.square(tf.cast(final_out_nhf, tf.float32) - final_out * 0.1))

    val_loss1 = tf.reduce_mean(tf.square(tf.cast(__y_eval, tf.float32) - final_out_eval))
    val_loss2 = tf.reduce_mean(tf.square(tf.cast(final_out_nhf_eval, tf.float32) - final_out_eval * 0.1))

    train_loss = train_loss1 + 0.5 * train_loss2
    val_loss = val_loss1 + 0.5 * val_loss2

    """
        Metrics
    """
    val_metric = tf.contrib.metrics.streaming_pearson_correlation(tf.cast(__y_eval, tf.float32), final_out_eval)
    val_mae = tf.metrics.mean_absolute_error(tf.cast(__y_eval, tf.float32), final_out_eval)
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(train_loss)
    saver = tf.train.Saver()

    epochs = 500
    stop_limit = 40
    val_count = 0
    validate_every = 35
    min_val_loss = 1e10
    step = 0
    epoch = get_train_data()[0].shape[0] // get_batch_size()
    val_size = get_validation_data()[0].shape[0]

    epoch_counter = 0
    batch_losses = []
    saver = tf.train.Saver()
    #
    # def train(self):
    #     with tf.Session() as sess:
    #         sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
    #         print('Initialized')
    #         while val_count <= stop_limit and epoch_counter < num_epochs:
    #             _, train_batch_loss = sess.run([train_op, train_loss])
    #             if step % validate_every == 0:
    #                 print ('Step: ' + str(step) + ', Train Batch Loss ' + str(train_batch_loss))
    #             batch_losses.append(train_batch_loss)
    #             if val_size > 0 and step > 0 and step % epoch == 0:
    #                 train_epoch_loss = np.mean(batch_losses)
    #                 print ('*** Epoch: ' + str(epoch_counter) + ', Train Epoch Loss: ' + str(train_epoch_loss) + ' ***')
    #                 batch_losses = []
    #                 val_loss_value, val_pcc, mae = sess.run([val_loss, val_metric, val_mae])
    #                 epoch_counter += 1
    #                 if val_loss_value > min_val_loss:
    #                     val_count += 1
    #                 else:
    #                     min_val_loss = val_loss_value
    #                     val_count = 0
    #
    #                     #########################
    #
    #                     save_path = saver.save(sess, "/home/ubuntu/lstm_session/model.ckpt")
    #
    #                     train_results_path = "/home/ubuntu/lstm_session/train_details"
    #                     if not os.path.isdir(train_results_path):
    #                         os.makedirs(train_results_path, 0777)
    #                     name = "train_results.json"
    #                     filename = os.path.join(train_results_path, name)
    #                     experiments = {'epoch': int(epoch_counter),
    #                                    'val loss': float(val_loss_value),
    #                                    'train loss': float(train_epoch_loss),
    #                                    'mae': float(mae[1]),
    #                                    'val pcc': float(val_pcc[1])}
    #
    #                     with open(filename, 'w') as f:
    #                         json.dump(experiments, f, sort_keys=True)
    #
    #                 patience = stop_limit - val_count
    #                 print ('Step: ' + str(step) + ', Validation PCC: ' + str(val_pcc) + ', Validation Loss: ' + str(
    #                     val_loss_value) + ', Validation Mae : ' + str(mae[1]) + ', Best Loss : ' + str(
    #                     min_val_loss) + ', Epoch: ' + str(epoch_counter) + ', Patience Left: ' + str(patience))
    #             step += 1
    #
    #
    #
    #
    #     meta_path = "/home/ubuntu/lstm_session/" + [f for f in os.listdir('/home/ubuntu/lstm_session/') if '.meta' in f][0]
    #


# TODO:(Spyros) Finish THIS
# # ### ORIGINAL PREDICTIONS
#
#
# with tf.Session() as sess:
#     sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
#     new_saver = tf.train.import_meta_graph(meta_path)
#     new_saver.restore(sess, tf.train.latest_checkpoint('/home/ubuntu/lstm_session/'))
#     predictions = sess.run(final_out_test)
#
# # plots
# sns.set(style="darkgrid")
# plt.figure(figsize=(15, 8))
# plt.title("Consumption/Day for all")
# plt.xlabel("Days")
# plt.ylabel("foVolConsumption")
# plt.plot(range(0, len(full_y)), full_y, color='b', label='Real')
# plt.plot(range(0, len(np.squeeze(predictions))), np.squeeze(predictions), color='r', label='Predicted')
# plt.legend(loc='upper left', fontsize='small')
# plt.savefig("LSTM-one-for-all.png")
# plt.show()
#
# # ### NHF PREDICTIONS
#
#
# with tf.Session() as sess:
#     sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
#     new_saver = tf.train.import_meta_graph(meta_path)
#     new_saver.restore(sess, tf.train.latest_checkpoint('/home/ubuntu/lstm_session/'))
#     predictions_nhf = sess.run(final_out_nhf_test)
#
# sns.set(style="darkgrid")
# plt.figure(figsize=(15, 8))
# plt.title("Consumption/Day for all")
# plt.xlabel("Days")
# plt.ylabel("foVolConsumption")
# plt.plot(range(0, len(full_y)), full_y, color='b', label='Real')
# plt.plot(range(0, len(np.squeeze(predictions_nhf))), np.squeeze(predictions_nhf), color='r', label='Predicted')
# plt.legend(loc='upper left', fontsize='small')
# plt.savefig("LSTM-one-for-all.png")
# plt.show()
#
# # ### FOVOL LOSS
#
# sns.set(style="darkgrid")
# plt.figure(figsize=(15, 8))
# plt.title("Consumption/Day for all")
# plt.xlabel("Days")
# plt.ylabel("foVolConsumption")
# plt.plot(range(0, len(full_y)), full_y, color='b', label='Real')
# w_e = np.squeeze(predictions) - np.squeeze(predictions_nhf)
# plt.plot(range(0, len(full_y)), w_e, color='r', label='NHF')
# plt.legend(loc='upper left', fontsize='small')
# plt.savefig("LSTM-one-for-all.png")
# plt.show()
#
# predictions = np.squeeze(predictions)
# import numpy as np
#
#
# def mean_absolute_percentage_error(y_true, y_pred):
#     x = zip(y_true, y_pred)
#     x_clean = [f for f in x if f[0] != 0]
#     d = zip(*x_clean)
#     y_true = d[0]
#     y_pred = d[1]
#     y_true, y_pred = np.array(y_true), np.array(y_pred)
#     return np.mean(np.abs(((y_true - y_pred) / (y_true))) * 100)
#
#
# x = zip(test_y, predictions)
# x_clean = [f for f in x if f[0] != 0]
# d = zip(*x_clean)
# y_true = d[0]
# y_pred = d[1]
# y_true, y_pred = np.array(y_true), np.array(y_pred)
# gg_list = []
# for index, _ in enumerate(y_true):
#     gg = np.abs((y_true[index] - y_pred[index]) / (y_true[index])) * 100
#     gg_list.append(gg)
# print "before:" + str(len(gg_list))
# gg_list = [f for f in gg_list if f <= 150]
# print "after:" + str(len(gg_list))
# df = pd.DataFrame({'x': gg_list})
# ax = df.plot.kde()
# line = ax.lines[0]
# y_data = line.get_ydata()
# x_data = line.get_xdata()
# pmae = x_data[np.argmax(y_data)]
# print pmae