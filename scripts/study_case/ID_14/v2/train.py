import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from v2.model import NetworkLayer3TruncatedNormal


class Train:
    def __init__(self):
        self.CKPT_DIR = './ckpt9'
        self.net = NetworkLayer3TruncatedNormal()

        self.sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=True))
        self.sess.run(tf.global_variables_initializer())

        # print("pwd:", os.getcwd())
        self.data = input_data.read_data_sets('../data/', one_hot=True)

    def train(self):
        saver = tf.train.Saver(max_to_keep=20)
        save_interval = 20000
        batch_size = 64
        train_step = 1100000
        step = 0

        # # merge所有的summary node 的op
        # merged_summary_op = tf.summary.merge_all()
        # # 存放在当前目录下的 log 文件夹中，获得文件句柄
        # merged_writer = tf.summary.FileWriter('./log', self.sess.graph)

        ckpt = tf.train.get_checkpoint_state(self.CKPT_DIR)
        if ckpt and ckpt.model_checkpoint_path:
            print('ckpt.model_checkpoint_path:', ckpt.model_checkpoint_path)
            saver.restore(self.sess, ckpt.model_checkpoint_path)
            step = self.sess.run(self.net.global_step)
            print('Continue from')
            print('       -> Minibatch update : ', step)

        while step < train_step:
            x, label = self.data.train.next_batch(batch_size)
            _, loss = self.sess.run([self.net.train, self.net.loss],
                                    feed_dict={
                                        self.net.x: x,
                                        self.net.label: label
                                    })
            step = self.sess.run(self.net.global_step)
            # if step % 100 == 0:
            #     merged_writer.add_summary(merged_summary, step)
            # if step % 10 == 0:
            #     print('step %5d loss: %.3f' % (step, loss))
            if step % save_interval == 0:
                saver.save(self.sess, self.CKPT_DIR + '/model', global_step=step)
                print('%s/model-%d saved' % (self.CKPT_DIR, step))
            #     self.calculate_accuracy()
            if step % 2000 == 0:
                print('step %d loss: %f' % (step, loss))
            if step % 20000 == 0:
                self.calculate_train_error()
                self.calculate_test_error()

    def calculate_train_error(self):
        train_x = self.data.train.images
        train_label = self.data.train.labels

        error_rate = self.sess.run(self.net.error_rate,
                                   feed_dict={
                                       self.net.x: train_x,
                                       self.net.label: train_label
                                   })
        print("     -> Training set error_rate: %f" % error_rate)

    def calculate_test_error(self):
        test_x = self.data.test.images
        test_label = self.data.test.labels

        error_rate = self.sess.run(self.net.error_rate,
                                   feed_dict={
                                       self.net.x: test_x,
                                       self.net.label: test_label
                                   })
        print("     -> Test set error_rate: %f" % error_rate)


if __name__ == "__main__":
    app = Train()
    app.train()
    app.calculate_test_error()
