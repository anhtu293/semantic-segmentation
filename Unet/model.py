import tensorflow as tf
import os

class Unet:
    def __init__(self, input_dim, batch_size, learning_rate, input_img, label, scope='train', loss = 'entropy'):
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.scope = scope
        self.loss = loss
        self.label = label

        with tf.variable_scope(self.scope):
            self.input = input_img
            with tf.variable_scope('down1'):
                self.conv1 = tf.layers.Conv2D(filters = 64, kernel_size = (3,3), strides = 1, name = 'conv1')(self.input)
                self.activation1 = tf.nn.relu(self.conv1)
                self.conv2 = tf.layers.Conv2D(inputs = self.activation1, filters = 64, kernel_size = (3,3), strides = 1,
                                              name = 'conv2')(self.activation1)
                self.activation2 = tf.nn.relu(self.conv2)
                self.maxpool1 = tf.layers.MaxPooling2D(pool_size = (2,2), strides = 2,
                                                       name = 'maxpool1')(self.activation2)
            with tf.variable_scope('down2'):
                self.conv3 = tf.layers.Conv2D(filters = 128, kernel_siez = (3,3), strides = 1,
                                              name = 'conv3')(self.maxpool1)
                self.activation3 = tf.nn.relu(self.conv3)
                self.conv4 = tf.layers.Conv2D(filters = 128, kernel_size = (3,3), strides = 1,
                                              name = 'conv4')(self.activation3)
                self.activation4 = tf.nn.relu(self.conv4)
                self.maxpool2 = tf.layers.MaxPooling2D(pool_size = (2,2), strides = 2,
                                                       name = 'maxxpool2')(self.activation4)
            with tf.variable_scope('donw3'):
                self.conv5 = tf.layers.Conv2D(filters = 256, kernel_size = (3,3), strides = 1,
                                              name = 'conv5')(self.maxpool2)
                self.activation5 = tf.nn.relu(self.conv5)
                self.conv6 = tf.layers.Conv2D(filters = 256, kernel_size = (3,3), strides = 1,
                                              name = 'conv6')(self.activation5)
                self.activation6 = tf.nn.relu(self.conv6)
                self.maxpool3 = tf.layers.MaxPooling2D(pool_size = (2,2), strides = 2,
                                                       name = 'maxpool3')(self.activation6)
            with tf.variable_scope('down4'):
                self.conv7 = tf.layers.Conv2D(filters = 512, kernel_size = (3,3), strides = 1,
                                              name = 'conv7')(self.maxpool3)
                self.activation7 = tf.nn.relu(self.conv7)
                self.conv8 = tf.layers.Conv2D(filters = 512, kernel_size = (3,3), strides = 1,
                                              name = 'conv8')(self.activation7)
                self.activiation8 = tf.nn.relu(self.conv8)
                self.maxpool4 = tf.layers.MaxPooling2D(pool_size = (2,2), strides = 2,
                                                       name = 'maxpool4')(self.activation8)

            with tf.variable_scope('down5'):
                self.conv9 = tf.layers.Conv2D(filters = 1024, kernel_size = (1,1), strides = 1,
                                              name = 'conv8')(self.maxpool4)
                self.activation9 = tf.nn.relu(self.conv9)
                self.conv10 = tf.layers.Conv2D(filters = 1024, kernel_size = (1,1), strides = 1,
                                               name = 'conv9')(self.activation9)
                self.activation10 = tf.nn.relu(self.conv10)
                self.upconv1 = tf.layers.Conv2DTranspose(filters = 512, kernel_size = (2,2), strides = (2,2),
                                                         name = 'upconv1')(self.activation10)
                self.upsampling1 = tf.concat([self.activation8, self.upconv1])

            with tf.variable_scope('up1'):
                self.conv11 = tf.layers.Conv2D(filters = 512, kernel_size = (3,3), strides = 1,
                                               name = 'conv11')(self.upsampling1)
                self.activation11 = tf.nn.relu(self.conv11)
                self.conv12 = tf.layers.Conv2D(filters = 512, kernel_size = (3,3), strides = 1,
                                               name = 'conv12')(self.activation11)
                self.activation12 = tf.nn.relu(self.conv12)
                self.upconv2 = tf.layers.Conv2DTranspose(filters = 256, kernel_size = (2,2), strides = (2,2),
                                                         name = 'upconv2')(self.activation12)
                self.upsampling2 = tf.concat([self.activation6, self.upconv2])

            with tf.variable_scope('up2'):
                self.conv13 = tf.layers.Conv2D(filters = 256, kernel_size = (3,3), strides = 1,
                                               name = 'conv13')(self.upsampling2)
                self.activation13 = tf.nn.relu(self.conv13)
                self.conv14 = tf.layers.Conv2D(filters = 256, kernel_size = (3,3), strides = 1,
                                               name = 'conv13')(self.activation13)
                self.activation14 = tf.nn.relu(self.conv14)
                self.upconv3 = tf.layers.Conv2DTranspose(filters = 128, kernel_size = (2,2), strides = (2,2),
                                                         name = 'upconv3')(self.activation14)
                self.upsampling3 = tf.concat([self.activation4, self.upconv3])

            with tf.variable_scope('up3'):
                self.conv15 = tf.layers.Conv2D(filters = 128, kernel_size = (3,3), strides = 1,
                                               name = 'conv15')(self.upsampling3)
                self.activation15 = tf.nn.relu(self.conv15)
                self.conv16 = tf.layers.Conv2D(filters = 128, kernel_size = (3,3), strides = 1,
                                               name = 'conv16')(sell.activation15)
                self.activation16 = tf.nn.relu(self.conv16)
                self.upconv4 = tf.layers.Conv2DTranspose(filters = 64, kernel_size = (2,2), strides = (2,2),
                                                         name = 'upconv4')(self.activation16)
                self.upsampling4 = tf.concat([self.activation2, self.upconv4])

            with tf.variable_scope('up4'):
                self.conv17 = tf.layers.Conv2D(filters = 64, kernel_size = (3,3), strides = 1,
                                               name = 'conv17')(self.upsampling4)
                self.activation17 = tf.nn.relu(self.conv17)
                self.conv18 = tf.layers.Conv2D(filters = 64, kernel_size = (3,3), strides = 1,
                                               name = 'conv18')(self.activation17)
                self.activation18 = tf.nn.relu(self.conv18)

            with tf.variable_scope('output'):
                self.conv19 = tf.layers.Conv2D(filters = 10, kernel_size = (1,1), strides = 1,
                                               name = 'output')(self.activation18)
                self.output = tf.nn.softmax(self.conv19)


    """def train_op(self):
        if self.loss == 'entropy':
            with tf.variable_scope('train'):
                with tf.variable_scope('loss'):
                    self.optimizer = tf.train.MomentumOptimizer(learning_rate = self.learning_rate, momentum = 0.99)
                    self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = self.label,
                                                                                     logits = self.output))
                    self.train = self.optimizer.minimize(self.loss)
        elif self.loss == 'dice':
            with tf.variable_scope('train'):
                with tf.variable_scope('loss'):
                    self.optimizer = tf.train.MomentumOptimizer(learning_rate = self.learning_rate, momentum = 0.99)
"""