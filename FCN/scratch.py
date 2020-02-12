import tensorflow as tf
import os
from itertools import permutations
from utils import softmax

class FCN:
    def __init__(self,input_img,nb_classes,scope='train'):
        self.scope = scope
        self.nb_classes = nb_classes

        with tf.variable_scope(self.scope):
            self.input = input_img
            with tf.variable_scope('permute'):
                a = [1,2,3]
                b = self.input
                self.permute = tf.gather(b,a)
            with tf.variable_scope('conv1'):
                self.conv1_1 = tf.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1,
                                              padding='same', name='conv1_1')(self.permute)
                self.activation1_1 = tf.nn.relu(self.conv1_1)
                self.conv1_2 = tf.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1,
                                              padding='same', name='conv1_2')(self.activation1_1)
                self.activation1_2 = tf.nn.relu(self.conv1_2)
                self.max_pool1 = tf.layers.MaxPooling2D(pool_size = (2,2), strides = 2,
                                                       name = 'maxpool1')(self.activation1_2)
            with tf.variable_scope('conv2'):
                self.conv2_1 = tf.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=1,
                                              padding='same', name='conv2_1')(self.max_pool1)
                self.activation2_1 = tf.nn.relu(self.conv2_1)
                self.conv2_2 = tf.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=1,
                                              padding='same', name='conv2_2')(self.activation2_1)
                self.activation2_2 = tf.nn.relu(self.conv2_2)
                self.max_pool2 = tf.layers.MaxPooling2D(pool_size = (2,2), strides = 2,
                                                       name = 'maxpool2')(self.activation2_2)
            with tf.variable_scope('conv3'):
                self.conv3_1 = tf.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=1,
                                              padding='same', name='conv3_1')(self.max_pool2)
                self.activation3_1 = tf.nn.relu(self.conv3_1)
                self.conv3_2 = tf.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=1,
                                              padding='same', name='conv3_2')(self.activation3_1)
                self.activation3_2 = tf.nn.relu(self.conv3_2)
                self.conv3_3 = tf.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=1,
                                                padding='same', name='conv3_3')(self.activation3_2)
                self.activation3_3 = tf.nn.relu(self.conv3_3)
                self.max_pool3 = tf.layers.MaxPooling2D(pool_size = (2,2), strides = 2,
                                                       name = 'maxpool3')(self.activation3_3)
            with tf.variable_scope('conv4'):
                self.conv4_1 = tf.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=1,
                                              padding='same', name='conv4_1')(self.max_pool3)
                self.activation4_1 = tf.nn.relu(self.conv4_1)
                self.conv4_2 = tf.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=1,
                                              padding='same', name='conv4_2')(self.activation4_1)
                self.activation4_2 = tf.nn.relu(self.conv4_2)
                self.conv4_3 = tf.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=1,
                                                padding='same', name='conv4_3')(self.activation4_2)
                self.activation4_3 = tf.nn.relu(self.conv4_3)
                self.max_pool4 = tf.layers.MaxPooling2D(pool_size = (2,2), strides = 2,
                                                       name = 'maxpool3')(self.activation4_3)
            with tf.variable_scope('score_pool4'):
                self.conv4_pool = tf.layers.Conv2D(filters=self.nb_classes, kernel_size=(1, 1), strides=1,
                                              padding='same', name='conv4_pool')(self.max_pool4)
                self.activation4_pool = tf.nn.relu(self.conv4_pool)
            with tf.variable_scope('conv5'):
                self.conv5_1 = tf.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=1,
                                              padding='same', name='conv5_1')(self.max_pool4)
                self.activation5_1 = tf.nn.relu(self.conv5_1)
                self.conv5_2 = tf.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=1,
                                              padding='same', name='conv5_2')(self.activation5_1)
                self.activation5_2 = tf.nn.relu(self.conv5_2)
                self.conv5_3 = tf.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=1,
                                                padding='same', name='conv5_3')(self.activation5_2)
                self.activation5_3 = tf.nn.relu(self.conv5_3)
                self.max_pool5 = tf.layers.MaxPooling2D(pool_size = (2,2), strides = 2,
                                                       name = 'maxpool3')(self.activation5_3)
            with tf.variable_scope('fc6'):
                self.conv6 = tf.layers.Conv2D(filters=4096, kernel_size=(7, 7), strides=1,
                                              padding='same', name='conv5_1')(self.max_pool5)
                self.activation6 = tf.nn.relu(self.conv6)
            with tf.variable_scope('fc7'):
                self.conv7 = tf.layers.Conv2D(filters=4096, kernel_size=(1, 1), strides=1,
                                              padding='same', name='conv5_1')(self.activation6)
                self.activation7 = tf.nn.relu(self.conv7)
            with tf.variable_scope('score_fr'):
                self.conv_fr = tf.layers.Conv2D(filters=self.nb_classes, kernel_size=(1, 1), strides=1,
                                                padding='same', name='conv_fr')(self.activation7)
                self.activation_fr = tf.nn.relu(self.conv_fr)
                self.deconv_fr = tf.layers.Conv2DTranspose(filters=self.nb_classes, kernel_size=(4, 4), strides=(2, 2),
                                                           padding='valid', name='score2')(self.activation_fr)
                Conv_size = self.activation7.output_shape[2]
                Deconv_size = self.activation_fr.output_shape[2]
                Extra = (Deconv_size - 2 * Conv_size)
                crop_size = (Deconv_size - Extra,Deconv_size - Extra)
                self.cropping = tf.image.crop_and_resize(self.deconv_fr,boxes=(1, 4), box_ind=None, crop_size=crop_size,
                                                         method='bilinear')
            with tf.variable_scope('output'):
                x = tf.constant(self.cropping,self.activation4_pool)
                self.sum = tf.reduce_sum(x,0)
                self.up = tf.layers.Conv2DTranspose(filters=self.nb_classes, kernel_size=(32, 32), strides=(16, 16),
                                                           padding='valid', name='up')(self.sum)
                self.out = tf.image.crop_and_resize(self.deconv_fr, boxes=(1, 4), box_ind=None,
                                                         crop_size=crop_size,
                                                         method='bilinear')