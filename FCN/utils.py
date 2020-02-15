import tensorflow as tf
import json
import numpy as np

def softmax(feature_map):
    shape = tf.shape(feature_map)
    exp_feature = tf.exp(feature_map)
    sum_exp = tf.reduce_sum(feature_map, axis = 3, keepdims = True)
    return exp_feature/sum_exp

def img_generator(filename):
    #filename = 'images.json'
    with open(filename, 'r') as f:
        data = json.load(f)
        for row in data:
            d = {}
            d['id'] = row['id']
            d['file_name'] = row['file_name']
            d['height'] = row['height']
            d['width'] = row['width']
            yield d

def probaToBinaryMask(proba_softmax):
    dim = proba_softmax.shape
    proba = proba_softmax.reshape((-1, dim[2]))
    categories = np.zeros((dim[0]*dim[1], dim[2]))
    classes = np.argmax(proba, axis = 1)
    for i in range(dim[0]*dim[1]):
        categories[i,classes[i]] = 1
    mask = categories.reshape((dim[0], dim[1], dim[2]))
    return mask