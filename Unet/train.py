import tensorflow as tf
import numpy as np
from model import Unet
import sys
import skimage.io as io
import cv2
from cv2 import resize
sys.path.append("../tools")
from pycocotools.coco import COCO
from pycocotools import mask
from utils import img_generator, probaToBinaryMask
import argparse
from tqdm import tqdm

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nb_classes", default = "10", type = int, help = "number of categories")
    parser.add_argument("--epochs", default = "10", type = int, help = "number of epochs")
    parser.add_argument("--batch_size", default = "1", type = int, help = "size of miniibatch")
    parser.add_argument("--learning_rate", default = "0.001", type = float, help = "learning_rate")
    parser.add_argument("--loss", default = "crossentropy", help = "type of loss function : crossentropy or dice")
    args = parser.parse_args()
    return args
class Trainer:
    def __init__(self, args):
        #save args
        self.args = args
        #init coco utils
        self.coco = COCO("../annotations/instances_train2014.json")
        #init tensorflow session
        tf.reset_default_graph()
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config = config)
        #init model
        self.input_img = tf.placeholder(tf.float32, shape = (None,None,None,3))
        self.label = tf.placeholder(tf.float32, shape = (None,None,None,args.nb_classes))
        self.model = Unet(input_img = self.input_img, nb_classes = args.nb_classes)
        #define loss : Cross Entropy and Dice
        with tf.variable_scope('optimization'):
            with tf.variable_scope('loss'):
                if args.loss == 'crossentropy':
                    """logits = tf.reshape(self.model.output_log, [-1, args.nb_classes])
                    labels = tf.reshape(self.label, [-1, args.nb_classes])"""
                    self.loss = -tf.reduce_mean(tf.multiply(self.label,tf.log(self.model.output_proba)))
                elif args.loss == "dice":
                    labels = self.label
                    proba = self.model.output_proba
                    intersection = tf.reduce_sum(proba*labels)
                    union = tf.reduce_sum(proba + labels)
                    self.loss = -intersection/union
            #Optimizer
            self.optimizer = tf.train.MomentumOptimizer(learning_rate=args.learning_rate, momentum=0.99)
            self.train_op = self.optimizer.minimize(self.loss)
        #summary file for tensorboard
        self.tf_train_loss = tf.Variable(0.0, trainable = False, name = 'Train_Loss')
        self.tf_train_loss_summary = tf.summary.scalar("Loss", self.tf_train_loss)
        self.tf_train_accuracy = tf.Variable(0.0, trainable = False, name = 'Train_Accuracy')
        self.tf_train_accuracy_summary = tf.summary.scalar("Train Accuracy", self.tf_train_accuracy)
        self.tf_train_dice = tf.Variable(0.0, trainable=False, name="Train_Dice_Coef")
        self.tf_train_dice_summary = tf.summary.scalar("Train Dice Coef", self.tf_train_dice)
        self.tf_eval_accuracy = tf.Variable(0.0, trainable = False, name = 'Eval_accuracy')
        self.tf_eval_accuracy_summary = tf.summary.scalar('Evaluation Accuracy', self.tf_eval_accuracy)
        self.tf_eval_dice = tf.Variable(0.0, trainable = False, name = "Eval_Dice_Coef")
        self.tf_eval_dice_summary = tf.summary.scalar("Evaluation Dice Coef", self.tf_eval_dice)
        self.writer = tf.summary.FileWriter('./graphs', self.sess.graph)
        #saver
        self.saver = tf.train.Saver()
        self.sess.run(tf.initialize_all_variables())

    def save_model(self, filename):
        with tf.Graph().as_default():
            self.saver.save(self.sess, filename)

    def train(self):
        with tf.Graph().as_default():
            for i_epoch in range(1, self.args.epochs+1):
                #init paramters for summary
                loss_train = []
                accuracy_train = []
                accuracy_val = []
                dice_train = []
                dice_val = []
                #streaming image
                images_train = img_generator('images_train.json')
                images_val = img_generator('images_val.json')
                #checkpoint
                self.save_model(filename = './checkpoints/checkpoint_epoch-{}.ckpt'.format(i_epoch))
                #train
                catIDs = list(range(0,self.args.nb_classes))
                print("Epoch {} \n".format(i_epoch))
                print("Train \n")
                for image in tqdm(images_train, total = 82783):
                    #create grouth truth map
                    y = np.zeros((512, 512, self.args.nb_classes))
                    for cat in catIDs:
                        annIds = self.coco.getAnnIds(imgIds = image['id'], catIds = [cat+1])
                        anns = self.coco.loadAnns(annIds)
                        if len(anns) > 0:
                            for ann in anns:
                                mask = self.coco.annToMask(ann)
                                mask = resize(mask, (512,512), interpolation = cv2.INTER_NEAREST)
                                y[:,:,cat] = np.logical_or(y[:,:,cat], mask).astype(np.float32)
                    #import image
                    img = io.imread("../train2014/{}".format(image["file_name"]))
                    img = resize(img, (512, 512))
                    if img.shape == (512,512):
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                    # print(np.expand_dims(img, axis = 0).shape)
                    #feed forward + back propagation
                    self.sess.run(self.train_op, feed_dict = {
                        self.input_img : np.expand_dims(img, axis = 0),
                        self.label : np.expand_dims(y, axis = 0)
                    })
                    #get loss training
                    loss_train.append(self.sess.run(self.loss, feed_dict = {
                        self.input_img : np.expand_dims(img, axis = 0),
                        self.label : np.expand_dims(y, axis = 0)
                    }))
                    #get accuracy training
                    softmax = self.sess.run(self.model.output_proba, feed_dict = {
                        self.input_img : np.expand_dims(img, axis = 0)
                    })[0]
                    predicted_mask =probaToBinaryMask(softmax)
                    nb_TP_bit = np.sum(np.logical_and(predicted_mask,y))
                    nb_total_bit = image["width"]*image["height"]*self.args.nb_classes
                    accuracy_train.append(nb_TP_bit/nb_total_bit)
                    #get dice coef training
                    intersection = nb_TP_bit
                    union = np.sum(predicted_mask) + np.sum(y)
                    dice_train.append(2*intersection/union)
                #evaluation
                print("Evaluation \n")
                for image in images_val:
                    #create grouth truth map
                    y = np.zeros((512, 512, self.args.nb_classes))
                    for cat in catIDs:
                        annIds = self.coco.getAnnIds(imgIds=image['id'], catIds=[cat])
                        anns = self.coco.loadAnns(annIds)
                        if len(anns) > 0:
                            for ann in anns:
                                mask = self.coco.annToMask(ann)
                                mask = resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
                                y[:, :, cat] = np.logical_or(y[:, :, cat], mask).astype(np.float32)
                    #import image
                    img = io.imread("../train2014/{}".format(image["file_name"]))
                    img = resize(img, (512,512))
                    if img.shape == (512,512):
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                    #predict
                    softmax = np.sess.run(self.model.output_proba, feed_dict = {
                        self.input_img : np.expand_dims(img, axis = 0)
                    })
                    #get accuracy val
                    predicted_mask = probaToBinaryMask(softmax)
                    nb_TP_bit = np.sum(np.logical_and(predicted_mask, y))
                    np_total_bit = image["height"]*image["width"]*self.args.nb_classes
                    accuracy_val.append(append(nb_TP_bit/nb_total_bit))
                    #get dice val
                    intersection = nb_TP_bit
                    union = np.sum(predicted_mask) + np.sum(y)
                    dice_val.append(2*intersection/union)
                #write event for tensorboard
                summary = self.sess.run(self.tf_train_accuracy_summary, feed_dict = {
                    self.tf_train_accuracy : np.mean(np.asarray(accuracy_train))
                })
                self.writer.add_summary(summary, i_epoch)
                summary = self.sess.run(self.tf_train_loss_summary, feed_dict = {
                    self.tf_train_loss : np.mean(np.asarray(loss_train))
                })
                self.writer.add_summary(summary, i_epoch)
                summary = self.sess.run(self.tf_train_dice_summary, feed_dict = {
                    self.tf_train_dice : np.mean(np.asarray(dice_train))
                })
                self.writer.add_summary(summary, i_epoch)
                summary = self.sess.run(self.tf_eval_accuracy_summary, feed_dict = {
                    self.tf_eval_accuracy : np.mean(np.asarray(accuracy_val))
                })
                self.writer.add_summary(summary, i_epoch)
                summary = self.sess.run(self.tf_eval_dice_summary, feed_dict = {
                    self.tf_eval_dice : np.mean(np.asarray(dice_val))
                })

if __name__ == "__main__":
    args = arg_parser()
    trainer = Trainer(args)
    print("===========Start Training===========\n")
    trainer.train()
    print("===========Straining Completed===========")