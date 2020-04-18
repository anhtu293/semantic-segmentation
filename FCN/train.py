import tensorflow as tf
import numpy as np
from model import FCN
import sys
import skimage.io as io
import cv2
from cv2 import resize
sys.path.append("../tools")
from protocol.coco import COCO
from utils import softmax, img_generator
import argparse
from tqdm import tqdm

def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nb_classes", default="10", type=int, help="number of categories")
    parser.add_argument("--epochs", default="10", type=int, help="number of epoches")
    parser.add_argument("--batch_size", default="2", type=int, help="size of minibatch")
    parser.add_argument("--learning_rate", default="0.001", type=float, help="learning_rate")
    parser.add_argument("--loss", default="crossentropy", help = "type of loss function : crossentropy or dice")
    parser.add_argument("--width", default="512", type=int, help="size of width and hight")
    args = parser.parse_args()
    return args

class Trainer2:
    def __init__(self,args):
        #save args
        self.args = args
        #init data
        self.coco_train = COCO("./annotations/instances_train2014.json")
        self.coco_val = COCO("./annotations/instances_val2014.json")
        #init tensorflow session
        tf.compat.v1.get_default_graph()
        config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config = config)
        #init model
        self.input_img = tf.placeholder(tf.float32, shape=(None,None,None,3))
        self.label = tf.placeholder(tf.float32, shape=(None,None,None,args.nb_classes))
        self.model = FCN(input_img = self.input_img, nb_classes = args.nb_classes)
        #define loss :Cross entropy or dice
        with tf.variable_scope('optimization'):
            with tf.variable_scope('loss'):
                labels = self.label
                proba = self.model.pred_up
                if args.loss == 'crossentropy' :
                    self.loss = -tf.reduce_mean(tf.multiply(labels,tf.log(proba)))
                elif args.loss == 'dice':
                    intersection = tf.reduce_sum(proba*labels)
                    union = tf.reduce_sum(proba + labels)
                    self.loss = -intersection/union
            self.optimizer = tf.train.MomentumOptimizer(learning_rate=args.learning_rate, momentum=0.99)
            self.train_op = self.optimizer.minimize(self.loss)
        #summary file on tensorboard
        self.tf_train_loss = tf.Variable(0.0, trainable = False, name = 'Train_Loss')
        self.tf_train_loss_summary = tf.summary.scalar("Loss", self.tf_train_loss)
        self.tf_train_accuracy = tf.Variable(0.0, trainable = False, name = 'Train_Accuracy')
        self.tf_train_accuracy_summary = tf.summary.scalar("Train_Accuracy", self.tf_train_accuracy)
        self.tf_train_dice = tf.Variable(0.0, trainable=False, name="Train_Dice_Coef")
        self.tf_train_dice_summary = tf.summary.scalar("Train_Dice_Coef", self.tf_train_dice)
        self.tf_eval_accuracy = tf.Variable(0.0, trainable = False, name = 'Eval_accuracy')
        self.tf_eval_accuracy_summary = tf.summary.scalar('Evaluation_Accuracy', self.tf_eval_accuracy)
        self.tf_eval_dice = tf.Variable(0.0, trainable = False, name = "Eval_Dice_Coef")
        self.tf_eval_dice_summary = tf.summary.scalar("Evaluation_Dice_Coef", self.tf_eval_dice)
        self.writer = tf.summary.FileWriter('./graphs', self.sess.graph)
        #saver
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())

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
                #images_train = img_generator('images_train.json')
                #images_val = img_generator('images_val.json')
                #checkpoint
                self.save_model(filename = './checkpoints/checkpoint_epoch-{}.ckpt'.format(i_epoch))
                #train
                catIDs = list(range(2,self.args.nb_classes+2))
                print("Epoch {} \n".format(i_epoch))
                print("Train \n")
                #minibatch
                minibatch_image = []
                minibatch_label = []
                count = 0
                #Find images with categories
                imgIds = []
                for catID in catIDs:
                  imgIds = list(set(imgIds)|set(self.coco_train.getImgIds(catIds = catID)))
                catIDs = [x-2 for x in catIDs]
                for imgId in tqdm(imgIds):
                    count += 1
                    #get image
                    image = self.coco_train.loadImgs([imgId])
                    #create grouth truth map
                    y = np.zeros((512, 512, self.args.nb_classes))
                    for cat in catIDs:
                        annIds = self.coco_train.getAnnIds(imgIds = image[0]['id'], catIds = [cat+1])
                        anns = self.coco_train.loadAnns(annIds)
                        if len(anns) > 0:
                            for ann in anns:
                                mask = self.coco_train.annToMask(ann)
                                mask = resize(mask, (512,512), interpolation = cv2.INTER_NEAREST)
                                y[:,:,cat] = np.logical_or(y[:,:,cat], mask).astype(np.float32)
                    #import image
                    img = io.imread("./train2014/{}".format(image[0]["file_name"]))
                    img = resize(img, (512, 512))
                    if img.shape == (512,512):
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                    minibatch_image.append(img)
                    minibatch_label.append(y)
                    if len(minibatch_image) == self.args.batch_size or count == len(imgIds):
                        # get loss training
                        loss_train.append(self.sess.run(self.loss, feed_dict={
                            self.input_img: np.asarray(minibatch_image),
                            self.label: np.asarray(minibatch_label)
                        }))
                        #feed forward + back propagation
                        self.sess.run(self.train_op, feed_dict = {
                            self.input_img : np.asarray(minibatch_image),
                            self.label : np.asarray(minibatch_label)
                        })
                        #get accuracy training
                        softmax = self.sess.run(self.model.pred_up, feed_dict = {
                            self.input_img : np.asarray(minibatch_image)
                        })
                        nb_total_bit = 512 * 512 * self.args.nb_classes
                        for i_batch in range(softmax.shape[0]):
                            predicted_mask =probaToBinaryMask(softmax[i_batch])
                            nb_TP_bit = np.sum(np.logical_and(predicted_mask,minibatch_label[i_batch]))
                            accuracy_train.append(nb_TP_bit/nb_total_bit)
                            #get dice coef training
                            intersection = nb_TP_bit
                            union = np.sum(predicted_mask) + np.sum(minibatch_label[i_batch])
                            dice_train.append(2*intersection/union)
                        #reset minibatch
                        minibatch_label.clear()
                        minibatch_image.clear()
                #evaluation
                #Find image with categories
                catIDs = list(range(2, self.args.nb_classes + 2))
                imgIds = []
                for catID in catIDs:
                  imgIds = list(set(imgIds)| set(self.coco_val.getImgIds(catIds = [2,3])))
                catIDs = [x-2 for x in catIDs]
                print("Evaluation \n")
                for imgId in tqdm(imgIds):
                    #get image
                    image = self.coco_val.loadImgs([imgId])
                    #create grouth truth map
                    y = np.zeros((512, 512, self.args.nb_classes))
                    for cat in catIDs:
                        annIds = self.coco_val.getAnnIds(imgIds=image[0]['id'], catIds=[cat])
                        anns = self.coco_val.loadAnns(annIds)
                        if len(anns) > 0:
                            for ann in anns:
                                mask = self.coco_val.annToMask(ann)
                                mask = resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
                                y[:, :, cat] = np.logical_or(y[:, :, cat], mask).astype(np.float32)
                    #import image
                    img = io.imread("./val2014/{}".format(image[0]["file_name"]))
                    img = resize(img, (512,512))
                    if img.shape == (512,512):
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                    #predict
                    softmax = self.sess.run(self.model.pred_up, feed_dict = {
                        self.input_img : np.expand_dims(img, axis = 0)
                    })
                    #get accuracy val
                    predicted_mask = probaToBinaryMask(softmax)
                    nb_TP_bit = np.sum(np.logical_and(predicted_mask, y))
                    np_total_bit = 512*512*self.args.nb_classes
                    accuracy_val.append(nb_TP_bit/nb_total_bit)
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
        self.save_model(filename='./model-{}.ckpt'.format(i_epoch))


if __name__ == "__main__":
    args = argument_parser()
    trainer = Trainer(args)
    print("===========Start Training===========\n")
    trainer.train()
    print("===========Straining Completed===========")
