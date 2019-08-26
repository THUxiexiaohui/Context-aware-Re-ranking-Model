#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@paper: Improving Web Image Search using Contextual Information
@model: SIRM & Rocchio model
"""
import cv2
import os
from PIL import Image
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from numpy import dot
from numpy.linalg import norm
import cPickle as pickle
import time
no_clusters = 200


class ImageHelpers:
    def __init__(self):
        self.sift_object = cv2.xfeatures2d.SIFT_create()

    def gray(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray

    def features(self, image):
        keypoints, descriptors = self.sift_object.detectAndCompute(image, None)
        return [keypoints, descriptors]


class BOVHelpers:
    def __init__(self, n_clusters=20):
        self.n_clusters = n_clusters
        self.kmeans_obj = MiniBatchKMeans(n_clusters=n_clusters, n_init=1, batch_size=10000, tol=1e-4, verbose=True)
        self.descriptor_vstack = None

    def cluster(self):
        self.kmeans_obj.fit(self.descriptor_vstack)

    def formatND(self, l):
        l_sum = 0
        new_l = []
        for i in range(len(l)):
            l_sum += l[i].shape[0]
            new_l.append(l[i])
        vStack = np.zeros((l_sum, 128))
        start, end = 0, 0
        for i in range(len(new_l)):
            end = new_l[i].shape[0] + start
            vStack[start:end, :] = new_l[i]
            start += new_l[i].shape[0]
        self.descriptor_vstack = vStack.copy()


bov_helper = BOVHelpers(no_clusters)
im_helper = ImageHelpers()


id_path_map = pickle.load(open("../dataset/image_id_path.pkl", "r"))
img_root = "../dataset/img_data/"
img_file_map = {}


def read_img(img_id):
    img_file = img_root + id_path_map[img_id]
    if not os.path.exists(img_file):
        return []
    img_file_map[img_file] = 0
    im = Image.open(img_file).convert('L')
    im = im.resize((224, 224))
    im = np.array(im)
    return im


def readfile(data_file):
    count = 0
    train_img = {}
    data_cnt = 0
    for data in data_file:
        source, target = data
        data_cnt += 1
        for i in range(len(source)):
            if source[i][1] not in train_img:
                train_img[source[i][1]] = 0
        for i in range(len(target[1])):
            if target[1][i][0] not in train_img:
                train_img[target[1][i][0]] = 0
    descriptor_list = []
    for img in train_img:
        img = read_img(img, "train")
        if len(img) == 0:
            continue
        kp, des = im_helper.features(img)
        descriptor_list.append(des)
        count += 1
        if count % 10000 == 0:
            print "read_file", count / 10000
    print len(descriptor_list)
    # extract SIFT Features from each image
    bov_helper.formatND(descriptor_list)
    print "format end"
    bov_helper.cluster()
    print "cluster end"
    output = open("../dataset/bov_helper.pkl", "wb")
    pickle.dump(bov_helper.kmeans_obj, output)
    output.close()


#Social-sensed Image Re-ranking Model
def SIRM(data_file):
    data_score = []
    max_data_score = []
    data_cnt = 0
    kmeans_obj = pickle.load(open("../dataset/bov_helper.pkl", "r"))
    kmeans_obj.verbose = False
    for data in data_file:
        source, target = data
        source_vocab = np.zeros(no_clusters)
        source_cnt = 0
        if len(source) != 0:
            for i in range(len(source)):
                img = source[i][1]
                img = read_img(img)
                vocab = np.zeros(no_clusters)
                if len(img) != 0:
                    kp, des = im_helper.features(img)
                    test_ret = kmeans_obj.predict(des)
                    for each in test_ret:
                        vocab[each] += 1
                    source_cnt += 1
                source_vocab += vocab
            if source_cnt != 0:
                source_vocab = source_vocab * 1.0 / source_cnt
        source_vocab = source_vocab.astype(int)
        img_score_list = []
        for i in range(len(target[1])):
            img = target[1][i][0]
            img = read_img(img)
            vocab = np.zeros(no_clusters)
            if len(img) != 0:
                kp, des = im_helper.features(img)
                vocab = np.zeros(no_clusters)
                test_ret = kmeans_obj.predict(des)
                for each in test_ret:
                    vocab[each] += 1
            vocab = vocab.astype(int)
            associate = 0
            all = 0
            for index in range(no_clusters):
                if vocab[index] != 0 and source_vocab[index] != 0:
                    associate += 1
                if vocab[index] != 0 or source_vocab[index] != 0:
                    all += 1
            if all == 0:
                score = 0
            else:
                score = associate * 1.0 / all
            img_score_list.append([score, target[1][i][1]])
        img_score_list = sorted(img_score_list, key=lambda x: x[0], reverse=True)


#Rocchio Model
def rocchio(data_file):
    rocchio_img_score = []
    data_score = []
    max_data_score = []
    data_cnt = 0
    kmeans_obj = pickle.load(open("../dataset/bov_helper.pkl", "r"))
    kmeans_obj.verbose = False
    for data in data_file:
        source, target = data
        source_skip_vocab = []
        source_ch_vocab = []
        source_skip_cnt = 0
        source_ch_cnt = 0
        if len(source) != 0:
            for i in range(len(source)):
                img = source[i][1]
                img_type = source[i][2]
                vocab = np.zeros(no_clusters)
                img = read_img(img)
                kp, des = im_helper.features(img)
                test_ret = kmeans_obj.predict(des)
                for each in test_ret:
                    vocab[each] += 1
                if img_type == 0:
                    source_skip_cnt += 1
                if img_type != 0:
                    source_ch_cnt += 1
                if img_type == 0:
                    source_skip_vocab.append(vocab)
                if img_type != 0:
                    source_ch_vocab.append(vocab)
        source_skip = np.zeros(no_clusters)
        for vocab in source_skip_vocab:
            vocab = np.array(vocab) + 1e-5
            vocab = vocab * 1.0 / np.linalg.norm(vocab)
            source_skip += vocab
        if len(source_skip_vocab) > 0:
            source_skip = source_skip * 1.0/len(source_skip_vocab)

        source_ch = np.zeros(no_clusters)
        for vocab in source_ch_vocab:
            vocab = np.array(vocab) + 1e-5
            vocab = vocab * 1.0 / np.linalg.norm(vocab)
            source_ch += vocab
        if len(source_ch_vocab) > 0:
            source_ch = source_ch * 1.0 / len(source_ch_vocab)
        source_vocab = source_skip - source_ch
        img_score_list = []
        for i in range(len(target[1])):
            img = target[1][i][0]
            vocab = np.zeros(no_clusters)
            img = read_img(img)
            kp, des = im_helper.features(img)
            vocab = np.zeros(no_clusters)
            test_ret = kmeans_obj.predict(des)
            for each in test_ret:
                vocab[each] += 1
            vocab = np.array(vocab)
            vocab += 1e-5
            vocab = vocab * 1.0 / np.linalg.norm(vocab)
            associate = 0
            all = 0
            score = dot(vocab, source_vocab) / (norm(vocab) * norm(source_vocab))
            img_score_list.append([score, target[1][i][1]])
        img_score_list = sorted(img_score_list, key=lambda x: x[0], reverse=True)
        

if __name__ == "__main__":
    pass
