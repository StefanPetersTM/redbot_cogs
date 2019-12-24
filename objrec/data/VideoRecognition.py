from __future__ import division, print_function
import time
import os

import tensorflow as tf
import numpy as np
import cv2

from .utils.misc_utils import parse_anchors, read_class_names
from .utils.nms_utils import gpu_nms
from .utils.plot_utils import get_color_table, plot_one_box
from .utils.data_aug import letterbox_resize

from .model1 import yolov3

def vid_rec(input_video, sess2, boxes, scores, labels, input_data, classes, color_table, anchor_path = r'D:\GithubProjects\TM\data\yolo_anchors.txt', new_size = [416, 416], letterbox_resize1 = True, class_name_path = r'D:\GithubProjects\TM\data\coco.names', restore_path = r'D:\GithubProjects\TM\data\darknet_weights\yolov3.ckpt'):

    anchors = parse_anchors(anchor_path)
    classes = read_class_names(class_name_path)
    num_class = len(classes)

    color_table = get_color_table(num_class)

    vid = cv2.VideoCapture(input_video)
    video_frame_cnt = int(vid.get(7))
    video_width = int(vid.get(3))
    video_height = int(vid.get(4))
    video_fps = int(vid.get(5))


    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

    filepath, file_extension = os.path.splitext(input_video)
    processed_vid = '{}_yolo_out.mp4'.format(filepath)

    videoWriter = cv2.VideoWriter(processed_vid, fourcc, video_fps, (video_width, video_height))

    final_labels = []
    inf_time = 0

    #with tf.Session() as sess2:
        #input_data = tf.placeholder(tf.float32, [1, new_size[1], new_size[0], 3], name='input_data')
        #yolo_model = yolov3(num_class, anchors)
        #with tf.variable_scope('yolov3'):
            #pred_feature_maps = yolo_model.forward(input_data, False)
        #pred_boxes, pred_confs, pred_probs = yolo_model.predict(pred_feature_maps)

        #pred_scores = pred_confs * pred_probs

        #boxes, scores, labels = gpu_nms(pred_boxes, pred_scores, num_class, max_boxes=200, score_thresh=0.5, nms_thresh=0.45)

        #saver = tf.train.Saver()
        #saver.restore(sess2, restore_path)

    for i in range(video_frame_cnt):
        ret, img_ori = vid.read()
        if letterbox_resize:
            img, resize_ratio, dw, dh = letterbox_resize(img_ori, new_size[0], new_size[1])
        else:
            height_ori, width_ori = img_ori.shape[:2]
            img = cv2.resize(img_ori, tuple(new_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.asarray(img, np.float32)
        img = img[np.newaxis, :] / 255.

        start_time = time.time()
        boxes_, scores_, labels_ = sess2.run([boxes, scores, labels], feed_dict={input_data: img})
        end_time = time.time()
        inference_time = (end_time - start_time) * 1000
        # rescale the coordinates to the original image
        if letterbox_resize:
            boxes_[:, [0, 2]] = (boxes_[:, [0, 2]] - dw) / resize_ratio
            boxes_[:, [1, 3]] = (boxes_[:, [1, 3]] - dh) / resize_ratio
        else:
            boxes_[:, [0, 2]] *= (width_ori/float(new_size[0]))
            boxes_[:, [1, 3]] *= (height_ori/float(new_size[1]))


        for i in range(len(boxes_)):
            x0, y0, x1, y1 = boxes_[i]
            plot_one_box(img_ori, [x0, y0, x1, y1], label=classes[labels_[i]] + ', {:.2f}%'.format(scores_[i] * 100), color=color_table[labels_[i]])

            if classes[labels_[i]] not in final_labels:
                final_labels.append(classes[labels_[i]])

        cv2.putText(img_ori, '{:.2f}ms'.format(inference_time), (40, 40), 0,
                    fontScale=1, color=(0, 255, 0), thickness=2)
        #cv2.imshow('image', img_ori)
        videoWriter.write(img_ori)
        if i != 0:
            inf_time += inference_time
        video_frame_cnt = i
    try:
        inference_time = round(inf_time/(video_frame_cnt))
    except:
        pass
    vid.release()
    videoWriter.release()
    print("Path to processed video: " + processed_vid)


    #print(labels_)
    #for i in labels_:
    #    if i not in final_labels:
    #        final_labels.append(i)

    print("\nLabels: " + str(final_labels))
    print("Inference time: " + str(inference_time))

    return processed_vid, final_labels, inference_time