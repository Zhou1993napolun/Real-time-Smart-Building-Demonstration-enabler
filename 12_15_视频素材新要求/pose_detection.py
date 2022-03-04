from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import csv
import os
import shutil

from PIL import Image
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision
import cv2
import numpy as np
import time

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import imageio

import _init_paths
import models
from config import cfg
from config import update_config
from core.function import get_final_preds
from utils.transforms import get_affine_transform

from models.pose_hrnet import get_pose_net

COCO_KEYPOINT_INDEXES = {
    0: 'nose',
    1: 'left_eye',
    2: 'right_eye',
    3: 'left_ear',
    4: 'right_ear',
    5: 'left_shoulder',
    6: 'right_shoulder',
    7: 'left_elbow',
    8: 'right_elbow',
    9: 'left_wrist',
    10: 'right_wrist',
    11: 'left_hip',
    12: 'right_hip',
    13: 'left_knee',
    14: 'right_knee',
    15: 'left_ankle',
    16: 'right_ankle'
}

COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

SKELETON = [
    [1, 3], [1, 0], [2, 4], [2, 0], [0, 5], [0, 6], [5, 7], [7, 9], [6, 8], [8, 10], [5, 11], [6, 12], [11, 12],
    [11, 13], [13, 15], [12, 14], [14, 16]
]

CocoColors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

NUM_KPTS = 17

CTX = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

font = cv2.FONT_HERSHEY_SIMPLEX  # 定义字体

image_to_video_num = 16

def conv3D_output_size(img_size, padding, kernel_size, stride):
    # compute output shape of conv3D
    outshape = (np.floor((img_size[0] + 2 * padding[0] - (kernel_size[0] - 1) - 1) / stride[0] + 1).astype(int),
                np.floor((img_size[1] + 2 * padding[1] - (kernel_size[1] - 1) - 1) / stride[1] + 1).astype(int),
                np.floor((img_size[2] + 2 * padding[2] - (kernel_size[2] - 1) - 1) / stride[2] + 1).astype(int))
    return outshape

class CNN3D(nn.Module):
    def __init__(self, t_dim=16, img_x=240, img_y=320, drop_p=0.2, fc_hidden1=128, num_classes=5):
        super(CNN3D, self).__init__()

        # set video dimension
        self.t_dim = t_dim
        self.img_x = img_x
        self.img_y = img_y
        # fully connected layer hidden nodes
        self.fc_hidden1 = fc_hidden1
        self.drop_p = drop_p
        self.num_classes = num_classes
        self.ch1, self.ch2 = 16, 32
        self.k1, self.k2 = (5, 5, 5), (3, 3, 3)  # 3d kernel size
        self.s1, self.s2 = (2, 2, 2), (2, 2, 2)  # 3d strides
        self.pd1, self.pd2 = (1, 0, 0), (1, 0, 0)  # 3d padding

        # compute conv1 & conv2 output shape
        self.conv1_outshape = conv3D_output_size((self.t_dim, self.img_x, self.img_y), self.pd1, self.k1, self.s1)
        self.conv2_outshape = conv3D_output_size(self.conv1_outshape, self.pd2, self.k2, self.s2)
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=self.ch1, kernel_size=self.k1, stride=self.s1,
                               padding=self.pd1)
        self.bn1 = nn.BatchNorm3d(self.ch1)
        self.conv2 = nn.Conv3d(in_channels=self.ch1, out_channels=self.ch2, kernel_size=self.k2, stride=self.s2,
                               padding=self.pd2)
        self.bn2 = nn.BatchNorm3d(self.ch2)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout3d(self.drop_p)
        self.pool = nn.MaxPool3d(2)

        self.fc1 = nn.Linear(self.ch2 * self.conv2_outshape[0] * self.conv2_outshape[1] * self.conv2_outshape[2],
                             self.fc_hidden1)  # fully connected hidden layer

        self.fc2 = nn.Linear(self.fc_hidden1, self.num_classes)# fully connected layer, output = multi-classes

    def forward(self, x_3d):
        # Conv 1
        x = self.conv1(x_3d)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.drop(x)
        # Conv 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.drop(x)
        # FC 1 and 2
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc2(x)

        return x

class_model = torch.load('./new_dataset/mobile_epoch1.pth')
class_model.to(CTX)
class_model.eval()

def draw_pose(keypoints, img):
    """draw the keypoints and the skeletons.
    :params keypoints: the shape should be equal to [17,2]
    :params img:
    """
    assert keypoints.shape == (NUM_KPTS, 2)
    for i in range(len(SKELETON)):
        kpt_a, kpt_b = SKELETON[i][0], SKELETON[i][1]
        x_a, y_a = keypoints[kpt_a][0], keypoints[kpt_a][1]
        x_b, y_b = keypoints[kpt_b][0], keypoints[kpt_b][1]
        cv2.circle(img, (int(x_a), int(y_a)), 6, CocoColors[i], -1)
        cv2.circle(img, (int(x_b), int(y_b)), 6, CocoColors[i], -1)
        cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)), CocoColors[i], 2)


def draw_bbox(box, img):
    """draw the detected bounding box on the image.
    :param img:
    """
    cv2.rectangle(img, box[0], box[1], color=(0, 255, 0), thickness=3)


def get_person_detection_boxes(model, img, threshold=0.5):
    pred = model(img)
    pred_classes = [COCO_INSTANCE_CATEGORY_NAMES[i]
                    for i in list(pred[0]['labels'].cpu().numpy())]  # Get the Prediction Score
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])]
                  for i in list(pred[0]['boxes'].detach().cpu().numpy())]  # Bounding boxes
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    if not pred_score or max(pred_score) < threshold:
        return []
    # Get list of index with score greater than threshold
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
    pred_boxes = pred_boxes[:pred_t + 1]
    pred_classes = pred_classes[:pred_t + 1]

    person_boxes = []
    for idx, box in enumerate(pred_boxes):
        if pred_classes[idx] == 'person':
            person_boxes.append(box)

    return person_boxes


def get_pose_estimation_prediction(pose_model, image, center, scale):
    rotation = 0
    # print("the center is :  ", end='')
    # print(center)
    # print(center.shape)
    # print("the scale is :  ", end='')
    # print(scale)
    # print(scale.shape)

    # pose estimation transformation
    trans = get_affine_transform(center, scale, rotation, cfg.MODEL.IMAGE_SIZE)
    model_input = cv2.warpAffine(
        image,
        trans,
        (int(cfg.MODEL.IMAGE_SIZE[0]), int(cfg.MODEL.IMAGE_SIZE[1])),
        flags=cv2.INTER_LINEAR)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # pose estimation inference
    model_input = transform(model_input).unsqueeze(0)
    # switch to evaluate mode
    pose_model.eval()
    with torch.no_grad():
        # compute output heatmap
        output = pose_model(model_input)
        preds, _ = get_final_preds(
            cfg,
            output.clone().cpu().numpy(),
            np.asarray([center]),
            np.asarray([scale]))

        return preds


def box_to_center_scale(box, model_image_width, model_image_height):
    """convert a box to center,scale information required for pose transformation
    Parameters
    ----------
    box : list of tuple
        list of length 2 with two tuples of floats representing
        bottom left and top right corner of a box
    model_image_width : int
    model_image_height : int

    Returns
    -------
    (numpy array, numpy array)
        Two numpy arrays, coordinates for the center of the box and the scale of the box
    """
    center = np.zeros((2), dtype=np.float32)

    bottom_left_corner = box[0]
    top_right_corner = box[1]
    box_width = top_right_corner[0] - bottom_left_corner[0]
    box_height = top_right_corner[1] - bottom_left_corner[1]
    bottom_left_x = bottom_left_corner[0]
    bottom_left_y = bottom_left_corner[1]
    center[0] = bottom_left_x + box_width * 0.5
    center[1] = bottom_left_y + box_height * 0.5

    aspect_ratio = model_image_width * 1.0 / model_image_height
    pixel_std = 200

    if box_width > aspect_ratio * box_height:
        box_height = box_width * 1.0 / aspect_ratio
    elif box_width < aspect_ratio * box_height:
        box_width = box_height * aspect_ratio
    scale = np.array(
        [box_width * 1.0 / pixel_std, box_height * 1.0 / pixel_std],
        dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25

    return center, scale


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg', type=str, default='./inference-config.yaml')
    parser.add_argument('--video', type=str)
    parser.add_argument('--webcam', action='store_true')
    parser.add_argument('--image', type=str)
    parser.add_argument('--write', action='store_true')
    parser.add_argument('--showFps', action='store_true')

    parser.add_argument('opts',
                        help='Modify config options using the command-line',
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    # args expected by supporting codebase
    args.modelDir = ''
    args.logDir = ''
    args.dataDir = ''
    args.prevModelDir = ''
    return args


def main():
    count_num = 0
    special_cnt = 0
    state_name = "wait"
    cnt_list = []
    if True:
        # cudnn related setting
        cudnn.benchmark = cfg.CUDNN.BENCHMARK
        torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
        torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

        args = parse_args()
        update_config(cfg, args)

        box_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        box_model.to(CTX)
        box_model.eval()

        pose_model = eval('models.' + cfg.MODEL.NAME + '.get_pose_net')(
            cfg, is_train=False
        )

        if cfg.TEST.MODEL_FILE:
            print('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
            pose_model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
        else:
            print('expected model defined in config at TEST.MODEL_FILE')

        pose_model = torch.nn.DataParallel(pose_model, device_ids=cfg.GPUS)
        pose_model.to(CTX)
        pose_model.eval()

    data_create = False
    pose_detection_flag = True
    on_cloud_host = True
    data_img_num_list = [174, 121, 129, 84, 83]
    result_txt_name = ['run', 'walk', 'wave', 'box', 'down']
    num_result = 1

    # Loading an video or an image or webcam
    if args.webcam:
        vidcap = cv2.VideoCapture(0)
    elif args.video:
        vidcap = cv2.VideoCapture(args.video)
    elif args.image:
        image_bgr = cv2.imread(args.image)
    # else:
    #     print('please use --video or --webcam or --image to define the input.')
    #     return

    if args.webcam or args.video:
        if args.write:
            save_path = '12_15_pose_output.mp4'
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(save_path, fourcc, 24.0, (int(vidcap.get(3)), int(vidcap.get(4))))
        while True:
            ret, image_bgr = vidcap.read()
            if ret:
                last_time = time.time()
                image = image_bgr[:, :, [2, 1, 0]]

                input = []
                img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                img_tensor = torch.from_numpy(img / 255.).permute(2, 0, 1).float().to(CTX)
                input.append(img_tensor)

                # object detection box
                pred_boxes = get_person_detection_boxes(box_model, input, threshold=0.9)

                # pose estimation
                if len(pred_boxes) >= 1:
                    for box in pred_boxes:

                        print("the box is :   ", end='')
                        print(box)

                        center, scale = box_to_center_scale(box, cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1])
                        image_pose = image.copy() if cfg.DATASET.COLOR_RGB else image_bgr.copy()
                        pose_preds = get_pose_estimation_prediction(pose_model, image_pose, center, scale)
                        if len(pose_preds) >= 1:
                            for kpt in pose_preds:
                                draw_pose(kpt, image_bgr)  # draw the poses

                                '''
                                special add
                                '''

                                if pose_detection_flag:
                                    if special_cnt == 0:
                                        cnt_list.clear()

                                    # print(kpt)
                                    for i in range(17):
                                        kpt[i][0] = kpt[i][0] * 240 / image_bgr.shape[0]
                                        kpt[i][1] = kpt[i][1] * 320 / image_bgr.shape[1]
                                    # print(kpt)
                                    # print(image_bgr.shape)

                                    mid_img = np.zeros((240, 320))
                                    for j in range(len(SKELETON)):
                                        kpt_a, kpt_b = SKELETON[j][0], SKELETON[j][1]
                                        x_a, y_a = kpt[kpt_a][0], kpt[kpt_a][1]
                                        x_b, y_b = kpt[kpt_b][0], kpt[kpt_b][1]
                                        cv2.line(mid_img, (int(x_a), int(y_a)), (int(x_b), int(y_b)), 255, 2)

                                    cv2.imwrite("./mid_see/{}.png".format(count_num), mid_img)
                                    count_num = count_num + 1
                                    special_cnt += 1
                                    # temp_input = mid_img.reshape((240, 320))
                                    cnt_list.append(mid_img)

                                    if special_cnt == image_to_video_num:
                                        # imageio.mimsave('./temp_gif_data/{}.gif'.format(count_num), cnt_list, 'GIF', duration=0.35)
                                        cnt_np = np.array(cnt_list).reshape((1, 1, image_to_video_num, 240, 320))
                                        torch_input = Variable(torch.Tensor(cnt_np.astype(np.float64) / 127.5 - 1)).to(CTX)
                                        class_result = class_model(torch_input)

                                        # print(class_result.shape)

                                        max_loc = 0
                                        for i in range(len(result_txt_name)):
                                            if class_result[0][max_loc] < class_result[0][i]:
                                                max_loc = i

                                        state_name = result_txt_name[max_loc]

                                        special_cnt = 0
                                    # print(class_result)
                                    # print(result_txt_name[max_loc])

                                    image_bgr = cv2.putText(image_bgr, state_name, box[0], font, 3, (0, 255, 0), 3)
                                '''
                                end
                                '''
                

                if args.showFps:
                    fps = 1 / (time.time() - last_time)
                    img = cv2.putText(image_bgr, 'fps: ' + "%.2f" % (fps), (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                                      (0, 255, 0), 2)

                if args.write:
                    out.write(image_bgr)

                if on_cloud_host == False:
                    cv2.imshow('demo', image_bgr)
                if cv2.waitKey(1) & 0XFF == ord('q'):
                    break
            else:
                print('cannot load the video.')
                break

        cv2.destroyAllWindows()
        vidcap.release()
        if args.write:
            print('video has been saved as {}'.format(save_path))
            out.release()

    elif data_create:
        pred_boxes = []
        loss_loc = []
        for i in range(data_img_num_list[num_result]):
            data_img_list = []
            img_bgr = cv2.imread('./data_img/'+result_txt_name[num_result]+'/{}.jpg'.format(i))
            img_rgb = img_bgr[:, :, [2, 1, 0]]
            img = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img / 255.).permute(2, 0, 1).float().to(CTX)

            data_img_list.append(img_tensor)
            temp_pred_boxes = get_person_detection_boxes(box_model, data_img_list, threshold=0.9)
            if len(temp_pred_boxes) == 0:
                pred_boxes.append(temp_pred_boxes)
            else:
                pred_boxes.append(temp_pred_boxes[0])
                loss_loc.append(i)
            if i % 100 == 0 and i != 0:
                print("{} images finished".format(i))

        # pred_boxes = get_person_detection_boxes(box_model, data_img_list, threshold=0.9)
        print("step 1 finish")

        if len(pred_boxes) >= 1:
            data_result = []
            num_cnt = 0
            for box in pred_boxes:
                if box != []:
                    center, scale = box_to_center_scale(box, cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1])
                    img_bgr = cv2.imread('./data_img/'+result_txt_name[num_result]+'/{}.jpg'.format(loss_loc[num_cnt]))
                    img_rgb = img_bgr[:, :, [2, 1, 0]]
                    image_pose = img_rgb.copy() if cfg.DATASET.COLOR_RGB else img_bgr.copy()
                    pose_preds = get_pose_estimation_prediction(pose_model, image_pose, center, scale)
                    data_result.append(pose_preds)
                    num_cnt += 1
                    if num_cnt % 100 == 0 and num_cnt != 0:
                        print("{} images finished".format(num_cnt))
                        for kpt in pose_preds:
                            draw_pose(kpt, img_bgr)  # draw the poses
                        cv2.imwrite('test_out_{}.jpg'.format(num_cnt), img_bgr)
            result = np.array(data_result)
            result_temp = result.reshape((-1, 17, 2))
            result_0 = result_temp[:, :, 0].reshape((-1, 17))
            result_1 = result_temp[:, :, 1].reshape((-1, 17))
            print(result.shape)
            np.savetxt(result_txt_name[num_result] + "_0.txt", result_0)
            np.savetxt(result_txt_name[num_result] + "_1.txt", result_1)
        else:
            print('error I : find None. ')

    else:
        # estimate on the image
        last_time = time.time()
        image = image_bgr[:, :, [2, 1, 0]]

        input = []
        img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img / 255.).permute(2, 0, 1).float().to(CTX)
        input.append(img_tensor)

        # object detection box
        pred_boxes = get_person_detection_boxes(box_model, input, threshold=0.9)

        # pose estimation
        if len(pred_boxes) >= 1:
            for box in pred_boxes:
                center, scale = box_to_center_scale(box, cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1])
                image_pose = image.copy() if cfg.DATASET.COLOR_RGB else image_bgr.copy()
                pose_preds = get_pose_estimation_prediction(pose_model, image_pose, center, scale)
                if len(pose_preds) >= 1:
                    for kpt in pose_preds:
                        draw_pose(kpt, image_bgr)  # draw the poses

        if args.showFps:
            fps = 1 / (time.time() - last_time)
            img = cv2.putText(image_bgr, 'fps: ' + "%.2f" % (fps), (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0),
                              2)

        if args.write:
            save_path = 'output.jpg'
            cv2.imwrite(save_path, image_bgr)
            print('the result image has been saved as {}'.format(save_path))

        cv2.imshow('demo', image_bgr)
        if cv2.waitKey(0) & 0XFF == ord('q'):
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
