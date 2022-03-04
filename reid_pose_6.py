#!/usr/bin/env python3

"""
 Copyright (c) 2019-2020 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import argparse
import time
import queue
from threading import Thread
import json
import logging as log
import os
import random
import sys
import torch

import cv2 as cv
import numpy as np

from utils.network_wrappers import Detector, VectorCNN, MaskRCNN, DetectionsFromFileReader
from mc_tracker.mct import MultiCameraTracker
from utils.analyzer import save_embeddings
from utils.misc import read_py_config, check_pressed_keys, AverageEstimator, set_log_config
from utils.video import MulticamCapture, NormalizerCLAHE
# from utils.visualization import visualize_multicam_detections, get_target_size

'''
special add
'''
import torch.nn as nn
import torch.nn.functional as F

from utils.visualization import get_target_size
from utils.my_visualization import visualize_multicam_detections
from my_model.pose_hrnet import *
from my_model.default import *
from my_model.default import _C as cfg

import imageio
'''
end
'''

from openvino.inference_engine import IECore  # pylint: disable=import-error,E0611

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                             'common/python'))
import monitors
import models
from openvino.inference_engine import IECore


set_log_config()

# clothes add
from pathlib import Path
from pipelines import AsyncPipeline
import colorsys

CLASS_NAMES_MODANET = ['bag',
                       'belt',
                       'boots',
                       'footwear',
                       'outer',
                       'dress',
                       'sunglasses',
                       'pants',
                       'top',
                       'shorts',
                       'skirt',
                       'headwear',
                       'scarf/tie']

CLASS_NAMES_DF = ['short_sleeve_top',
                  'long_sleeve_top',
                  'short_sleeve_outwear',
                  'long_sleeve_outwear',
                  'vest',
                  'sling',
                  'shorts',
                  'trousers',
                  'skirt',
                  'short_sleeve_dress',
                  'long_sleeve_dress',
                  'vest_dress',
                  'sling_dress']

'''
add of 3D-CNN
'''
def conv3D_output_size(img_size, padding, kernel_size, stride):
    # compute output shape of conv3D
    outshape = (np.floor((img_size[0] + 2 * padding[0] - (kernel_size[0] - 1) - 1) / stride[0] + 1).astype(int),
                np.floor((img_size[1] + 2 * padding[1] - (kernel_size[1] - 1) - 1) / stride[1] + 1).astype(int),
                np.floor((img_size[2] + 2 * padding[2] - (kernel_size[2] - 1) - 1) / stride[2] + 1).astype(int))
    return outshape

class CNN3D(nn.Module):
    def __init__(self, t_dim=16, img_x=240, img_y=320, drop_p=0.2, fc_hidden1=128, num_classes=3):
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
'''
end
'''

def check_detectors(args):
    detectors = {
        '--m_detector': args.m_detector,
        '--m_segmentation': args.m_segmentation,
        '--detections': args.detections
    }
    non_empty_detectors = [(det, value) for det, value in detectors.items() if value]
    det_number = len(non_empty_detectors)
    if det_number == 0:
        log.error('No detector specified, please specify one of the following parameters: '
                  '\'--m_detector\', \'--m_segmentation\' or \'--detections\'')
    elif det_number > 1:
        det_string = ''.join('\n\t{}={}'.format(det[0], det[1]) for det in non_empty_detectors)
        log.error('Only one detector expected but got {}, please specify one of them:{}'
                  .format(len(non_empty_detectors), det_string))
    return det_number


def update_detections(output, detections, frame_number):
    for i, detection in enumerate(detections):
        entry = {'frame_id': frame_number, 'scores': [], 'boxes': []}
        for det in detection:
            entry['boxes'].append(det[0])
            entry['scores'].append(float(det[1]))
        output[i].append(entry)


def save_json_file(save_path, data, description=''):
    save_dir = os.path.dirname(save_path)
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(save_path, 'w') as outfile:
        json.dump(data, outfile)
    if description:
        log.info('{} saved to {}'.format(description, save_path))


class FramesThreadBody:
    def __init__(self, capture, max_queue_length=2):
        self.process = True
        self.frames_queue = queue.Queue()
        self.capture = capture
        self.max_queue_length = max_queue_length

    def __call__(self):
        while self.process:
            if self.frames_queue.qsize() > self.max_queue_length:
                time.sleep(0.1)
                continue
            has_frames, frames = self.capture.get_frames()
            if not has_frames and self.frames_queue.empty():
                self.process = False
                break
            if has_frames:
                self.frames_queue.put(frames)


# clothes add
def draw_detections(frame, detections, palette, labels, threshold, dataset):
    size = frame.shape[:2]
    for detection in detections:
        if detection.score > threshold:
            xmin = max(int(detection.xmin), 0)
            ymin = max(int(detection.ymin), 0)
            xmax = min(int(detection.xmax), size[1])
            ymax = min(int(detection.ymax), size[0])
            class_id = int(detection.id)
            if dataset == 'df2':
                class_name = CLASS_NAMES_DF[class_id]
                color = palette[class_id]
                det_label = labels[class_id] if labels and len(labels) >= class_id else '#{}'.format(class_id)
                cv.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 6)
                cv.putText(frame, '{} {:.1%}'.format(class_name, detection.score),
                            (xmin, ymin - 7), cv.FONT_HERSHEY_COMPLEX, 1, color, 2)
            elif dataset == 'modanet':
                class_name = CLASS_NAMES_MODANET[class_id]
                color = palette[class_id]
                det_label = labels[class_id] if labels and len(labels) >= class_id else '#{}'.format(class_id)
                cv.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 6)
                cv.putText(frame, '{} {:.1%}'.format(class_name, detection.score),
                            (xmin, ymin - 7), cv.FONT_HERSHEY_COMPLEX, 2, color, 3)
            else:
                raise RuntimeError('Invalid dataset name: {}'.format(dataset))
            if isinstance(detection, models.DetectionWithLandmarks):
                for landmark in detection.landmarks:
                    cv.circle(frame, (int(landmark[0]), int(landmark[1])), 2, (0, 255, 255), 2)
    return frame

class ColorPalette:
    def __init__(self, n, rng=None):
        assert n > 0

        if rng is None:
            rng = random.Random(0xACE)

        candidates_num = 100
        hsv_colors = [(1.0, 1.0, 1.0)]
        for _ in range(1, n):
            colors_candidates = [(rng.random(), rng.uniform(0.8, 1.0), rng.uniform(0.5, 1.0))
                                 for _ in range(candidates_num)]
            min_distances = [self.min_distance(hsv_colors, c) for c in colors_candidates]
            arg_max = np.argmax(min_distances)
            hsv_colors.append(colors_candidates[arg_max])

        self.palette = [self.hsv2rgb(*hsv) for hsv in hsv_colors]

    @staticmethod
    def dist(c1, c2):
        dh = min(abs(c1[0] - c2[0]), 1 - abs(c1[0] - c2[0])) * 2
        ds = abs(c1[1] - c2[1])
        dv = abs(c1[2] - c2[2])
        return dh * dh + ds * ds + dv * dv

    @classmethod
    def min_distance(cls, colors_set, color_candidate):
        distances = [cls.dist(o, color_candidate) for o in colors_set]
        return np.min(distances)

    @staticmethod
    def hsv2rgb(h, s, v):
        return tuple(round(c * 255) for c in colorsys.hsv_to_rgb(h, s, v))

    def __getitem__(self, n):
        return self.palette[n % len(self.palette)]

    def __len__(self):
        return len(self.palette)



def run(params, config, capture, detector, reid, sk_model, class_model, detector_pipeline, clo_model, palette, cfg):
    win_name = 'Multi camera tracking'
    frame_number = 0
    avg_latency = AverageEstimator()
    output_detections = [[] for _ in range(capture.get_num_sources())]
    key = -1

    if config['normalizer_config']['enabled']:
        capture.add_transform(
            NormalizerCLAHE(
                config['normalizer_config']['clip_limit'],
                config['normalizer_config']['tile_size'],
            )
        )

    tracker = MultiCameraTracker(capture.get_num_sources(), reid, config['sct_config'], **config['mct_config'],
                                 visual_analyze=config['analyzer'])

    thread_body = FramesThreadBody(capture, max_queue_length=len(capture.captures) * 2)
    frames_thread = Thread(target=thread_body)
    frames_thread.start()

    frames_read = False
    set_output_params = False

    prev_frames = thread_body.frames_queue.get()
    detector.run_async(prev_frames, frame_number)
    presenter = monitors.Presenter(params.utilization_monitors, 0)

    id_list = []
    pose_list = []
    my_cnt = 0

    gif_cnt = 0

    # clothes add
    next_frame_id = 0
    next_frame_id_to_show = 0
    img_cnt = 0

    while thread_body.process:
        if not params.no_show:
            key = check_pressed_keys(key)
            if key == 27:
                break
            presenter.handleKey(key)
        start = time.perf_counter()
        try:
            frames = thread_body.frames_queue.get_nowait()
            frames_read = True
        except queue.Empty:
            frames = None

        if frames is None:
            continue

        all_detections = detector.wait_and_grab()
        if params.save_detections:
            update_detections(output_detections, all_detections, frame_number)
        frame_number += 1
        detector.run_async(frames, frame_number)

        all_masks = [[] for _ in range(len(all_detections))]
        for i, detections in enumerate(all_detections):
            all_detections[i] = [det[0] for det in detections]
            all_masks[i] = [det[2] for det in detections if len(det) == 3]

        tracker.process(prev_frames, all_detections, all_masks)
        tracked_objects = tracker.get_tracked_objects()

        latency = max(time.perf_counter() - start, sys.float_info.epsilon)
        avg_latency.update(latency)
        fps = round(1. / latency, 1)

        # prev_frame[0] 是我们输入视频中的每张原始图片
        '''      
        tracked_objects已经储存了所有的人物识别点了，其是一个二维的list，第一维的长度与prev_frame的长度一致（这里是1）
        第二维的长度是图像中人物的个数
        基本元素为：TrackedObj(rect=(786, 277, 953, 886), label='ID -1')，其.rect为一个元组，便包含了位置坐标
        print(tracked_objects[0][0].rect)
        print(tracked_objects[0][0].rect.shape)
        print(len(tracked_objects[0][0]))
        '''

        '''
        要修改的部分是在visualize_multicam_detections函数里
        '''

        # print("The prev_frames shape is :  ")
        # print(len(prev_frames))             # len(prev_frames) = 1
        # print(prev_frames[0].shape)         # shape is (1080, 1920, 3)

        results = detector_pipeline.get_result(next_frame_id_to_show)
        if results:
            next_frame_id_to_show += 1
            objects, frame_meta = results

            temp_out = frame_meta['frame']
            cv.imwrite('/home/ubuntu/hws/reid_pose/12_1_dir/temp_out_img/{}.jpg'.format(next_frame_id_to_show), temp_out)

            prev_frames[0] = draw_detections(prev_frames[0], objects, palette, clo_model.labels, 0.5, 'df2')


        if detector_pipeline.is_ready():
            temp_frames = prev_frames
            detector_pipeline.submit_data(temp_frames[0], next_frame_id, {'frame': temp_frames[0]})
            next_frame_id += 1



        # vis = visualize_multicam_detections(prev_frames, tracked_objects, fps, **config['visualization_config'])
        vis, id_list, pose_list, gif_cnt, img_cnt = visualize_multicam_detections(prev_frames, tracked_objects, id_list, pose_list, sk_model, class_model, cfg, img_cnt, gif_cnt, fps='', **config['visualization_config'])

        # 这里的vis已经是打上ID的图片了
        # print(vis.shape)
        cv.imwrite("/home/ubuntu/hws/reid_pose/temp_images/{}.jpg".format(my_cnt), vis)
        my_cnt += 1

        presenter.drawGraphs(vis)
        # if not params.no_show:
            # cv.imshow(win_name, vis)

        if frames_read and not set_output_params:
            set_output_params = True
            if len(params.output_video):

                # print("see")
                # for frame in frames:
                #     print(frame.shape)

                frame_size = [frame.shape[::-1] for frame in frames]

                print("frame_size is : ")
                print(frame_size)


                fps = capture.get_fps()
                target_width, target_height = get_target_size(frame_size, None, **config['visualization_config'])
                video_output_size = (target_width, target_height)
                fourcc = cv.VideoWriter_fourcc(*'XVID')
                output_video = cv.VideoWriter(params.output_video, fourcc, min(fps), video_output_size)
            else:
                output_video = None
        if set_output_params and output_video:
            output_video.write(cv.resize(vis, video_output_size))

        print('\rProcessing frame: {}, fps = {} (avg_fps = {:.3})'.format(
                            frame_number, fps, 1. / avg_latency.get()), end="")
        prev_frames, frames = frames, prev_frames
    print(presenter.reportMeans())
    print('')

    thread_body.process = False
    frames_thread.join()

    if len(params.history_file):
        save_json_file(params.history_file, tracker.get_all_tracks_history(), description='History file')
    if len(params.save_detections):
        save_json_file(params.save_detections, output_detections, description='Detections')

    if len(config['embeddings']['save_path']):
        save_embeddings(tracker.scts, **config['embeddings'])


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    """Prepares data for the object tracking demo"""
    parser = argparse.ArgumentParser(description='Multi camera multi object \
                                                  tracking live demo script')
    parser.add_argument('-i', '--input', required=True, nargs='+',
                        help='Input sources (indexes of cameras or paths to video files)')

    parser.add_argument('--loop', default=False, action='store_true',
                        help='Optional. Enable reading the input in a loop')
    parser.add_argument('--config', type=str, default=os.path.join(current_dir, 'configs/person.py'), required=False,
                        help='Configuration file')

    # 加的
    parser.add_argument('--cfg', type=str, default='./inference-config.yaml')

    parser.add_argument('--detections', type=str, help='JSON file with bounding boxes')

    parser.add_argument('-m', '--m_detector', type=str, required=False,
                        help='Path to the object detection model')
    parser.add_argument('--t_detector', type=float, default=0.6,
                        help='Threshold for the object detection model')


    parser.add_argument('--m_segmentation', type=str, required=False,
                        help='Path to the object instance segmentation model')
    parser.add_argument('--t_segmentation', type=float, default=0.6,
                        help='Threshold for object instance segmentation model')

    parser.add_argument('--m_reid', type=str, required=True,
                        help='Path to the object re-identification model')

    parser.add_argument('--output_video', type=str, default='', required=False,
                        help='Optional. Path to output video')
    parser.add_argument('--history_file', type=str, default='', required=False,
                        help='Optional. Path to file in JSON format to save results of the demo')
    parser.add_argument('--save_detections', type=str, default='', required=False,
                        help='Optional. Path to file in JSON format to save bounding boxes')
    parser.add_argument("--no_show", help="Optional. Don't show output", action='store_true')

    parser.add_argument('-d', '--device', type=str, default='CPU')
    parser.add_argument('-l', '--cpu_extension',
                        help='MKLDNN (CPU)-targeted custom layers.Absolute \
                              path to a shared library with the kernels impl.',
                             type=str, default=None)
    parser.add_argument('-u', '--utilization_monitors', default='', type=str,
                        help='Optional. List of monitors to show initially.')
    # add
    parser.add_argument('opts',
                        help='Modify config options using the command-line',
                        default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('-clo_m', '--clo_model', help='Required. Path to an .xml file with a trained model.',
                        required=True, type=Path)



    args = parser.parse_args()

    # add
    # args expected by supporting codebase
    args.modelDir = ''
    args.logDir = ''
    args.dataDir = ''
    args.prevModelDir = ''


    if check_detectors(args) != 1:
        sys.exit(1)

    if len(args.config):
        log.info('Reading configuration file {}'.format(args.config))
        config = read_py_config(args.config)
    else:
        log.error('No configuration file specified. Please specify parameter \'--config\'')
        sys.exit(1)

    random.seed(config['random_seed'])
    capture = MulticamCapture(args.input, args.loop)


    '''
    clothes add
    '''
    log.info("Creating Inference Engine")
    ie = IECore()

    clo_model = models.YOLO(ie, args.clo_model, labels=None, threshold=0.5, keep_aspect_ratio=False)
    detector_pipeline = AsyncPipeline(ie, clo_model, {},
                                      device=args.device, max_num_requests=1)

    palette = ColorPalette(len(clo_model.labels) if clo_model.labels else 100)
    '''
    end
    '''


    if args.detections:
        object_detector = DetectionsFromFileReader(args.detections, args.t_detector)
    elif args.m_segmentation:
        object_detector = MaskRCNN(ie, args.m_segmentation,
                                   config['obj_segm']['trg_classes'],
                                   args.t_segmentation,
                                   args.device, args.cpu_extension,
                                   capture.get_num_sources())
    else:
        object_detector = Detector(ie, args.m_detector,
                                   config['obj_det']['trg_classes'],
                                   args.t_detector,
                                   args.device, args.cpu_extension,
                                   capture.get_num_sources())

    if args.m_reid:
        object_recognizer = VectorCNN(ie, args.m_reid, args.device, args.cpu_extension)
    else:
        object_recognizer = None

    # sk_model = torch.load('./my_model/sk_model.pth')
    update_config(cfg, args)
    sk_model = get_pose_net(cfg, False)
    sk_model.load_state_dict(torch.load("./my_model/sk_model.pth"))
    sk_model.eval()

    class_model = torch.load('./my_model/mobile_epoch2.pth')
    class_model.eval()

    run(args, config, capture, object_detector, object_recognizer, sk_model, class_model, detector_pipeline, clo_model, palette, cfg)
    log.info('Demo finished successfully')


if __name__ == '__main__':
    main()
