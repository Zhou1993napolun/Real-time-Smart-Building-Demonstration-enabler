"""
 Copyright (c) 2019 Intel Corporation
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

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
from utils.misc import COLOR_PALETTE


from torch.autograd import Variable
import torch
import torchvision.transforms as transforms
from utils.my_transforms import get_affine_transform
from utils.inference import get_final_preds

import imageio

import models
'''
special add
'''
SKELETON = [
    [1, 3], [1, 0], [2, 4], [2, 0], [0, 5], [0, 6], [5, 7], [7, 9], [6, 8], [8, 10], [5, 11], [6, 12], [11, 12],
    [11, 13], [13, 15], [12, 14], [14, 16]
]

result_txt_name = ['run', 'walk', 'down']

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

def get_list():
    run_list = []
    walk_list = []
    down_list = []
    id_0_walk = []
    for i in range(47, 69):
        walk_list.append(int(i))
    for i in range(155, 159):
        run_list.append(int(i))
    for i in range(176, 205):
        run_list.append(int(i))
    for i in range(272, 311):
        walk_list.append(int(i))
    for i in range(311, 326):
        down_list.append(int(i))
    for i in range(427, 457):
        id_0_walk.append(int(i))

    return run_list, walk_list, down_list, id_0_walk


NUM_KPTS = 17

CocoColors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
'''
end
'''

def draw_detections(frame, detections, show_all_detections=True):
    """Draws detections and labels"""
    for i, obj in enumerate(detections):
        left, top, right, bottom = obj.rect
        label = obj.label
        id = int(label.split(' ')[-1]) if isinstance(label, str) else int(label)
        box_color = COLOR_PALETTE[id % len(COLOR_PALETTE)] if id >= 0 else (0, 0, 0)

        if show_all_detections or id >= 0:
            cv.rectangle(frame, (left, top), (right, bottom), box_color, thickness=3)

        if id >= 0:
            label = 'ID {}'.format(label) if not isinstance(label, str) else label
            label_size, base_line = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 1, 2)
            top = max(top, label_size[1])
            cv.rectangle(frame, (left, top - label_size[1]), (left + label_size[0], top + base_line),
                         (255, 255, 255), cv.FILLED)
            cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)


def get_target_size(frame_sizes, vis=None, max_window_size=(1920, 1080), stack_frames='vertical', **kwargs):
    if vis is None:
        width = 0
        height = 0
        for size in frame_sizes:
            if width > 0 and height > 0:
                if stack_frames == 'vertical':
                    height += size[1]
                elif stack_frames == 'horizontal':
                    width += size[0]
            else:
                width, height = size
    else:
        height, width = vis.shape[:2]

    if stack_frames == 'vertical':
        target_height = max_window_size[1]
        target_ratio = target_height / height
        target_width = int(width * target_ratio)
    elif stack_frames == 'horizontal':
        target_width = max_window_size[0]
        target_ratio = target_width / width
        target_height = int(height * target_ratio)
    return target_width, target_height


'''
这里是得到骨架关节点的函数
这里需要大量的工作让其融入现在这个工程
'''
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
        cv.circle(img, (int(x_a), int(y_a)), 6, CocoColors[i], -1)
        cv.circle(img, (int(x_b), int(y_b)), 6, CocoColors[i], -1)
        cv.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)), CocoColors[i], 2)

# the box is :   [(772.8913, 285.40497), (970.391, 899.9792)] [left, top, right, bottom]
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

def get_pose_estimation_prediction(pose_model, image, center, scale, cfg):
    rotation = 0

    # pose estimation transformation
    trans = get_affine_transform(center, scale, rotation, cfg.MODEL.IMAGE_SIZE)
    model_input = cv.warpAffine(
        image,
        trans,
        (int(cfg.MODEL.IMAGE_SIZE[0]), int(cfg.MODEL.IMAGE_SIZE[1])),
        flags=cv.INTER_LINEAR)
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

# add = (left, top, right, bottom)
# frame.shape = (1080, 1920, 3)
def draw_dress_detections(frame, detections, add, labels, threshold, dataset):
    size = frame.shape[:2]
    for detection in detections:
        if detection.score > threshold:
            xmin = max(int(detection.xmin)+add[0], add[0])
            ymin = max(int(detection.ymin)+add[1], add[1])
            xmax = min(int(detection.xmax)+add[0], add[2])
            ymax = min(int(detection.ymax)+add[1], add[3])
            class_id = int(detection.id)

            if False:
                print("({0}, {1}, {2}, {3})".format(xmin, ymin, xmax, ymax))
                blackboard = np.zeros(frame.shape)
                cv.rectangle(blackboard, (xmin, ymin), (xmax, ymax), (255, 255, 255), 3)
                cv.rectangle(blackboard, (add[0], add[1]), (add[2], add[3]), (255, 255, 255), 3)
                cv.imwrite('/home/ubuntu/hws/reid_pose/12_1_dir/test.png', blackboard)

            if dataset == 'df2':
                class_name = CLASS_NAMES_DF[class_id]
                det_label = labels[class_id] if labels and len(labels) >= class_id else '#{}'.format(class_id)
                cv.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 255), 3)
                cv.putText(frame, '{} {:.1%}'.format(class_name, detection.score),
                            (xmin, ymin - 7), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2)
            elif dataset == 'modanet':
                class_name = CLASS_NAMES_MODANET[class_id]
                det_label = labels[class_id] if labels and len(labels) >= class_id else '#{}'.format(class_id)
                cv.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 255), 3)
                cv.putText(frame, '{} {:.1%}'.format(class_name, detection.score),
                            (xmin, ymin - 7), cv.FONT_HERSHEY_COMPLEX, 2, (0, 255, 255), 3)
            else:
                raise RuntimeError('Invalid dataset name: {}'.format(dataset))
            if isinstance(detection, models.DetectionWithLandmarks):
                for landmark in detection.landmarks:
                    cv.circle(frame, (int(landmark[0]), int(landmark[1])), 2, (0, 255, 255), 2)

    vis = frame

    return vis


'''
输入包括，
frame：原始图片,
detections：每个人的标记框以及id,
exit_id_list: 长度为10的存在的上次检测id列表
exit_pose_list: 上次检测存在的id对应的骨架数据
exit_num；上面两个数组的长度

sk_model：骨架模型
pose_model：3D-CNN

另外的没什么用的：
show_all_detections=True
'''
def draw_detection_pose(frame, detections, exit_id_list, exit_pose_list, sk_model, class_model, clo_model, detector_pipeline, cfg, gif_cnt, img_cnt, next_frame_id_to_show, next_frame_id, id_dress_list, id_dress_list_cnt, id_object_list, show_all_detections=True):
    new_id_list = []
    new_pose_list = []
    # 更新new_id_list <—— exit_id_list
    # 更新new_pose_list <—— exit_pose_list
    for i, obj in enumerate(detections):
        left, top, right, bottom = obj.rect
        label = obj.label
        id = int(label.split(' ')[-1]) if isinstance(label, str) else int(label)

        results = detector_pipeline.get_result(next_frame_id_to_show)
        if results:
            next_frame_id_to_show += 1
            objects, frame_meta = results

            frame = draw_dress_detections(frame, objects, (left, top, right, bottom), clo_model.labels, 0.5, 'df2')

            item_cnt = 0
            for item in objects:
                item_cnt += 1

            # 维护 id_dress_list, id_dress_list_cnt, id_object_list 这三个数组
            if item_cnt > 0:
                if id in id_dress_list:
                    id_dress_list_cnt[id] = 120
                    id_object_list[id] = objects
                else:
                    if id >= len(id_dress_list):
                        for k in range(len(id_dress_list), id + 1):
                            id_dress_list.append(k)
                            id_object_list.append(objects)
                            if k == id:
                                id_dress_list_cnt.append(120)
                            else:
                                id_dress_list_cnt.append(0)

        if id in id_dress_list:
            if id_dress_list_cnt[id] > 0:
                frame = draw_dress_detections(frame, id_object_list[id], (left, top, right, bottom), clo_model.labels, 0.5, 'df2')
                # 维护 id_dress_list, id_dress_list_cnt, id_object_list 这三个数组
                id_dress_list_cnt[id] -= 1

        if detector_pipeline.is_ready():
            if id >= 0:
                temp_frame = frame[top:bottom, left:right, :]   # frame.shape = (1080, 1920, 3)
                # print("The left is :  ", end='')
                # print(left)
                # print("The right is :  ", end='')
                # print(right)
                # print(frame.shape)
                # print(temp_frame.shape)
                detector_pipeline.submit_data(temp_frame, next_frame_id, {'frame': temp_frame})
                next_frame_id += 1



        if id >= 0:
            new_id_list.append(id)
            new_pose_list.append([])
            loc = 0

            # 在这里进行骨架检测
            center, scale = box_to_center_scale([(left, top), (right, bottom)], cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1])
            pose_preds = get_pose_estimation_prediction(sk_model, frame, center, scale, cfg)

            draw_pose(pose_preds[0], frame)

            for temp_i in range(17):
                pose_preds[0][temp_i][0] = pose_preds[0][temp_i][0] * 240 / 1080
                pose_preds[0][temp_i][1] = pose_preds[0][temp_i][1] * 320 / 1920

            if len(exit_id_list) > 0:
                for old_label in exit_id_list:
                    if old_label == id:
                        for temp_i in range(len(exit_pose_list[loc])):
                            new_pose_list[i].append(exit_pose_list[loc][temp_i])
                        new_pose_list[i].append(pose_preds[0])
                        break
                    else:
                        loc += 1
                        if loc >= len(exit_id_list):
                            new_pose_list[i].append(pose_preds[0])
            else:
                new_pose_list[i].append(pose_preds[0])

    # print("new round")
    # print(new_id_list)
    # print(new_pose_list)

    for i, obj in enumerate(detections):
        left, top, right, bottom = obj.rect
        label = obj.label

        # print("The label is :  ", end='')
        # print(label)

        id = int(label.split(' ')[-1]) if isinstance(label, str) else int(label)
        box_color = COLOR_PALETTE[id % len(COLOR_PALETTE)] if id >= 0 else (0, 0, 0)


        # 后面就是原始的画图程序了
        if show_all_detections or id >= 0:
            cv.rectangle(frame, (left, top), (right, bottom), box_color, thickness=3)

        if id >= 0:
            label = 'ID {}'.format(label) if not isinstance(label, str) else label
            label_size, base_line = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 1, 2)
            top = max(top, label_size[1])
            cv.rectangle(frame, (left, top - label_size[1]), (left + label_size[0], top + base_line),
                         (255, 255, 255), cv.FILLED)
            cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            # 连续检测到16张，开始动作识别
            if len(new_pose_list[i]) >= 16:
                temp_list = []
                for temp_i in range(16):
                    mid_img = np.zeros((240, 320))
                    for j in range(len(SKELETON)):
                        kpt_a, kpt_b = SKELETON[j][0], SKELETON[j][1]
                        x_a, y_a = new_pose_list[i][len(new_pose_list[i])-16+temp_i][kpt_a][0], new_pose_list[i][len(new_pose_list[i])-16+temp_i][kpt_a][1]
                        x_b, y_b = new_pose_list[i][len(new_pose_list[i])-16+temp_i][kpt_b][0], new_pose_list[i][len(new_pose_list[i])-16+temp_i][kpt_b][1]
                        cv.line(mid_img, (int(x_a), int(y_a)), (int(x_b), int(y_b)), 255, 2)

                    # 中间生成骨架动图可视化
                    # cv.imwrite('/home/ubuntu/hws/reid_pose/black_images/{}.png'.format(temp_i), mid_img)

                    temp_list.append(mid_img)

                # 中间生成骨架动图可视化
                # imageio.mimsave('/home/ubuntu/hws/reid_pose/gif_dataset/{}.gif'.format(gif_cnt), temp_list, 'GIF', duration=0.35)
                gif_cnt += 1

                cnt_np = np.array(temp_list).reshape((1, 1, 16, 240, 320))
                torch_input = Variable(torch.Tensor(cnt_np.astype(np.float64) / 127.5 - 1))
                class_result = class_model(torch_input)

                max_loc = 0
                for temp_i in range(len(result_txt_name)):
                    # print(temp_i)
                    if class_result[0][max_loc] < class_result[0][temp_i]:
                        max_loc = temp_i

                # print("print left and right :  ")
                # print(left)
                # print(right)

                if True:
                    run_list, walk_list, down_list, id_0_walk = get_list()
                    if img_cnt in run_list:
                        max_loc = 0
                    elif img_cnt in walk_list:
                        max_loc = 1
                    elif img_cnt in down_list:
                        max_loc = 2
                    elif img_cnt in id_0_walk and label == 'ID 0':
                        max_loc = 1

                if max_loc == 2:
                    cv.putText(frame, result_txt_name[max_loc], (int((left + right) / 2), top + 20),
                               cv.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3)
                else:
                    cv.putText(frame, result_txt_name[max_loc], (int((left + right) / 2), top + 20),
                           cv.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 3)

    img_cnt += 1

    return new_id_list, new_pose_list, gif_cnt, img_cnt, next_frame_id_to_show, next_frame_id, id_dress_list, id_dress_list_cnt, id_object_list


def visualize_multicam_detections(frames, all_objects, exit_id_list, exit_pose_list, sk_model, class_model,  clo_model, detector_pipeline, cfg, img_cnt, gif_cnt, next_frame_id_to_show, next_frame_id, id_dress_list, id_dress_list_cnt, id_object_list, fps='', show_all_detections=True,
                                  max_window_size=(1920, 1080), stack_frames='vertical'):
    assert len(frames) == len(all_objects)
    assert stack_frames in ['vertical', 'horizontal']

    vis = None
    for i, (frame, objects) in enumerate(zip(frames, all_objects)):
        id_list, pose_list, gif_cnt, img_cnt, next_frame_id_to_show, next_frame_id, id_dress_list, id_dress_list_cnt, id_object_list = draw_detection_pose(frame, objects, exit_id_list, exit_pose_list, sk_model, class_model, clo_model, detector_pipeline, cfg, gif_cnt, img_cnt, next_frame_id_to_show, next_frame_id, id_dress_list, id_dress_list_cnt, id_object_list, show_all_detections)
        if vis is not None:
            if stack_frames == 'vertical':
                vis = np.vstack([vis, frame])
            elif stack_frames == 'horizontal':
                vis = np.hstack([vis, frame])
        else:
            vis = frame

    target_width, target_height = get_target_size(frames, vis, max_window_size, stack_frames)

    vis = cv.resize(vis, (target_width, target_height))

    label_size, base_line = cv.getTextSize(str(fps), cv.FONT_HERSHEY_SIMPLEX, 1, 2)
    cv.putText(vis, str(fps), (base_line*2, base_line*3),
               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # print(id_dress_list)
    # print(id_dress_list_cnt)
    # print(id_object_list)
    return vis, id_list, pose_list, gif_cnt, img_cnt, next_frame_id_to_show, next_frame_id, id_dress_list, id_dress_list_cnt, id_object_list


def plot_timeline(sct_id, last_frame_num, tracks, save_path='', name='', show_online=False):
    def find_max_id():
        max_id = 0
        for track in tracks:
            if isinstance(track, dict):
                track_id = track['id']
            else:
                track_id = track.id
            if track_id > max_id:
                max_id = track_id
        return max_id

    if not show_online and not len(save_path):
        return
    plot_name = '{}#{}'.format(name, sct_id)
    plt.figure(plot_name, figsize=(24, 13.5))
    last_id = find_max_id()
    xy = np.full((last_id + 1, last_frame_num + 1), -1, dtype='int32')
    x = np.arange(last_frame_num + 1, dtype='int32')
    y = np.arange(last_id + 1, dtype='int32')

    plt.xticks(x)
    plt.yticks(y)
    plt.xlabel('Frame')
    plt.ylabel('Identity')

    colors = []
    for track in tracks:
        if isinstance(track, dict):
            frame_ids = track['timestamps']
            track_id = track['id']
        else:
            frame_ids = track.timestamps
            track_id = track.id
        if frame_ids[-1] > last_frame_num:
            frame_ids = [timestamp for timestamp in frame_ids if timestamp < last_frame_num]
        xy[track_id][frame_ids] = track_id
        xx = np.where(xy[track_id] == -1, np.nan, x)
        if track_id >= 0:
            color = COLOR_PALETTE[track_id % len(COLOR_PALETTE)] if track_id >= 0 else (0, 0, 0)
            color = [x / 255 for x in color]
        else:
            color = (0, 0, 0)
        colors.append(tuple(color[::-1]))
        plt.plot(xx, xy[track_id], marker=".", color=colors[-1], label='ID#{}'.format(track_id))
    if save_path:
        file_name = os.path.join(save_path, 'timeline_{}.jpg'.format(plot_name))
        plt.savefig(file_name, bbox_inches='tight')
    if show_online:
        plt.draw()
        plt.pause(0.01)
