# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Dataloaders and dataset utils
"""
import glob
from pathlib import Path
import numpy as np
import time
import re
import contextlib
# Parameters
IMG_FORMATS = ['bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp']  # include image suffixes
VID_FORMATS = ['asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'wmv']  # include video suffixes
FONT = 'Arial.ttf'  # https://ultralytics.com/assets/Arial.ttf


def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        path = Path(f"{path}{sep}{n}{suffix}")  # increment path
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory
    return path


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    # y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=(), max_det=300, nm=0):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """
    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - nm - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    min_wh, max_wh = 2, 7680  # (pixels) minimum and maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    mi = 5 + nc
    # output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    output = [np.zeros((0, 6 + nm))] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height 找到box超出 min和max的行，将其conf置为0，这样后面就可以删除掉
        x = x[xc[xi]]  # confidence 筛选出conf大于阈值的数据

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf 每个类别的分数都乘上conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])
        mask = x[:, mi:]
        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
            x = np.concatenate((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf = np.max(x[:, 5:mi], axis=1).reshape(-1, 1)
            j = np.argmax(x[:, 5:mi], axis=1).reshape(-1, 1)
            x = np.concatenate((box, conf, np.float32(j), mask), 1)[conf.reshape(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            # x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]
            x = x[(x[:, 5:6] == np.array(classes, device=x.device)).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes 如果boxes数量超过了max_nms 则只取0到max_nms的数据
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence
        # else:
        #     x = x[x[:, 4].argsort(descending=True)]  # sort by confidence


        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes 类别索引*max_wh求得每个类别的偏移量
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores 将box加上偏移量，既可以分别对每个类别进行nms
        # i = cv2.dnn.NMSBoxes(boxes, scores, conf_thres, iou_thres)  #cv2.dnn.NMSBoxes得出的结果和官方不一致，因此用了自己写的
        i = numpy_nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output


def clip_boxes(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    # if isinstance(boxes, torch.Tensor):  # faster individually
    if not isinstance(boxes, np.ndarray):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2


def scale_boxes(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_boxes(coords, img0_shape)
    return coords


class Profile(contextlib.ContextDecorator):
    # YOLOv5 Profile class. Usage: @Profile() decorator or 'with Profile():' context manager
    def __init__(self, t=0.0):
        self.t = t
        # self.cuda = torch.cuda.is_available()

    def __enter__(self):
        self.start = self.time()
        return self

    def __exit__(self, type, value, traceback):
        self.dt = self.time() - self.start  # delta-time
        self.t += self.dt  # accumulate dt

    def time(self):
        # if self.cuda:
        #     torch.cuda.synchronize()
        return time.time()


def box_area(boxes: np.ndarray):
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(box1:np.ndarray, box2: np.ndarray):
    area1 = box_area(box1)  # N
    area2 = box_area(box2)  # M
    # broadcasting, 两个数组各维度大小 从后往前对比一致， 或者 有一维度值为1；
    lt = np.maximum(box1[:, np.newaxis, :2], box2[:, :2])
    rb = np.minimum(box1[:, np.newaxis, 2:], box2[:, 2:])
    wh = rb - lt
    wh = np.maximum(0, wh) # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]
    iou = inter / (area1[:, np.newaxis] + area2 - inter)
    return iou


def numpy_nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float):
    idxs = scores.argsort()  # 按分数 降序排列的索引 [N]
    keep = []
    while idxs.size > 0:  # 统计数组中元素的个数
        max_score_index = idxs[-1]
        max_score_box = boxes[max_score_index][None, :]
        keep.append(max_score_index)
        if idxs.size == 1:
            break
        idxs = idxs[:-1]  # 将得分最大框 从索引中删除； 剩余索引对应的框 和 得分最大框 计算IoU;
        other_boxes = boxes[idxs]  # [?, 4]
        ious = box_iou(max_score_box, other_boxes)  # 一个框和其余框比较 1XM
        idxs = idxs[ious[0] <= iou_threshold]
    keep = np.array(keep)
    return keep


def clip_segments(segments, shape):
    # Clip segments (xy1,xy2,...) to image shape (height, width)
    segments[:, 0] = segments[:, 0].clip(0, shape[1])  # x
    segments[:, 1] = segments[:, 1].clip(0, shape[0])  # y


def scale_segments(img1_shape, segments, img0_shape, ratio_pad=None, normalize=False):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    segments[:, 0] -= pad[0]  # x padding
    segments[:, 1] -= pad[1]  # y padding
    segments /= gain
    clip_segments(segments, img0_shape)
    if normalize:
        segments[:, 0] /= img0_shape[1]  # width
        segments[:, 1] /= img0_shape[0]  # height
    return segments


# def get_rotate_crop_image(img, points):
#     """
#     根据坐标点截取图像
#     :param img:
#     :param points:
#     :return:
#     """
#
#     h, w, _ = img.shape
#
#     left = int(np.min(points[:, 0]))
#     right = int(np.max(points[:, 0]))
#     top = int(np.min(points[:, 1]))
#     bottom = int(np.max(points[:, 1]))
#
#     img_crop = img[top:bottom, left:right, :].copy()
#
#     points[:, 0] = points[:, 0] - left
#     points[:, 1] = points[:, 1] - top
#     img_crop_width = int(np.linalg.norm(points[0] - points[1]))
#
#     img_crop_height = int(np.linalg.norm(points[0] - points[3]))
#
#     pts_std = np.float32([[0, 0], [img_crop_width, 0], [img_crop_width, img_crop_height], [0, img_crop_height]])
#
#     M = cv2.getPerspectiveTransform(points, pts_std)
#
#     dst_img = cv2.warpPerspective(
#         img_crop,
#         M, (img_crop_width, img_crop_height),
#         borderMode=cv2.BORDER_REPLICATE)
#     dst_img_height, dst_img_width = dst_img.shape[0:2]
#     if dst_img_height * 1.0 / dst_img_width >= 1:
#         #         pass
#         #         print(dst_img_height * 1.0 / dst_img_width,dst_img_height,dst_img_width,'*-'*10)
#         dst_img = np.rot90(dst_img, -1)  # -1为逆时针，1为顺时针。
#
#     return dst_img
#
#
# def sorted_boxes(dt_boxes):
#     """
#     坐标点排序
#     """
#
#     num_boxes = dt_boxes.shape[0]
#     sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
#     _boxes = list(sorted_boxes)
#
#     for i in range(num_boxes - 1):
#         if abs(_boxes[i + 1][0][1] - _boxes[i][0][1]) < 10 and \
#                 (_boxes[i + 1][0][0] < _boxes[i][0][0]):
#             tmp = _boxes[i]
#             _boxes[i] = _boxes[i + 1]
#             _boxes[i + 1] = tmp
#
#     return _boxes