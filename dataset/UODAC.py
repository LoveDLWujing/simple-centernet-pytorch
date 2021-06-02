import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET


class UODAC:
    def __init__(self, dataset_dir, mode='val', input_size=(512, 512, 3), stride=4):
        self.input_size = input_size
        self.stride = stride
        self.dataset_dir = dataset_dir
        self.image_dir = os.path.join(dataset_dir, 'image')
        self.xml_dir = os.path.join(dataset_dir, 'box')
        with open(os.path.join(dataset_dir, mode + '.txt'), 'r') as f:
            self.filenames = [line.strip() for line in f.readlines()]
        self.class_names = ["holothurian", "echinus", "scallop", "starfish"]
        self.class_num = len(self.class_names)

    def __getitem__(self, index):
        filename = self.filenames[index]
        image_path = os.path.join(self.image_dir, filename + '.jpg')
        xml_path = os.path.join(self.xml_dir, filename + '.xml')
        img = cv2.imread(image_path)[..., ::-1]
        img_shape = img.shape
        img = cv2.resize(img, self.input_size[:2])
        boxes = self.format_label(xml_path)
        if len(boxes):
            boxes[:, [0, 2]] *= self.input_size[1] / img_shape[1]
            boxes[:, [1, 3]] *= self.input_size[0] / img_shape[0]
        label = self.compute_target_for_network(boxes)
        return img.transpose((2, 0, 1)).astype(np.float32) / 255. - 0.5, label.transpose((2, 0, 1))

    def compute_target_for_network(self, true_boxes):
        feature_shapes = np.array(self.input_size) // self.stride
        mesh_h = np.arange(0., feature_shapes[0], dtype='float32')
        mesh_w = np.arange(0., feature_shapes[1], dtype='float32')
        [meshgrid_x, meshgrid_y] = np.meshgrid(mesh_w, mesh_h)
        y_true = np.zeros((feature_shapes[0], feature_shapes[1], self.class_num + 4), dtype='float32')
        if len(true_boxes):
            true_boxes = np.array(true_boxes, dtype='float32')
            bbox_wh = np.abs((true_boxes[:, [2, 3]] - true_boxes[:, [0, 1]])) / self.stride
            bbox_xy = (true_boxes[:, [2, 3]] + true_boxes[:, [0, 1]]) / (2 * self.stride)
            bbox_xy_floor = np.floor(bbox_xy)
            offset_gt = bbox_xy - bbox_xy_floor
            size_gt = bbox_wh
            for idx in range(len(true_boxes)):
                x, y = int(bbox_xy_floor[idx][0]), int(bbox_xy_floor[idx][1])
                y_true[y, x, -4:-2] = offset_gt[idx]
                y_true[y, x, -2:] = size_gt[idx]
                sigma = self.gaussian_radius(height=bbox_wh[idx][1], width=bbox_wh[idx][0], min_overlap=0.7) / 3
                heatmap = np.exp(-((x - meshgrid_x) ** 2 + (y - meshgrid_y) ** 2) / (2 * sigma ** 2 + 1e-3))
                hm_idx = int(true_boxes[idx, -1])
                y_true[:, :, hm_idx] = np.maximum(heatmap, y_true[:, :, hm_idx])

        return y_true

    def __len__(self):
        return len(self.filenames)

    def format_label(self, xml_path):
        boxes = []
        tree = ET.parse(xml_path)
        root = tree.getroot()
        object_num = 0
        for object in root.findall('object'):
            object_num += 1
            object_name = object.find('name').text
            xmin = max(int(object.find('bndbox').find('xmin').text) - 2, 0)
            ymin = max(int(object.find('bndbox').find('ymin').text) - 2, 0)
            xmax = max(int(object.find('bndbox').find('xmax').text) - 2, 0)
            ymax = max(int(object.find('bndbox').find('ymax').text) - 2, 0)
            if object_name == "waterweeds":
                continue
            boxes.append([xmin, ymin, xmax, ymax, self.class_names.index(object_name)])
        boxes = np.array(boxes, dtype=np.float32)
        return boxes

    @staticmethod
    def gaussian_radius(height, width, min_overlap=0.7):
        a1 = 1
        b1 = (height + width)
        c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
        sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
        r1 = (b1 + sq1) / 2

        a2 = 4
        b2 = 2 * (height + width)
        c2 = (1 - min_overlap) * width * height
        sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
        r2 = (b2 + sq2) / 2

        a3 = 4 * min_overlap
        b3 = -2 * min_overlap * (height + width)
        c3 = (min_overlap - 1) * width * height
        sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
        r3 = (b3 + sq3) / 2

        return min(r1, r2, r3)
