import numpy as np


class AnchorBox:
    def __init__(
        self, input_shape, min_size, max_size=None, aspect_ratios=None, flip=True
    ):
        self.input_shape = input_shape
        self.min_size = min_size
        self.max_size = max_size
        self.aspect_ratios = []
        for ar in aspect_ratios:
            self.aspect_ratios.append(ar)
            self.aspect_ratios.append(1.0 / ar)

    def call(self, layer_shape, mask=None):
        layer_height = layer_shape[0]
        layer_width = layer_shape[1]
        img_height = self.input_shape[0]
        img_width = self.input_shape[1]
        box_widths = []
        box_heights = []
        for ar in self.aspect_ratios:
            if ar == 1 and len(box_widths) == 0:
                box_widths.append(self.min_size)
                box_heights.append(self.min_size)
            elif ar == 1 and len(box_widths) > 0:
                box_widths.append(np.sqrt(self.min_size * self.max_size))
                box_heights.append(np.sqrt(self.min_size * self.max_size))
            elif ar != 1:
                box_widths.append(self.min_size * np.sqrt(ar))
                box_heights.append(self.min_size / np.sqrt(ar))

        box_widths = 0.5 * np.array(box_widths)
        box_heights = 0.5 * np.array(box_heights)
        step_x = img_width / layer_width
        step_y = img_height / layer_height
        linx = np.linspace(0.5 * step_x, img_width - 0.5 * step_x, layer_width)
        liny = np.linspace(0.5 * step_y, img_height - 0.5 * step_y, layer_height)
        centers_x, centers_y = np.meshgrid(linx, liny)
        centers_x = centers_x.reshape(-1, 1)
        centers_y = centers_y.reshape(-1, 1)
        num_anchors_ = len(self.aspect_ratios)
        anchor_boxes = np.concatenate((centers_x, centers_y), axis=1)
        anchor_boxes = np.tile(anchor_boxes, (1, 2 * num_anchors_))
        anchor_boxes[:, ::4] -= box_widths
        anchor_boxes[:, 1::4] -= box_heights
        anchor_boxes[:, 2::4] += box_widths
        anchor_boxes[:, 3::4] += box_heights
        anchor_boxes[:, ::2] /= img_width
        anchor_boxes[:, 1::2] /= img_height
        anchor_boxes = anchor_boxes.reshape(-1, 4)
        anchor_boxes = np.minimum(np.maximum(anchor_boxes, 0.0), 1.0)
        return anchor_boxes


def get_vgg_output_length(height, width):
    filter_sizes = [3, 3, 3, 3, 3, 3, 3, 3]
    padding = [1, 1, 1, 1, 1, 1, 0, 0]
    stride = [2, 2, 2, 2, 2, 2, 1, 1]
    feature_heights = []
    feature_widths = []

    for i in range(len(filter_sizes)):
        height = (height + 2 * padding[i] - filter_sizes[i]) // stride[i] + 1
        width = (width + 2 * padding[i] - filter_sizes[i]) // stride[i] + 1
        feature_heights.append(height)
        feature_widths.append(width)
    return np.array(feature_heights)[-6:], np.array(feature_widths)[-6:]


def get_mobilenet_output_length(height, width):
    filter_sizes = [3, 3, 3, 3, 3, 3, 3, 3, 3]
    padding = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    stride = [2, 2, 2, 2, 2, 2, 2, 2, 2]
    feature_heights = []
    feature_widths = []

    for i in range(len(filter_sizes)):
        height = (height + 2 * padding[i] - filter_sizes[i]) // stride[i] + 1
        width = (width + 2 * padding[i] - filter_sizes[i]) // stride[i] + 1
        feature_heights.append(height)
        feature_widths.append(width)
    return np.array(feature_heights)[-6:], np.array(feature_widths)[-6:]


def get_anchors(
    input_shape=[300, 300],
    anchors_size=[30, 60, 111, 162, 213, 264, 315],
    backbone="vgg",
):
    if backbone == "vgg":
        feature_heights, feature_widths = get_vgg_output_length(
            input_shape[0], input_shape[1]
        )
        aspect_ratios = [[1, 2], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2], [1, 2]]
    else:
        feature_heights, feature_widths = get_mobilenet_output_length(
            input_shape[0], input_shape[1]
        )
        aspect_ratios = [
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3],
        ]

    anchors = []
    for i in range(len(feature_heights)):
        anchor_boxes = AnchorBox(
            input_shape,
            anchors_size[i],
            max_size=anchors_size[i + 1],
            aspect_ratios=aspect_ratios[i],
        ).call([feature_heights[i], feature_widths[i]])
        anchors.append(anchor_boxes)
    anchors = np.concatenate(anchors, axis=0)
    return anchors


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    class AnchorBox_for_Vision:
        def __init__(
            self, input_shape, min_size, max_size=None, aspect_ratios=None, flip=True
        ):
            self.input_shape = input_shape
            self.min_size = min_size
            self.max_size = max_size
            self.aspect_ratios = []
            for ar in aspect_ratios:
                self.aspect_ratios.append(ar)
                self.aspect_ratios.append(1.0 / ar)

        def call(self, layer_shape, mask=None):
            layer_height = layer_shape[0]
            layer_width = layer_shape[1]
            img_height = self.input_shape[0]
            img_width = self.input_shape[1]
            box_widths = []
            box_heights = []
            for ar in self.aspect_ratios:
                if ar == 1 and len(box_widths) == 0:
                    box_widths.append(self.min_size)
                    box_heights.append(self.min_size)
                elif ar == 1 and len(box_widths) > 0:
                    box_widths.append(np.sqrt(self.min_size * self.max_size))
                    box_heights.append(np.sqrt(self.min_size * self.max_size))
                elif ar != 1:
                    box_widths.append(self.min_size * np.sqrt(ar))
                    box_heights.append(self.min_size / np.sqrt(ar))
            print("box_widths:", box_widths)
            print("box_heights:", box_heights)
            box_widths = 0.5 * np.array(box_widths)
            box_heights = 0.5 * np.array(box_heights)
            step_x = img_width / layer_width
            step_y = img_height / layer_height
            linx = np.linspace(0.5 * step_x, img_width - 0.5 * step_x, layer_width)
            liny = np.linspace(0.5 * step_y, img_height - 0.5 * step_y, layer_height)
            centers_x, centers_y = np.meshgrid(linx, liny)
            centers_x = centers_x.reshape(-1, 1)
            centers_y = centers_y.reshape(-1, 1)
            if layer_height == 3:
                fig = plt.figure()
                ax = fig.add_subplot(111)
                plt.ylim(-50, 350)
                plt.xlim(-50, 350)
                plt.scatter(centers_x, centers_y)
            num_anchors_ = len(self.aspect_ratios)
            anchor_boxes = np.concatenate((centers_x, centers_y), axis=1)
            anchor_boxes = np.tile(anchor_boxes, (1, 2 * num_anchors_))
            anchor_boxes[:, ::4] -= box_widths
            anchor_boxes[:, 1::4] -= box_heights
            anchor_boxes[:, 2::4] += box_widths
            anchor_boxes[:, 3::4] += box_heights
            print(np.shape(anchor_boxes))
            if layer_height == 3:
                rect1 = plt.Rectangle(
                    [anchor_boxes[4, 0], anchor_boxes[4, 1]],
                    box_widths[0] * 2,
                    box_heights[0] * 2,
                    color="r",
                    fill=False,
                )
                rect2 = plt.Rectangle(
                    [anchor_boxes[4, 4], anchor_boxes[4, 5]],
                    box_widths[1] * 2,
                    box_heights[1] * 2,
                    color="r",
                    fill=False,
                )
                rect3 = plt.Rectangle(
                    [anchor_boxes[4, 8], anchor_boxes[4, 9]],
                    box_widths[2] * 2,
                    box_heights[2] * 2,
                    color="r",
                    fill=False,
                )
                rect4 = plt.Rectangle(
                    [anchor_boxes[4, 12], anchor_boxes[4, 13]],
                    box_widths[3] * 2,
                    box_heights[3] * 2,
                    color="r",
                    fill=False,
                )
                ax.add_patch(rect1)
                ax.add_patch(rect2)
                ax.add_patch(rect3)
                ax.add_patch(rect4)
                plt.show()
            anchor_boxes[:, ::2] /= img_width
            anchor_boxes[:, 1::2] /= img_height
            anchor_boxes = anchor_boxes.reshape(-1, 4)
            anchor_boxes = np.minimum(np.maximum(anchor_boxes, 0.0), 1.0)
            return anchor_boxes

    input_shape = [300, 300]
    anchors_size = [30, 60, 111, 162, 213, 264, 315]
    feature_heights, feature_widths = get_vgg_output_length(
        input_shape[0], input_shape[1]
    )
    aspect_ratios = [[1, 2], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2], [1, 2]]
    anchors = []
    for i in range(len(feature_heights)):
        anchors.append(
            AnchorBox_for_Vision(
                input_shape,
                anchors_size[i],
                max_size=anchors_size[i + 1],
                aspect_ratios=aspect_ratios[i],
            ).call([feature_heights[i], feature_widths[i]])
        )
    anchors = np.concatenate(anchors, axis=0)
    print(np.shape(anchors))
