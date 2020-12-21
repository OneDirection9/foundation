# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#
# Modified by: Zhipeng Han
from __future__ import absolute_import, division, print_function

import colorsys
import math
from typing import Optional, Sequence, Tuple, Union

import cv2
import matplotlib as mpl
import matplotlib.colors as mplc
import matplotlib.figure as mplfigure
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg

from .colormap import random_color

__all__ = ["Visualizer"]

ColorVector = Tuple[float, float, float]  # RGB
MPLColor = Union[str, ColorVector]
PointCoordinate = Tuple[float, float]  # XY
BoxCoordinate = Tuple[float, float, float, float]  # XYXY
RotatedBoxCoordinate = Tuple[float, float, float, float, float]  # CtrX,CtrY,W,H,A

_SMALL_OBJECT_AREA_THRESH = 1000
_LARGE_MASK_AREA_THRESH = 12000


class VisImage(object):
    def __init__(self, img: np.ndarray, scale: float = 1.0) -> None:
        """
        Args:
            img (np.ndarray): An RGB image of shape (H, W, 3).
            scale (float): Scale the input image.
        """
        if not isinstance(img, np.ndarray):
            raise TypeError("image should be np.ndarray. Got {}".format(type(img)))
        if img.ndim != 3 or img.shape[-1] != 3:
            raise ValueError("image should be of shape HxWx3. Got {}".format(img.shape))

        self.img = img
        self.scale = scale
        self.height, self.width = img.shape[:2]
        self.fig, self.ax, self.canvas = self._setup_figure()

    def _setup_figure(self) -> Tuple:
        """
        Returns:
            fig (matplotlib.pyplot.figure): Top level container for all the image plot elements.
            ax (matplotlib.pyplot.Axes): Contains figure elements and sets the coordinate system.
        """
        fig = mplfigure.Figure(frameon=False)
        dpi = fig.get_dpi()
        # add a small 1e-2 to avoid precision lost due to matplotlib's truncation
        # (https://github.com/matplotlib/matplotlib/issues/15363)
        fig.set_size_inches(
            (self.width * self.scale + 1e-2) / dpi,
            (self.height * self.scale + 1e-2) / dpi,
        )
        canvas = FigureCanvasAgg(fig)
        # self.canvas = mpl.backends.backend_cairo.FigureCanvasCairo(fig)
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        ax.axis("off")
        # Need to show this first so that other patches can be drawn on top
        ax.imshow(self.img, extent=(0, self.width, self.height, 0), interpolation="nearest")

        return fig, ax, canvas

    def save(self, filepath: str) -> None:
        """
        Args:
            filepath: The absolute path where the visualized image will be saved.
        """
        self.fig.savefig(filepath)

    def get_image(self) -> np.ndarray:
        """
        Returns:
            The visualized image of shape HxWx3 (RGB) in uint8 type. The shape is scaled w.r.t the
            input image using the given `scale` argument.
        """
        canvas = self.canvas
        s, (width, height) = canvas.print_to_buffer()
        # buf = io.BytesIO()  # works for cairo backend
        # canvas.print_rgba(buf)
        # width, height = self.width, self.height
        # s = buf.getvalue()

        buffer = np.frombuffer(s, dtype="uint8")

        img_rgba = buffer.reshape(height, width, 4)
        rgb, alpha = np.split(img_rgba, [3], axis=2)
        return rgb.astype("uint8")


class Visualizer(object):
    """
    Visualizer that draws data about detection/segmentation on images.

    It contains methods like `draw_{text,box,circle,line,binary_mask,polygon}` that draw primitive
    objects to images. Implementing high-level wrappers to draw custom data in some pre-defined
    style.

    This visualizer focuses on high rendering quality rather than performance. It is not
    designed to be used for real-time applications.
    """

    def __init__(self, img_rgb: np.ndarray, scale: float = 1.0):
        """
        Args:
            img_rgb: An RGB image with shape HxWx3.
            scale: Scale the input image.
        """
        self.img = np.asarray(img_rgb).clip(0, 255).astype(np.uint8)
        self.output = VisImage(self.img, scale=scale)

        # too small texts are useless, therefore clamp to 9
        self._default_font_size = max(
            np.sqrt(self.output.height * self.output.width) // 90, 10 // scale
        )

    """
    Primitive drawing functions:
    """

    def draw_text(
        self,
        text: str,
        position: PointCoordinate,
        *,
        color: MPLColor = "g",
        font_size: Optional[int] = None,
        horizontal_alignment: str = "center",
        rotation: float = 0.0,
    ) -> VisImage:
        """
        Args:
            text: Class label.
            position: A tuple of the x and y coordinates to place text on image.
            color: Color of the text. Refer to `matplotlib.colors` for full list of formats that are
                accepted.
            font_size: Font of the text. If not provided, a font size proportional to the image
                width is calculated and used.
            horizontal_alignment: See `matplotlib.text.Text`
            rotation: Rotation angle in degrees CCW.

        Returns:
            output (VisImage): Image object with text drawn.
        """
        if not font_size:
            font_size = self._default_font_size

        # since the text background is dark, we don't want the text to be dark
        color = np.maximum(list(mplc.to_rgb(color)), 0.2)
        color[np.argmax(color)] = max(0.8, np.max(color))

        x, y = position
        self.output.ax.text(
            x,
            y,
            text,
            size=font_size * self.output.scale,
            family="sans-serif",
            bbox={"facecolor": "black", "alpha": 0.8, "pad": 0.7, "edgecolor": "none"},
            verticalalignment="top",
            horizontalalignment=horizontal_alignment,
            color=color,
            zorder=10,
            rotation=rotation,
        )
        return self.output

    def draw_box(
        self,
        box_coord: BoxCoordinate,
        *,
        alpha: float = 0.5,
        edge_color: MPLColor = "g",
        line_style: str = "-",
        label: Optional[str] = None,
    ) -> VisImage:
        """
        Args:
            box_coord: A tuple containing x0, y0, x1, y1 coordinates, where x0 and y0 are the
                coordinates of the image's top left corner. x1 and y1 are the coordinates of the
                image's bottom right corner.
            alpha: Blending efficient. Smaller values lead to more transparent masks.
            edge_color: Color of the outline of the box. Refer to `matplotlib.colors` for full list
                of formats that are accepted.
            line_style: The string to use to create the outline of the boxes.
            label: Label for box. It will not be rendered when set to None.

        Returns:
            output (VisImage): Image object with box drawn.
        """
        x0, y0, x1, y1 = box_coord
        width = x1 - x0
        height = y1 - y0

        linewidth = max(self._default_font_size / 4, 1)

        self.output.ax.add_patch(
            mpl.patches.Rectangle(
                (x0, y0),
                width,
                height,
                fill=False,
                edgecolor=edge_color,
                linewidth=linewidth * self.output.scale,
                alpha=alpha,
                linestyle=line_style,
            )
        )

        if label is not None:
            height_ratio = (y1 - y0) / np.sqrt(self.output.height * self.output.width)
            label_color = self._change_color_brightness(edge_color, brightness_factor=0.7)
            font_size = (
                np.clip((height_ratio - 0.02) / 0.08 + 1, 1.2, 2) * 0.5 * self._default_font_size
            )

            self.draw_text(
                label,
                (x0, y0),
                color=label_color,
                horizontal_alignment="left",
                font_size=font_size,
            )
        return self.output

    def draw_rotated_box(
        self,
        rotated_box: RotatedBoxCoordinate,
        *,
        edge_color: MPLColor = "g",
        line_style: str = "-",
        label: Optional[str] = None,
    ) -> VisImage:
        """
        Args:
            rotated_box: A tuple containing (ctr_x, ctr_y, w, h, angle), where ctr_x and ctr_y are
                the center coordinates of the box. w and h are the width and height of the box.
                angle represents how many degrees the box is rotated CCW with regard to the 0-degree
                box.
            edge_color: Color of the outline of the box. Refer to `matplotlib.colors` for full list
                of formats that are accepted.
            line_style: The string to use to create the outline of the boxes.
            label: Label for rotated box. It will not be rendered when set to None.

        Returns:
            output (VisImage): Image object with box drawn.
        """
        ctr_x, ctr_y, w, h, angle = rotated_box
        area = w * h
        # use thinner lines when the box is small
        linewidth = self._default_font_size / (
            6 if area < _SMALL_OBJECT_AREA_THRESH * self.output.scale else 3
        )

        theta = angle * math.pi / 180.0
        c = math.cos(theta)
        s = math.sin(theta)
        rect = [(-w / 2, h / 2), (-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2)]
        # x: left->right ; y: top->down
        rotated_rect = [(s * yy + c * xx + ctr_x, c * yy - s * xx + ctr_y) for (xx, yy) in rect]
        for k in range(4):
            j = (k + 1) % 4
            self.draw_line(
                [rotated_rect[k][0], rotated_rect[j][0]],
                [rotated_rect[k][1], rotated_rect[j][1]],
                color=edge_color,
                linestyle="--" if k == 1 else line_style,
                linewidth=linewidth,
            )

        if label is not None:
            text_pos = rotated_rect[1]  # top-left corner

            height_ratio = h / np.sqrt(self.output.height * self.output.width)
            label_color = self._change_color_brightness(edge_color, brightness_factor=0.7)
            font_size = (
                np.clip((height_ratio - 0.02) / 0.08 + 1, 1.2, 2) * 0.5 * self._default_font_size
            )
            self.draw_text(label, text_pos, color=label_color, font_size=font_size, rotation=angle)

        return self.output

    def draw_circle(
        self, circle_coord: PointCoordinate, color: MPLColor, radius: int = 3
    ) -> VisImage:
        """
        Args:
            circle_coord: A tuple of x and y coordinates of the center of the circle.
            color: Color of the polygon. Refer to `matplotlib.colors` for a full list of formats
                that are accepted.
            radius: Radius of the circle.

        Returns:
            output (VisImage): Image object with circle drawn.
        """
        self.output.ax.add_patch(
            mpl.patches.Circle(circle_coord, radius=radius, fill=True, color=color)
        )
        return self.output

    def draw_line(
        self,
        x_data: Sequence[float],
        y_data: Sequence[float],
        color: MPLColor,
        linestyle: str = "-",
        linewidth: Optional[float] = None,
    ) -> VisImage:
        """
        Args:
            x_data: A list containing x values of all the points being drawn. Length of list should
                match the length of y_data.
            y_data: A list containing y values of all the points being drawn. Length of list should
                match the length of x_data.
            color: Color of the line. Refer to `matplotlib.colors` for a full list of formats that
                are accepted.
            linestyle: Style of the line. Refer to `matplotlib.lines.Line2D` for a full list of
                formats that are accepted.
            linewidth: Width of the line. When it's None, a default value will be computed and used.

        Returns:
            output (VisImage): Image object with line drawn.
        """
        if linewidth is None:
            linewidth = self._default_font_size / 3
        linewidth = max(linewidth, 1)
        self.output.ax.add_line(
            mpl.lines.Line2D(
                x_data,
                y_data,
                linewidth=linewidth * self.output.scale,
                color=color,
                linestyle=linestyle,
            )
        )
        return self.output

    def draw_binary_mask(
        self,
        binary_mask: np.ndarray,
        *,
        color: Optional[MPLColor] = None,
        alpha: float = 0.5,
        label: Optional[str] = None,
    ) -> VisImage:
        """
        Args:
            binary_mask: Numpy array of shape HxW, where H is the image height and W is the image
                width. Each value in the array is either a 0 or 1 value of uint8 type.
            color: Color of the mask. Refer to `matplotlib.colors` for a full list of formats that
                are accepted. If None, will pick a random color.
            alpha: Blending efficient. Smaller values lead to more transparent masks.
            label: If provided, will be drawn in the object's center of mass.

        Returns:
            output (VisImage): Image object with mask drawn.
        """
        if not isinstance(binary_mask, np.ndarray):
            raise TypeError("binary_mask should be np.ndarray. Got {}".format(type(binary_mask)))
        if binary_mask.ndim != 2:
            raise ValueError("binary_mask should be 2D. Got {}D".format(binary_mask.ndim))

        if color is None:
            color = random_color(rgb=True, maximum=1)

        binary_mask = binary_mask.astype("uint8")  # opencv needs uint8
        shape2d = (binary_mask.shape[0], binary_mask.shape[1])

        rgba = np.zeros(shape2d + (4,), dtype="float32")
        rgba[:, :, :3] = color
        rgba[:, :, 3] = (binary_mask == 1).astype("float32") * alpha
        self.output.ax.imshow(rgba)

        if label is not None:
            # TODO sometimes drawn on wrong objects. the heuristics here can improve.
            lighter_color = self._change_color_brightness(color, brightness_factor=0.7)
            _num_cc, cc_labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, 8)
            largest_component_id = np.argmax(stats[1:, -1]) + 1

            # draw text on the largest component, as well as other very large components.
            for cid in range(1, _num_cc):
                if cid == largest_component_id or stats[cid, -1] > _LARGE_MASK_AREA_THRESH:
                    # median is more stable than centroid
                    # center = centroids[largest_component_id]
                    center = np.median((cc_labels == cid).nonzero(), axis=1)[::-1]
                    self.draw_text(label, center, color=lighter_color)
        return self.output

    def draw_polygon(
        self,
        segment: np.ndarray,
        color: Optional[MPLColor] = None,
        edge_color: Optional[MPLColor] = None,
        alpha: float = 0.5,
    ) -> VisImage:
        """
        Args:
            segment: A numpy array of shape Nx2, containing all the points in the polygon.
            color: Color of the polygon. Refer to `matplotlib.colors` for a full list of formats
                that are accepted.
            edge_color: Color of the polygon edges. Refer to `matplotlib.colors` for a full list of
                formats that are accepted. If not provided, a darker shade of the polygon color will
                be used instead.
            alpha: Blending efficient. Smaller values lead to more transparent masks.

        Returns:
            output (VisImage): Image object with polygon drawn.
        """
        if not isinstance(segment, np.ndarray):
            raise TypeError("segment should be np.ndarray. Got {}".format(type(segment)))
        if segment.ndim != 2 or segment.shape[1] != 2:
            raise ValueError("segment should be of shape Nx2. Got {}".format(segment.shape))

        if color is None:
            color = random_color(rgb=True, maximum=1)

        if edge_color is None:
            # make edge color darker than the polygon color
            if alpha > 0.8:
                edge_color = self._change_color_brightness(color, brightness_factor=-0.7)
            else:
                edge_color = color
        edge_color = mplc.to_rgb(edge_color) + (1,)

        self.output.ax.add_patch(
            mpl.patches.Polygon(
                segment,
                fill=True,
                facecolor=mplc.to_rgb(color) + (alpha,),
                edgecolor=edge_color,
                linewidth=max(self._default_font_size // 15 * self.output.scale, 1),
            )
        )
        return self.output

    """
    Internal methods:
    """

    def _jitter(self, color: MPLColor) -> ColorVector:
        """
        Randomly modifies given color to produce a slightly different color than the color given.

        Args:
            color: The original color. Refer to `matplotlib.colors` for a full list of formats that
                are accepted.

        Returns:
            A tuple of 3 elements, containing the RGB values of the color after being jittered. The
            values in the list are in the [0.0, 1.0] range.
        """
        color = mplc.to_rgb(color)
        vec = np.random.rand(3)
        # better to do it in another color space
        vec = vec / np.linalg.norm(vec) * 0.5
        res = np.clip(vec + color, 0, 1)
        return tuple(res)

    def _create_grayscale_image(self, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Create a grayscale version of the original image.

        Args:
            mask: The array with dtype np.bool. If provided, the colors in masked area will be kept.
        """
        img_bw = self.img.astype("f4").mean(axis=2)
        img_bw = np.stack([img_bw] * 3, axis=2)
        if mask is not None:
            img_bw[mask] = self.img[mask]
        return img_bw

    def _change_color_brightness(self, color: MPLColor, brightness_factor: float) -> ColorVector:
        """
        Depending on the brightness_factor, gives a lighter or darker color i.e. a color with
        less or more saturation than the original color.

        Args:
            color: Color of the polygon. Refer to `matplotlib.colors` for a full list of formats
                that are accepted.
            brightness_factor: A value in [-1.0, 1.0] range. A lightness factor of 0 will correspond
                to no change, a factor in [-1.0, 0) range will result in a darker color and a factor
                in (0, 1.0] range will result in a lighter color.

        Returns:
            modified_color: A tuple containing the RGB values of the modified color. Each value in
                the tuple is in the [0.0, 1.0] range.
        """
        if not -1.0 <= brightness_factor <= 1.0:
            raise ValueError("brightness_factor should be in range [-1, 1]")

        color = mplc.to_rgb(color)
        polygon_color = colorsys.rgb_to_hls(*mplc.to_rgb(color))
        modified_lightness = polygon_color[1] + (brightness_factor * polygon_color[1])
        modified_lightness = 0.0 if modified_lightness < 0.0 else modified_lightness
        modified_lightness = 1.0 if modified_lightness > 1.0 else modified_lightness
        modified_color = colorsys.hls_to_rgb(polygon_color[0], modified_lightness, polygon_color[2])
        return modified_color

    def get_output(self) -> VisImage:
        """
        Returns:
            output (VisImage): The image output containing the visualizations added to the image.
        """
        return self.output
