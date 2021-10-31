#! /usr/bin/env python3

__all__ = [
    'FrameCorners',
    'CornerStorage',
    'build',
    'dump',
    'load',
    'draw',
    'calc_track_interval_mappings',
    'calc_track_len_array_mapping',
    'without_short_tracks'
]

import copy

from collections import namedtuple
from skimage import img_as_ubyte

import click
import cv2.cv2 as cv2
import numpy as np
import pims

from _corners import (
    FrameCorners,
    CornerStorage,
    StorageImpl,
    dump,
    load,
    draw,
    calc_track_interval_mappings,
    calc_track_len_array_mapping,
    without_short_tracks,
    create_cli
)


class _CornerStorageBuilder:

    def __init__(self, progress_indicator=None):
        self._progress_indicator = progress_indicator
        self._corners = dict()

    def set_corners_at_frame(self, frame, corners):
        self._corners[frame] = corners
        if self._progress_indicator is not None:
            self._progress_indicator.update(1)

    def build_corner_storage(self):
        return StorageImpl(item[1] for item in sorted(self._corners.items()))


CornersCollector = namedtuple('Corners', ['ids', 'points', 'sizes'])


def define_params(dims):
    params = {
        'corners': {
            'maxCorners': 400,
            'qualityLevel': 0.003,
            'minDistance': (dims[0] + dims[1]) // 250,
            'blockSize': 7,
        },
        'optFlow': {
            'winSize': (15, 15),
            'maxLevel': 7,
            'criteria': (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                50,
                0.02
            ),
            'minEigThreshold': 0.003,
        },
        'common': {
            'cornerSize': (dims[0] + dims[1]) // 400,
            'borderPadding': 15,
        }
    }
    return params


def create_empty_corners_collector():
    return CornersCollector(
        np.array([]),
        np.array([]).reshape((-1, 1, 2)),
        np.array([])
    )


def to_frame_corners(corners):
    return FrameCorners(
        corners.ids.astype(np.int0),
        corners.points,
        corners.sizes
    )


class CornersTracker:
    def __init__(self, frame, frames_count, params):
        self.fresh_id = 1
        self.idx = 0
        self.shape = frame.shape
        self.frames_count = frames_count
        self.params = params
        self.frame = frame
        self.corners = create_empty_corners_collector()

    def get_new_ids(self, count):
        old_fresh_id = self.fresh_id
        self.fresh_id += count
        return np.arange(old_fresh_id, self.fresh_id)

    def get_corner_sizes(self, count):
        corner_size = self.params['common']['cornerSize']
        sizes = np.empty(count)
        sizes.fill(corner_size)
        return sizes

    def append_corners(
            self,
            corner_ids,
            corner_points,
            corner_sizes,
    ):
        self.corners = CornersCollector(
            np.append(self.corners.ids, corner_ids, axis=0),
            np.append(self.corners.points, corner_points, axis=0),
            np.append(self.corners.sizes, corner_sizes, axis=0)
        )

    def add_new_corners(
        self,
        corner_points
    ):
        corners_count = corner_points.shape[0]
        corner_ids = self.get_new_ids(corners_count)
        corner_sizes = self.get_corner_sizes(corners_count)
        self.append_corners(
            corner_ids,
            corner_points,
            corner_sizes
        )

    def filter_points(self, state, points):
        padding = self.params['common']['borderPadding']
        for i, point in enumerate(points):
            x, y = point[0]
            if (
                x < padding or self.shape[1] - x <= padding
            ) or (
                y < padding or self.shape[0] - y <= padding
            ):
                state[i] = 0

    def get_corners(self, mask=None):
        return cv2.goodFeaturesToTrack(
            self.frame,
            mask=mask if mask is None else mask.astype(np.uint8),
            **self.params['corners'],
        )

    def acquire_optical_flow(self, old_frame, old_corners):
        next_points, state, _ = cv2.calcOpticalFlowPyrLK(
            img_as_ubyte(old_frame),
            img_as_ubyte(self.frame),
            np.float32(old_corners.points),
            None,
            **self.params['optFlow']
        )
        state = state.flatten()
        return next_points, state

    def build_shade_mask(self, points):
        dist = self.params['corners']['minDistance']
        mask = np.ones(self.shape)
        spot = np.zeros((2 * dist - 1, 2 * dist - 1))
        for point in points:
            x, y = np.int0(point[0])
            x1 = max(x - dist + 1, 0)
            x2 = min(x + dist, self.shape[1])
            y1 = max(y - dist + 1, 0)
            y2 = min(y + dist, self.shape[0])
            mask[y1:y2, x1:x2] = spot[:y2 - y1, :x2 - x1]
        return mask

    def handle_first_frame(self):
        corner_points = self.get_corners()
        state = np.ones(corner_points.shape[0])
        self.filter_points(state, corner_points)
        self.add_new_corners(corner_points[state == 1])

    def save_corners(self, corners, builder):
        builder.set_corners_at_frame(
            self.idx,
            to_frame_corners(corners)
        )
        self.idx += 1

    def handle_frame(self, frame, builder):
        prev_corners = copy.deepcopy(self.corners)
        prev_frame = copy.deepcopy(self.frame)
        self.frame = frame
        self.corners = create_empty_corners_collector()

        cur_points, state = self.acquire_optical_flow(prev_frame, prev_corners)
        self.filter_points(state, cur_points)
        prev_good_ids = prev_corners.ids[state == 1]
        cur_good_points = cur_points[state == 1]

        self.append_corners(
            prev_good_ids,
            cur_good_points,
            self.get_corner_sizes(cur_good_points.shape[0])
        )

        prev_good_points = prev_corners.points[state == 1]
        prev_corners = CornersCollector(
            prev_good_ids,
            prev_good_points,
            self.get_corner_sizes(prev_good_points.shape[0])
        )

        shade_mask = self.build_shade_mask(np.round(cur_good_points))
        extra_corner_points = self.get_corners(shade_mask)
        state = np.ones(extra_corner_points.shape[0])
        self.filter_points(state, extra_corner_points)
        self.add_new_corners(extra_corner_points[state == 1])

        self.save_corners(prev_corners, builder)
        if self.idx == self.frames_count - 1:
            self.save_corners(self.corners, builder)


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    frame0 = frame_sequence[0]
    shape = frame0.shape
    params = define_params(shape)
    tracker = CornersTracker(frame0, len(frame_sequence), params)
    tracker.handle_first_frame()
    for frame in frame_sequence[1:]:
        tracker.handle_frame(frame, builder)


def build(frame_sequence: pims.FramesSequence,
          progress: bool = True) -> CornerStorage:
    """
    Build corners for all frames of a frame sequence.

    :param frame_sequence: grayscale float32 frame sequence.
    :param progress: enable/disable building progress bar.
    :return: corners for all frames of given sequence.
    """
    if progress:
        with click.progressbar(length=len(frame_sequence),
                               label='Calculating corners') as progress_bar:
            builder = _CornerStorageBuilder(progress_bar)
            _build_impl(frame_sequence, builder)
    else:
        builder = _CornerStorageBuilder()
        _build_impl(frame_sequence, builder)
    return builder.build_corner_storage()


if __name__ == '__main__':
    create_cli(build)()  # pylint:disable=no-value-for-parameter
