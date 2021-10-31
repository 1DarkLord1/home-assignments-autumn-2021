#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

import logging

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)8s] %(module)s: %(message)s (%(filename)s:%(lineno)4s)',
    datefmt='%Y-%m-%d %H:%M:%S',
)

from functools import cmp_to_key

from collections import namedtuple

from typing import List, Optional, Tuple, Dict

import numpy as np
import sortednp as snp
import cv2.cv2 as cv2

from corners import (
    CornerStorage,
    FrameCorners
)
from data3d import CameraParameters, PointCloud, Pose
import frameseq
from pims import FramesSequence
from _camtrack import (
    PointCloudBuilder,
    create_cli,
    calc_point_cloud_colors,
    pose_to_view_mat3x4,
    to_opencv_camera_mat3x3,
    triangulate_correspondences,
    TriangulationParameters,
    compute_reprojection_errors,
    build_correspondences,
    Correspondences,
    rodrigues_and_translation_to_view_mat3x4,
    view_mat3x4_to_pose
)


def build_3d_2d_correspondences(
        points_3d: Dict,
        corners: FrameCorners,
) -> Correspondences:
    assert points_3d

    points_3d_ids = np.sort(np.array(list(points_3d.keys())))
    common_ids, (_, indices_2d) = snp.intersect(
        points_3d_ids,
        np.sort(corners.ids.flatten()),
        indices=True
    )

    return Correspondences(
        common_ids,
        np.array([points_3d[x] for x in common_ids]),
        corners.points[indices_2d],
    )


FramePnPInformation = namedtuple(
    'FramePnPInformation',
    ('frame', 'inliers_ratio', 'inliers', 'corresps', 'r_vec_0', 't_vec_0')
)


class CameraTracker:
    def __init__(
            self,
            rgb_sequence: FramesSequence,
            corner_storage: CornerStorage,
            intrinsic_mat: np.ndarray,
            known_view_1: Optional[Tuple[int, Pose]],
            known_view_2: Optional[Tuple[int, Pose]],
            triang_parameters: TriangulationParameters,
            other_parameters: Dict,
    ):
        self._rgb_sequence = rgb_sequence
        self._corner_storage = corner_storage
        self._intrinsic_mat = intrinsic_mat
        self._known_view_1 = known_view_1
        self._known_view_2 = known_view_2
        self._triang_parameters = triang_parameters
        self._other_parameters = other_parameters
        self._frames_count = len(corner_storage)
        self._poses = np.empty(self._frames_count, dtype=np.ndarray)
        self._points_3d = {}
        self._handled_frames = np.zeros(self._frames_count, dtype=bool)

    def _calc_new_points_3d(
            self,
            corners_1: FrameCorners,
            corners_2: FrameCorners,
            view_mat_1: np.ndarray,
            view_mat_2: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        correspondences = build_correspondences(corners_1, corners_2)
        points_3d, corresp_ids, median_cos = triangulate_correspondences(
            correspondences,
            view_mat_1,
            view_mat_2,
            self._intrinsic_mat,
            self._triang_parameters,
        )

        return points_3d, corresp_ids, median_cos

    def _pnp_ransac(
            self,
            points_3d: np.ndarray,
            points_2d: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        pnp_ransac_params = self._other_parameters['PnPRansacParams']

        success, r_vec_0, t_vec_0, inliers = cv2.solvePnPRansac(
            points_3d,
            points_2d,
            self._intrinsic_mat,
            np.array([]),
            **pnp_ransac_params
        )

        if not success:
            raise ValueError('Bad 3d-2d correspondences for PnP')

        return r_vec_0, t_vec_0, inliers

    def _get_camera_pose(
            self,
            points_3d: np.ndarray,
            points_2d: np.ndarray,
            r_vec_0: np.ndarray,
            t_vec_0: np.ndarray,
    ) -> np.ndarray:
        success, r_vec, t_vec = cv2.solvePnP(
            points_3d,
            points_2d,
            self._intrinsic_mat,
            np.array([]),
            r_vec_0,
            t_vec_0,
            True,
            cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            raise ValueError('Bad initial data for PnP')

        return rodrigues_and_translation_to_view_mat3x4(r_vec, t_vec)

    def _handle_best_frame_for_pnp(self) -> Optional[int]:
        """
        Для поиска лучшего кадра для PnP берется метрика (число инлаеров) / (число 3d-2d соответствий),
        число инлаеров определяется с помощью solvePnPRansac
        """
        best_frame = None

        for i, corners in enumerate(self._corner_storage):
            if self._handled_frames[i]:
                continue

            corresps = build_3d_2d_correspondences(
                self._points_3d,
                corners,
            )

            corresp_count = corresps.ids.shape[0]

            assert corresp_count >= 3

            r_vec_0, t_vec_0, inliers = self._pnp_ransac(
                corresps.points_1,
                corresps.points_2,
            )

            inliers = inliers.flatten()

            inliers_ratio = inliers.shape[0] / corresp_count

            frame_pnp_information = FramePnPInformation(
                i, inliers_ratio, inliers, corresps, r_vec_0, t_vec_0,
            )

            if best_frame is None or (
                    best_frame.inliers_ratio < frame_pnp_information.inliers_ratio
            ):
                best_frame = frame_pnp_information

        if best_frame is None:
            return None

        i = best_frame.frame
        inliers = best_frame.inliers
        r_vec_0, t_vec_0 = best_frame.r_vec_0, best_frame.t_vec_0
        corresps = best_frame.corresps

        self._poses[i] = self._get_camera_pose(
            corresps.points_1[inliers],
            corresps.points_2[inliers],
            r_vec_0,
            t_vec_0,
        )

        logger.info(f'Current best frame: number={i}, inliers count={inliers.shape[0]}')

        self._handled_frames[i] = True

        return i

    def _calc_mean_reprojection_error(self, point_3d, point_id, top_corners) -> float:
        error, total = 0, 0

        for i, corners in top_corners:
            id_pos = np.where(corners.ids.flatten() == point_id)[0]

            if len(id_pos) == 0:
                continue

            point_2d = corners.points[id_pos]
            proj_mat = self._intrinsic_mat @ self._poses[i]
            repr_error = compute_reprojection_errors(
                np.array([point_3d]),
                point_2d,
                proj_mat,
            )

            error += repr_error[0]

        return error / len(top_corners)

    def _enrich_point_cloud_smart(self, frame_1: int, corners_1: FrameCorners) -> None:
        """
        Эвристика нахождения лучших 3d точек облака. Получаем наборы точек, применяя триангуляцию к
        уже обсчитанным кадрам и frame_1, выбираем k из них с наименьшими median_cos, соединяем наборы и для
        каждого id выбираем 3d точку с минимальной средней ошибкой репроекции по всем кадрам. Работает очень долго.
        """
        k = 10
        point_3d_candidates = {}
        triang_results = []

        for frame_2, corners_2 in enumerate(self._corner_storage):
            if frame_1 == frame_2 or not self._handled_frames[frame_2]:
                continue

            new_points_3d, corresp_ids, median_cos = \
                self._calc_new_points_3d(
                    corners_1,
                    corners_2,
                    self._poses[frame_1],
                    self._poses[frame_2],
                )

            triang_results.append(
                (
                    median_cos,
                    corresp_ids,
                    new_points_3d,
                    frame_2,
                    corners_2
                )
            )

        def cmp(x, y):
            return x[0] < y[0]

        triang_results = sorted(triang_results, key=cmp_to_key(cmp))[:k]
        top_corners = [(result[3], result[4]) for result in triang_results]

        for _, corresp_ids, new_points_3d, _, _ in triang_results:
            for point_3d, point_id in zip(new_points_3d, corresp_ids):
                if point_id not in point_3d_candidates:
                    point_3d_candidates[point_id] = []

                point_3d_candidates[point_id].append(point_3d)

        for point_id, points_3d in point_3d_candidates.items():
            if point_id in self._points_3d:
                points_3d.append(self._points_3d[point_id])

            mean_errors = [
                self._calc_mean_reprojection_error(point_3d, point_id, top_corners)
                for point_3d in points_3d
            ]

            opt_point_3d = points_3d[np.argmin(mean_errors)]
            self._points_3d[point_id] = opt_point_3d

    def _enrich_point_cloud_stupid(self, frame_1: int, corners_1: FrameCorners) -> None:
        """
        Простая эвристика дополнения облака 3d точек.
        Находим кадр с минимальным median_cos относительно кадра frame_1 (т.е. с минимальным углом)
        и дополняем облако точками, получившимися в результате триангуляции этого кадра и frame_1.
        """
        best_cos, best_points_3d, best_ids = None, None, None

        for frame_2, corners_2 in enumerate(self._corner_storage):
            if frame_1 == frame_2 or not self._handled_frames[frame_2]:
                continue

            new_points_3d, corresp_ids, median_cos = \
                self._calc_new_points_3d(
                    corners_1,
                    corners_2,
                    self._poses[frame_1],
                    self._poses[frame_2],
                )

            if best_cos is None or median_cos < best_cos:
                best_cos = median_cos
                best_points_3d = new_points_3d
                best_ids = corresp_ids

        for point_3d, point_id in zip(best_points_3d, best_ids):
            self._points_3d[point_id] = point_3d

        logger.info(f'Best cos for triangulation: {best_cos}')

    def _init_points_3d(self) -> None:
        frame_1, frame_2 = self._known_view_1[0], self._known_view_2[0]
        view_mat_1 = pose_to_view_mat3x4(self._known_view_1[1])
        view_mat_2 = pose_to_view_mat3x4(self._known_view_2[1])

        points_3d, corresp_ids, median_cos = self._calc_new_points_3d(
            self._corner_storage[frame_1],
            self._corner_storage[frame_2],
            view_mat_1,
            view_mat_2,
        )

        for point_3d, point_id in zip(points_3d, corresp_ids):
            self._points_3d[point_id] = point_3d

        self._handled_frames[frame_1] = True
        self._handled_frames[frame_2] = True
        self._poses[frame_1] = view_mat_1
        self._poses[frame_2] = view_mat_2

    def camera_tracking(self) -> Tuple[List[Pose], PointCloud]:
        self._init_points_3d()

        while True:
            logger.info(f'Point cloud size: {len(self._points_3d)}')

            best_frame = self._handle_best_frame_for_pnp()

            if best_frame is None:
                break

            self._enrich_point_cloud_stupid(best_frame, self._corner_storage[best_frame])

        ids = np.array(list(self._points_3d.keys()))
        points = np.array(list(self._points_3d.values()))

        point_cloud_builder = PointCloudBuilder(ids, points)

        calc_point_cloud_colors(
            point_cloud_builder,
            self._rgb_sequence,
            list(self._poses),
            self._intrinsic_mat,
            self._corner_storage,
            self._other_parameters['maxReprojError'],
        )

        point_cloud = point_cloud_builder.build_point_cloud()

        return list(map(view_mat3x4_to_pose, self._poses)), point_cloud


CAMERA_TRACKER_PARAMETERS = {
    'maxReprojError': 7.5,
    'PnPRansacParams': {
        'flags': cv2.SOLVEPNP_EPNP,
        'iterationsCount': 100,
        'reprojectionError': 7.5,
    },
}

TRIANG_PARAMETERS = TriangulationParameters(7.5, 1.0, 0.1)


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    if known_view_1 is None or known_view_2 is None:
        raise NotImplementedError()

    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    camtracker = CameraTracker(
        rgb_sequence,
        corner_storage,
        intrinsic_mat,
        known_view_1,
        known_view_2,
        TRIANG_PARAMETERS,
        CAMERA_TRACKER_PARAMETERS,
    )
    return camtracker.camera_tracking()


if __name__ == '__main__':
    # pylint:disable=no-value-for-parameter
    create_cli(track_and_calc_colors)()
