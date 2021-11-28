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

from enum import Enum
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
    view_mat3x4_to_pose,
    remove_correspondences_with_ids,
    eye3x4
)


FramePnPInformation = namedtuple(
    'FramePnPInformation',
    ('frame', 'inliers_ratio', 'inliers', 'corresps', 'r_vec_0', 't_vec_0')
)

EssentialMatContext = namedtuple(
    'EssentialMatInformation',
    ('frame_1', 'frame_2', 'E', 'correspondences')
)

ViewMatsContext = namedtuple(
    'EssentialMatInformation',
    ('frame_1', 'frame_2', 'view_mat_1', 'view_mat_2')
)


class BestFrameState(Enum):
    success = 'success'
    not_found = 'not_found'
    bad_frame_found = 'bad_frame_found'


class CameraTracker:
    def __init__(
            self,
            rgb_sequence: FramesSequence,
            corner_storage: CornerStorage,
            intrinsic_mat: np.ndarray,
            triang_parameters: TriangulationParameters,
            other_parameters: Dict,
            known_view_1: Optional[Tuple[int, Pose]] = None,
            known_view_2: Optional[Tuple[int, Pose]] = None,
            init_view_mats_parameters: Optional[Dict] = None,
    ):
        self._rgb_sequence = rgb_sequence
        self._corner_storage = corner_storage
        self._intrinsic_mat = intrinsic_mat
        self._known_view_1 = known_view_1
        self._known_view_2 = known_view_2
        self._triang_parameters = triang_parameters
        self._other_parameters = other_parameters
        self._init_view_mats_parameters = init_view_mats_parameters

        if (known_view_1 is None or known_view_2 is None) and init_view_mats_parameters is None:
            raise ValueError('init_view_mats_parameters argument is None')

        self._frames_count = len(corner_storage)
        self._poses = np.empty(self._frames_count, dtype=np.ndarray)
        self._points_3d = {}
        self._handled_frames = np.zeros(self._frames_count, dtype=bool)
        self._bad_frames = np.zeros(self._frames_count, dtype=bool)

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

    def _build_3d_2d_correspondences(
            self,
            points_3d_ids: np.ndarray,
            corners: FrameCorners,
    ) -> Correspondences:
        common_ids, (_, indices_2d) = snp.intersect(
            points_3d_ids,
            np.sort(corners.ids.flatten()),
            indices=True
        )

        return Correspondences(
            common_ids,
            np.array([self._points_3d[x] for x in common_ids]),
            corners.points[indices_2d],
        )

    def _handle_best_frame_for_pnp(self) -> Tuple[BestFrameState, int]:
        """
        Для поиска лучшего кадра для PnP берется метрика (число инлаеров) / (число 3d-2d соответствий),
        число инлаеров определяется с помощью solvePnPRansac
        """
        best_frame = None
        points_3d_ids = np.sort(np.array(list(self._points_3d.keys())))

        # max_cnt = self._other_parameters['points3dCntForBestFrame']
        # points_3d_ids_subset = np.sort(
        #     points_3d_ids[
        #         np.random.choice(
        #             points_3d_ids.shape[0],
        #             min(points_3d_ids.shape[0], max_cnt),
        #             replace=False
        #         )
        #     ]
        # )

        for i, corners in enumerate(self._corner_storage):
            if self._handled_frames[i]:
                continue

            corresps = self._build_3d_2d_correspondences(
                points_3d_ids,
                corners,
            )

            corresp_count = corresps.ids.shape[0]

            if corresp_count < 4:
                continue

            try:
                r_vec_0, t_vec_0, inliers = self._pnp_ransac(
                    corresps.points_1,
                    corresps.points_2,
                )
            except ValueError:
                continue

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
            return BestFrameState.not_found, -1

        i = best_frame.frame
        inliers = best_frame.inliers
        r_vec_0, t_vec_0 = best_frame.r_vec_0, best_frame.t_vec_0
        corresps = best_frame.corresps

        try:
            self._poses[i] = self._get_camera_pose(
                corresps.points_1[inliers],
                corresps.points_2[inliers],
                r_vec_0,
                t_vec_0,
            )
        except ValueError:
            return BestFrameState.bad_frame_found, i

        logger.info(f'Current best frame: number={i}, inliers count={inliers.shape[0]}')

        self._handled_frames[i] = True

        return BestFrameState.success, i

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
            if frame_1 == frame_2 or not self._handled_frames[frame_2] or self._bad_frames[frame_2]:
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
            if frame_1 == frame_2 or not self._handled_frames[frame_2] or self._bad_frames[frame_2]:
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
        if self._known_view_1 is None or self._known_view_2 is None:
            view_mats = self.init_view_mats()
            self._known_view_1 = (view_mats.frame_1, view_mats.view_mat_1)
            self._known_view_2 = (view_mats.frame_2, view_mats.view_mat_2)

        self._init_points_3d()

        while True:
            logger.info(f'Point cloud size: {len(self._points_3d)}')

            state, best_frame = self._handle_best_frame_for_pnp()

            if state is BestFrameState.not_found:
                break

            if state is BestFrameState.bad_frame_found:
                self._bad_frames[best_frame] = True
                self._handled_frames[best_frame] = True
                break

            self._enrich_point_cloud_stupid(best_frame, self._corner_storage[best_frame])

        self._poses[self._handled_frames ^ True] = None
        self._poses[self._bad_frames] = None

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

    def _find_essent_mat(
            self,
            corners_1: FrameCorners,
            corners_2: FrameCorners,
    ) -> Tuple[np.ndarray, Correspondences, float]:
        correspondences = build_correspondences(corners_1, corners_2)
        E, essent_mat_inliers = cv2.findEssentialMat(
            correspondences.points_1,
            correspondences.points_2,
            method=cv2.RANSAC,
            cameraMatrix=self._intrinsic_mat,
            **self._init_view_mats_parameters['findEssentialMatrixParams'],
        )

        correspondences = remove_correspondences_with_ids(
            correspondences,
            np.argwhere(essent_mat_inliers.flatten() == 0),
        )

        _, hom_mat_inliers = cv2.findHomography(
            correspondences.points_1,
            correspondences.points_2,
            method=cv2.RANSAC,
            **self._init_view_mats_parameters['findHomographyParams'],
        )

        return E, correspondences, np.sum(hom_mat_inliers) / np.sum(essent_mat_inliers)

    def _find_initial_view_mats(
            self,
            correspondences: Correspondences,
            essential_mat: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        R_1, R_2, t = cv2.decomposeEssentialMat(essential_mat)

        view_mats = [
            np.hstack((R_1, t)),
            np.hstack((R_2, t)),
            np.hstack((R_1, -t)),
            np.hstack((R_2, -t))
        ]

        view_mat_1 = eye3x4()
        best_view_mat = None
        best_points_3d_cnt = None

        for view_mat_2 in view_mats:
            points_3d, _, _ = triangulate_correspondences(
                correspondences,
                view_mat_1,
                view_mat_2,
                self._intrinsic_mat,
                self._triang_parameters,
            )

            if best_view_mat is None or points_3d.shape[0] > best_points_3d_cnt:
                best_view_mat = view_mat_2
                best_points_3d_cnt = points_3d.shape[0]

        return view_mat_1, best_view_mat, best_points_3d_cnt

    def init_view_mats(self) -> ViewMatsContext:
        best_inliers_ratio = None
        best_essent_mat_context = None

        best_points_3d_cnt = None
        best_view_mats_context = None

        for frame_1, corners_1 in enumerate(self._corner_storage):
            for frame_2, corners_2 in enumerate(self._corner_storage):
                if frame_2 <= frame_1:
                    continue

                if frame_2 - frame_1 > self._init_view_mats_parameters['framesWindowSize']:
                    break

                logger.info(f'Handling pair ({frame_1}, {frame_2}) for initial camera positions')

                E, correspondences, inliers_ratio = self._find_essent_mat(
                    corners_1,
                    corners_2,
                )

                if best_essent_mat_context is None or inliers_ratio < best_inliers_ratio:
                    best_essent_mat_context = EssentialMatContext(
                        frame_1,
                        frame_2,
                        E,
                        correspondences
                    )

                    best_inliers_ratio = inliers_ratio

                if inliers_ratio > self._init_view_mats_parameters['inliersRatioThres']:
                    continue

                view_mat_1, view_mat_2, points_3d_cnt = self._find_initial_view_mats(correspondences, E)

                if best_view_mats_context is None or best_points_3d_cnt < points_3d_cnt:
                    best_view_mats_context = ViewMatsContext(
                        frame_1,
                        frame_2,
                        view_mat3x4_to_pose(view_mat_1),
                        view_mat3x4_to_pose(view_mat_2)
                    )
                    best_points_3d_cnt = points_3d_cnt

        if best_view_mats_context is None:
            view_mat_1, view_mat_2, _ = self._find_initial_view_mats(
                best_essent_mat_context.correspondences,
                best_essent_mat_context.E,
            )

            return ViewMatsContext(
                best_essent_mat_context.frame_1,
                best_essent_mat_context.frame_2,
                view_mat3x4_to_pose(view_mat_1),
                view_mat3x4_to_pose(view_mat_2),
            )

        return best_view_mats_context


CAMERA_TRACKER_PARAMETERS = {
    'maxReprojError': 3.5,
    'PnPRansacParams': {
        'flags': cv2.SOLVEPNP_EPNP,
        'iterationsCount': 100,
        'reprojectionError': 3.5,
    },
    'points3dCntForBestFrame': 200,
}

INIT_VIEW_MATS_PARAMETERS = {
    'inliersRatioThres': 0.5,
    'framesWindowSize': 50,
    'findEssentialMatrixParams': {
        'prob': 0.9999,
        'threshold': 3.5,
    },
    'findHomographyParams': {
        'ransacReprojThreshold': 3.5,
    },
}

TRIANG_PARAMETERS = TriangulationParameters(7.5, 1.0, 0.1)


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )
    camtracker = CameraTracker(
        rgb_sequence,
        corner_storage,
        intrinsic_mat,
        TRIANG_PARAMETERS,
        CAMERA_TRACKER_PARAMETERS,
        known_view_1,
        known_view_2,
        INIT_VIEW_MATS_PARAMETERS,
    )

    return camtracker.camera_tracking()


if __name__ == '__main__':
    # pylint:disable=no-value-for-parameter
    create_cli(track_and_calc_colors)()
