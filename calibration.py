import numpy as np
import cv2


class Calib:
    """
    :parameter: calib_file -> result of json.load('some.json')
    """

    def __init__(self, calib):
        # Front camera
        self.calib = calib

    def img_rectification(self, cam_direction):
        cam = cam_direction_parser(cam_direction)
        img_width, img_height = self.img_width_height(cam_direction)
        img_size = (img_width, img_height)
        kmatrix = self.kmatrix(cam_direction)
        dist_coeffs = self.dist_coeffs(cam_direction)
        kmatrix_rec = self.kmatrix_rec(cam_direction)
        map_x, map_y = image_rectification(self.calib[cam]["MODEL"].lower().strip(),
                                           kmatrix, dist_coeffs,
                                           kmatrix_rec, img_size, cam_direction)
        return map_x, map_y

    @property
    def roof_lidar2front_cam_rotation(self):
        return np.array(
            self.calib['CAMERA_FRONT']["EXTRINSIC_CALIBRATION"]["lidar_roof_center-images_front"]['ROTATION'])

    @property
    def roof_lidar2front_cam_trans(self):
        return np.array(
            self.calib['CAMERA_FRONT']["EXTRINSIC_CALIBRATION"]["lidar_roof_center-images_front"]['TRANSLATION'])

    def kmatrix_rec(self, cam_direction):
        cam = cam_direction_parser(cam_direction)
        return np.array(self.calib['INTRINSIC_CALIBRATION']["UNDISTORTED_IMAGE"]["INTRINSICS"]).reshape([3, 3])

    def kmatrix_rec_3x4(self, cam_direction):
        kmatrix = self.kmatrix_rec(cam_direction)
        K_matrix_rec_3x4 = np.zeros([3, 4], dtype=kmatrix.dtype)
        K_matrix_rec_3x4[:3, :3] = kmatrix
        return K_matrix_rec_3x4

    def kmatrix(self, cam_direction):
        cam = cam_direction_parser(cam_direction)
        return np.array(self.calib['INTRINSIC_CALIBRATION']["ORIGINAL_IMAGE"]["INTRINSICS"])

    def dist_coeffs(self, cam_direction):
        cam = cam_direction_parser(cam_direction)
        return np.array(self.calib['INTRINSIC_CALIBRATION']["ORIGINAL_IMAGE"]["DIST_COEFFS"])

    def img_width_height(self, cam_direction: str = 'front'):
        cam = cam_direction_parser(cam_direction)
        img_width = self.calib[cam]['IMAGE_WIDTH']
        img_height = self.calib[cam]['IMAGE_HEIGHT']
        return img_width, img_height

    def extrinsic_calib_cam2veh(self, cam_direction):
        cam = cam_direction_parser(cam_direction)
        rotation = np.array(
            self.calib[cam]['EXTRINSIC_CALIBRATION']['TRANSFORM_POINTS_FROM_CAM_TO_VEHICLE']['ROTATION'])
        trans = np.array(
            self.calib[cam]['EXTRINSIC_CALIBRATION']['TRANSFORM_POINTS_FROM_CAM_TO_VEHICLE']['TRANSLATION'])
        return rotation, trans

    def transform_lidar_front2rsv(self, target_cam_direction):
        """
        p' = [R|t]p, [R|t] == roof_center_lidar-> front_cam - > vehicle -> side cam
        """
        lidar2frontCam_rotation = self.roof_lidar2front_cam_rotation
        lidar2frontCam_trans = self.roof_lidar2front_cam_trans
        rot_front2veh, trans_front2veh = self.extrinsic_calib_cam2veh('front')
        rot_rear2veh, trans_rear2veh = self.extrinsic_calib_cam2veh(target_cam_direction)
        rot_veh2rear = rot_rear2veh.T
        trans_veh2rear = -rot_veh2rear.dot(trans_rear2veh)
        rotation = rot_veh2rear.dot(rot_front2veh.dot(lidar2frontCam_rotation))
        trans = rot_veh2rear.dot(rot_front2veh.dot(lidar2frontCam_trans.T)).T + \
                rot_veh2rear.dot(trans_front2veh.T) + trans_veh2rear
        return rotation, trans

    def transform_plane_in_lidar2rsv(self, cam_direction, plane):
        """
        Transforms a plane given in lidar coordinates into RSV camera coordinates
        plane = [nx, ny, nz, h]
        """
        rotation, trans = self.transform_lidar_front2rsv(cam_direction)
        normal_lidar = plane[:3]
        height_lidar = plane[3]
        normal_rsv = rotation @ normal_lidar
        foot_point_lidar = -height_lidar * normal_lidar
        foot_point_rsv = rotation @ foot_point_lidar + trans
        height_rsv = -(foot_point_rsv @ normal_rsv)[0]
        return np.array([normal_rsv[0], normal_rsv[1], normal_rsv[2], height_rsv])


def image_rectification(cam_model: str, kmatrix, dist_coeffs, kmatrix_undistorted, img_size, cam_direction, vg_way = False):

    if cam_direction == 'rear_right' and vg_way:
        # rotation matrix from VG
        rot = np.array([[0.829217851, -0.0174437948, -0.558653295],
                        [-0.00309330039, 0.999354303, -0.0357959978],
                        [0.558916986, 0.0314107612, 0.828628421]])
        img_size = (896, 512)
    else:
        rot = np.eye(3)

    if cam_model == "fisheye":
        if vg_way:
            kmatrix_undistorted = np.array([[1200, 0.0, 448], [0.0, 1200, 256], [0, 0, 1]])
        map_x, map_y = cv2.fisheye.initUndistortRectifyMap(kmatrix, dist_coeffs, rot,
                                                           kmatrix_undistorted,
                                                           img_size, cv2.CV_16SC2)
    elif cam_model == "pinhole":
        map_x, map_y = cv2.initUndistortRectifyMap(kmatrix, dist_coeffs, None, kmatrix_undistorted, img_size,
                                                   cv2.CV_16SC2)
    else:
        raise ValueError(f"Camera model '{cam_model}' is not supported. Supported cam model: fisheye or pinhole")

    return map_x, map_y


def cam_direction_parser(cam_direction: str):
    if cam_direction == 'front':
        return 'CAMERA_FRONT'
    elif cam_direction == 'rear_left':
        return 'CAMERA_REAR_LEFT'
    elif cam_direction == 'rear_right':
        return 'CAMERA_REAR_RIGHT'
    elif cam_direction == 'SVM_front':
        return 'CAMERA'
    elif cam_direction == 'SVM_left':
        return 'CAMERA_FSIR_LEFT'
    elif cam_direction == 'SVM_right':
        return 'CAMERA_FSIR_RIGHT'
    else:
        raise ValueError("Please specify a supported camera direction")


