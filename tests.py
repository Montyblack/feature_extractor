import unittest
import matplotlib.pyplot as plt
import numpy as np
from skimage import util
from skimage.transform import rescale
import utils.graphic_calcs as ugc
# from mp_colors.crop import FaceCropper

from get_colortypes_csv import \
    get_landmarks, \
    get_face_undereyes_color, \
    get_face_forehead_color, \
    get_face_cheeks_color, get_face_hair_color
# , \
#     get_lips_mediapipe, \
#     get_cheek_mediapipe, \
#     get_forehead_mediapipe, \
#     get_nose_mediapipe, \
#     get_skin_mediapipe, \
#     get_eyes_mediapipe, \
#     get_iris_mediapipe

# from mp_colors.colors import get_keypoints
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# face_cropper = FaceCropper(alpha=0.4)


class Case1(unittest.TestCase):
    def setUp(self):
        self.image_path = './data/photo_augmentation/ColortypesDone2CropedAngelEurop/summer/00000428.jpg'
        self.image = plt.imread(self.image_path)
        self.image = util.img_as_ubyte(rescale(self.image, (0.5, 0.5, 1), anti_aliasing=False))
        self.imageBGR = ugc.switch_rgb(self.image)
        self.landmarks, _ = get_landmarks(self.image, '00000428.jpg')

        _, _, _, self.imageRGB = get_face_hair_color(self.imageBGR)
#         self.faces = face_cropper.detect(self.imageRGB)
#         self.face = self.faces[0]
#         self.mp_keypoints = np.load('./mp_keypoints_1.npy', allow_pickle=False)

    # def test_keypoints(self):
    #     self.assertEqual((np.abs(self.face - np.load('./face.npy'))).sum(), 0)

    def test_get_face_undereyes_color(self):
        face_undereyes_color_h, face_undereyes_color_s, \
            face_undereyes_color_v = get_face_undereyes_color(self.imageBGR, self.landmarks[1],
                                                              self.landmarks[29],
                                                              self.landmarks[15])
        self.assertEqual([face_undereyes_color_h, face_undereyes_color_s, face_undereyes_color_v],
                         [0.043478260869565154, 0.2090909090909091, 0.8627450980392157])

    def test_get_face_forehead_color(self):
        face_forehead_color_h, face_forehead_color_s, \
            face_forehead_color_v = get_face_forehead_color(self.imageBGR, self.landmarks)
        self.assertEqual([face_forehead_color_h, face_forehead_color_s, face_forehead_color_v],
                         [0.058641975308642014, 0.23175965665236056, 0.9137254901960784])

    def test_get_face_cheeks_color(self):
        face_cheeks_color_h, face_cheeks_color_s, \
            face_cheeks_color_v = get_face_cheeks_color(self.imageBGR, self.landmarks[1],
                                                       self.landmarks[15],
                                                       self.landmarks[12],
                                                       self.landmarks[4])
        self.assertEqual([face_cheeks_color_h, face_cheeks_color_s, face_cheeks_color_v],
                         [0.050847457627118696, 0.25764192139737996, 0.8980392156862745])

    def test_get_face_hair_color(self):
        face_hair_color_h, face_hair_color_s, \
            face_hair_color_v, _ = get_face_hair_color(self.imageBGR)
        self.assertEqual([face_hair_color_h, face_hair_color_s, face_hair_color_v],
                         [0.0, 0.0, 1.0])

#     def test_get_lips_mediapipe(self):
#         mp_lip_r, mp_lip_g, \
#             mp_lip_b = get_lips_mediapipe(self.face, self.mp_keypoints)
#         self.assertEqual([mp_lip_r, mp_lip_g, mp_lip_b], [192, 124, 124])

#     def test_get_cheek_mediapipe(self):
#         mp_cheek_r, mp_cheek_g, \
#             mp_cheek_b = get_cheek_mediapipe(self.face, self.mp_keypoints)
#         self.assertEqual([mp_cheek_r, mp_cheek_g, mp_cheek_b], [224, 179, 161])

#     def test_get_forehead_mediapipe(self):
#         mp_forehead_r, mp_forehead_g, \
#             mp_forehead_b = get_forehead_mediapipe(self.face, self.mp_keypoints)
#         self.assertEqual([mp_forehead_r, mp_forehead_g, mp_forehead_b], [222, 185, 166])

#     def test_get_nose_mediapipe(self):
#         mp_nose_r, mp_nose_g, \
#             mp_nose_b = get_nose_mediapipe(self.face, self.mp_keypoints)
#         self.assertEqual([mp_nose_r, mp_nose_g, mp_nose_b], [222, 187, 170])

#     def test_get_skin_mediapipe(self):
#         mp_skin_r, mp_skin_g, \
#             mp_skin_b = get_skin_mediapipe(self.face, self.mp_keypoints)
#         self.assertEqual([mp_skin_r, mp_skin_g, mp_skin_b], [223, 183, 165])

#     def test_get_eyes_mediapipe(self):
#         mp_eyes_r, mp_eyes_g, \
#             mp_eyes_b = get_eyes_mediapipe(self.face, self.mp_keypoints)
#         self.assertEqual([mp_eyes_r, mp_eyes_g, mp_eyes_b], [100, 81, 73])

#     def test_get_iris_mediapipe(self):
#         mp_iris_r, mp_iris_g, \
#             mp_iris_b = get_iris_mediapipe(self.face)
#         self.assertEqual([mp_iris_r, mp_iris_g, mp_iris_b], [35.0, 43.0, 58.0])
