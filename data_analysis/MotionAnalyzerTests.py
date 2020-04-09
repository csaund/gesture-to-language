import unittest
import numpy as np
import numbers
from GestureSentenceManager import *
from GestureClusterer import GestureClusterer, _normalize_across_features, _get_rl_hand_keypoints
from GestureMovementHelpers import _max_hands_apart, _min_hands_together, _get_point_dist, _wrists_apart, _wrists_together

TEST_GESTURE_IDS = [
    3225,    # back and forth        # 10._Personal_identity_Part_I_-_Identity_across_space_and_time_and_the_soul_theory-00KDsArsQ3A_346.012679_358.491825
    3215,    # big arm apart vertically   # 2._The_nature_of_persons_-_dualism_vs._physicalism-gh-6HyTRNNY_2430.296964_2440.774107
    1524,    # lots of shaking in beginning, high velocity shaking  14._What_matters_cont._The_nature_of_death_Part_I-0_vqoz05W-0_114.981648_136.936937
    3855,    # quite slow           # 3._Arguments_for_the_existence_of_the_soul_Part_I-GR63MMAi-fs_958.158158_960.894228
    1726,    # very complex, long, sweep up, down, shaking, cycling gah #   15._The_nature_of_death_cont._Believing_you_will_die-DnYt2hiOR8c_2041.207875_2058.558559
    4093,    # one handed   # 8._Plato_Part_III_-_Arguments_for_the_immortality_of_the_soul_cont.-oJzGgp-hoKc_2558.992326_2564.731398
    4091,    # big outward sweep to side    # 1._Course_introduction-p2J7wSuFRl8_1451.217885_1462.495829
    374,     # very long. 1:16          # 21._Other_bad_aspects_of_death_Part_II-0qrx6x3izQw_2300.633967_2390.19019
    4001,    # super big quick open at the end # 5._Arguments_for_the_existence_of_the_soul_Part_III_-_Free_will_and_near-death_experiences-CWVlorPM3rA_819.953287_835.502169
    4505,    # small, hold frame, slight shake  # 12._Personal_identity_Part_III_-_Objections_to_the_personality_theory-rzbJ5nkAVfE_1680.38038_1686.119453
]

GSM = GestureSentenceManager("test")
GC = GSM.GestureClusterer


TEST_FRAME = {'keyframes':
            [{'y': [247, 242, 387, 446, 260, 425, 418, 151, 127, 131, 418, 427, 438, 459, 482, 430, 445, 468, 479,
                    427, 456, 471, 478, 431, 461, 475, 486, 439, 464, 479, 491, 420, 416, 447, 415, 466, 443, 460,
                    473, 479, 432, 456, 472, 482, 432, 452, 462, 472, 427, 449, 454, 459],
              'x': [326, 199, 160, 267, 449, 499, 378, 327, 305, 350, 371, 341, 317, 297, 269, 316, 280, 298, 299,
                    330, 317, 318, 323, 347, 335, 335, 336, 360, 353, 351, 352, 367, 343, 282, 347, 262, 357, 353,
                    347, 351, 342, 334, 332, 335, 330, 320, 319, 315, 319, 307, 306, 306]}]}

TEST_FRAMES = {'keyframes':
                [{'y': [247, 242, 387, 446, 260, 425, 418, 151, 127, 131, 418, 427, 438, 459, 482, 430, 445, 468, 479,
                        427, 456, 471, 478, 431, 461, 475, 486, 439, 464, 479, 491, 420, 416, 447, 415, 466, 443, 460,
                        473, 479, 432, 456, 472, 482, 432, 452, 462, 472, 427, 449, 454, 459],
                  'x': [326, 199, 160, 267, 449, 499, 378, 327, 305, 350, 371, 341, 317, 297, 269, 316, 280, 298, 299,
                        330, 317, 318, 323, 347, 335, 335, 336, 360, 353, 351, 352, 367, 343, 282, 347, 262, 357, 353,
                        347, 351, 342, 334, 332, 335, 330, 320, 319, 315, 319, 307, 306, 306]},
                 {'y': [246, 244, 386, 447, 261, 426, 419, 152, 128, 132, 419, 428, 439, 460, 483, 431, 446, 469, 480,
                        428, 457, 472, 479, 432, 462, 476, 487, 440, 465, 480, 492, 421, 417, 448, 416, 467, 444, 461,
                        474, 480, 433, 457, 473, 483, 433, 453, 463, 473, 428, 450, 455, 460],
                  'x':[327, 200, 161, 268, 450, 500, 379, 328, 306, 351, 372, 342, 318, 298, 270, 317, 281, 299, 300,
                       331, 318, 319, 324, 348, 336, 336, 337, 361, 354, 352, 353, 368, 344, 283, 348, 263, 358, 354,
                       348, 352, 343, 335, 333, 336, 331, 321, 320, 316, 320, 308, 307, 307]}
                 ]}

TEST_FRAMES_APART = {'keyframes': [{
                    "y": [127, 132, 196, 239, 120, 171, 215, 71, 62, 60, 229, 223, 221, 217, 215, 236, 234, 232, 228, 242,
                          238, 235, 233, 243, 241, 238, 235, 243, 241, 239, 237, 252, 248, 248, 248, 248, 263, 261, 256,
                          252, 266, 262, 254, 251, 267, 261, 254, 253, 265, 261, 257,255],
                    "x": [338, 293, 276, 266, 383, 412, 426, 344, 330, 351, 427, 426, 436, 443, 451, 432, 424, 417, 412, 428, 419, 412, 407,
                          423, 417, 410, 405, 419, 414, 410, 406, 263, 257, 248, 239, 231, 253, 254, 258, 261, 260, 263, 265, 266, 265, 269,
                          269, 269, 270, 274, 272, 272]},
                    {"y": [127, 131, 195, 239, 120, 176, 215, 70, 60, 59, 229, 220, 218, 216, 213, 232, 230, 226, 223, 238, 234,
                          229, 227, 241, 236, 231, 229, 241, 239, 236, 232, 251, 246, 248, 248, 248, 265, 263, 258, 255,
                          269, 261, 255, 253, 271, 261, 254, 253, 270, 262, 257, 57],
                    "x": [339, 295, 277, 266, 385, 415, 429, 351, 337, 358, 431, 430, 440, 448, 456, 437, 433, 426, 422,
                           432, 424, 418, 416, 426, 419, 413, 412, 420, 417, 414, 412, 264, 258, 247, 239, 231, 251, 253,
                           257, 260, 257, 263, 265, 265, 262, 270, 270, 268, 267, 273, 272, 272]}
                    ]}


BASE_KEYPOINT = [0]
# DO NOT TRUST THESE
RIGHT_BODY_KEYPOINTS = [1, 2, 3, 31]
LEFT_BODY_KEYPOINTS = [4, 5, 6, 10]
RIGHT_WRIST_KEYPOINT = 31
LEFT_WRIST_KEYPOINT = 10
# LEFT_HAND_KEYPOINTS = lambda x: [7] + [8 + (x * 4) + j for j in range(4)]  THESE ARE NOT RIGHT
# RIGHT_HAND_KEYPOINTS = lambda x: [28] + [29 + (x * 4) + j for j in range(4)]   THESE ARE NOT RIGHT
ALL_RIGHT_HAND_KEYPOINTS = list(range(31, 52))
ALL_LEFT_HAND_KEYPOINTS = list(range(10, 31))
BODY_KEYPOINTS = RIGHT_BODY_KEYPOINTS + LEFT_BODY_KEYPOINTS
DIRECTION_ANGLE_SWITCH = 110


class TestMotionFeatures(unittest.TestCase):

    def assertAlmostArray(self, a, b, dec=7):
        self.assertTrue(len(a) == len(b))
        for i in range(len(a)):
            self.assertAlmostEqual(a[i], b[i], dec)

    def test_get_right_hand_multi_frame(self):
        tfk = TEST_FRAMES['keyframes']
        ALL_RIGHT_HAND_KEYPOINTS = np.array(list(range(31, 52)))
        x0 = np.array(tfk[0]['x'])
        y0 = np.array(tfk[0]['y'])
        x1 = np.array(tfk[1]['x'])
        y1 = np.array(tfk[1]['y'])
        should_be = [
            {
                'x': x0[ALL_RIGHT_HAND_KEYPOINTS],
                'y': y0[ALL_RIGHT_HAND_KEYPOINTS]
            },
            {
                'x': x1[ALL_RIGHT_HAND_KEYPOINTS],
                'y': y1[ALL_RIGHT_HAND_KEYPOINTS]
            }
        ]
        response = _get_rl_hand_keypoints(TEST_FRAMES, "r")
        self.assertAlmostArray(should_be[0]['x'], response[0]['x'])
        self.assertAlmostArray(should_be[0]['y'], response[0]['y'])
        self.assertAlmostArray(should_be[1]['x'], response[1]['x'])
        self.assertAlmostArray(should_be[1]['y'], response[1]['y'])

    def test_get_left_hand_multi_frame(self):
        tfk = TEST_FRAMES['keyframes']
        ALL_LEFT_HAND_KEYPOINTS = np.array(list(range(10, 31)))
        x0 = np.array(tfk[0]['x'])
        y0 = np.array(tfk[0]['y'])
        x1 = np.array(tfk[1]['x'])
        y1 = np.array(tfk[1]['y'])
        should_be = [
            {
                'x': x0[ALL_LEFT_HAND_KEYPOINTS],
                'y': y0[ALL_LEFT_HAND_KEYPOINTS]
            },
            {
                'x': x1[ALL_LEFT_HAND_KEYPOINTS],
                'y': y1[ALL_LEFT_HAND_KEYPOINTS]
            }
        ]
        response = _get_rl_hand_keypoints(TEST_FRAMES, "l")
        self.assertAlmostArray(should_be[0]['x'], response[0]['x'])
        self.assertAlmostArray(should_be[0]['y'], response[0]['y'])
        self.assertAlmostArray(should_be[1]['x'], response[1]['x'])
        self.assertAlmostArray(should_be[1]['y'], response[1]['y'])

    def test_assert_almost_equal(self):
        self.assertAlmostArray([1.6454354332, 3.6454354352], [1.6454354, 3.64543543])

    def test_normalize(self):
        should_be = np.array([[0, 0.37796447, 0.55215763, 0.26967994],
           [0.83462233, 0.75592895, 0.4417261, 0.13483997],
           [0.38949042, 0.37796447, 0.4417261, 0.94387981],
           [0.38949042, 0.37796447, 0.5521576, 0.13483997]])
        a = [0, 1, 5, 2]
        b = [15, 2, 4, 1]
        c = [7, 1, 4, 7]
        d = [7, 1, 5, 1]
        response = _normalize_across_features(np.array([a, b, c, d]))
        for i in range(len(response)):
            self.assertAlmostArray(should_be[i], response[i])

    def test_normalize_GC(self):
        a = [0, 1, 5, 2]
        b = [15, 2, 4, 1]
        c = [7, 1, 4, 7]
        d = [7, 1, 5, 1]
        should_be = [{'feature_vec': [0, 0.37796447, 0.55215763, 0.26967994]},
                      {'feature_vec': [0.83462233, 0.75592895, 0.4417261, 0.13483997]},
                      {'feature_vec': [0.38949042, 0.37796447, 0.4417261, 0.94387981]},
                      {'feature_vec': [0.38949042, 0.37796447, 0.55215763, 0.13483997]}]
        gesture_input = [{'feature_vec': a},
                      {'feature_vec': b},
                      {'feature_vec': c},
                      {'feature_vec': d}]
        response = GC._normalize_feature_values(gesture_input)
        for i in range(len(response)):
            self.assertAlmostArray(should_be[i]['feature_vec'], response[i]['feature_vec'])

    def test_get_gesture_features(self):
        response = GC._get_gesture_features(TEST_FRAMES)
        for feat in response:
            self.assertIsInstance(feat, numbers.Number)

    def test_get_max_hands_apart(self):
        right = _get_rl_hand_keypoints(TEST_FRAMES, 'r')
        left = _get_rl_hand_keypoints(TEST_FRAMES, 'l')
        response = _max_hands_apart(right, left)
        y1 = TEST_FRAME['keyframes'][0]['y']
        x1 = TEST_FRAME['keyframes'][0]['x']
        y1R = [y1[i] for i in ALL_RIGHT_HAND_KEYPOINTS]
        x1R = [x1[i] for i in ALL_RIGHT_HAND_KEYPOINTS]
        y1L = [y1[i] for i in ALL_LEFT_HAND_KEYPOINTS]
        x1L = [x1[i] for i in ALL_LEFT_HAND_KEYPOINTS]
        R = np.array((y1R, x1R))
        L = np.array((y1L, x1L))
        should_be = np.linalg.norm(R-L)
        self.assertAlmostEqual(should_be, response)

    def test_get_min_hands_together(self):
        r = _get_rl_hand_keypoints(TEST_FRAMES, 'r')
        l = _get_rl_hand_keypoints(TEST_FRAMES, 'l')
        response = _min_hands_together(r, l)
        y1 = TEST_FRAME['keyframes'][0]['y']
        x1 = TEST_FRAME['keyframes'][0]['x']
        y1R = [y1[i] for i in ALL_RIGHT_HAND_KEYPOINTS]
        x1R = [x1[i] for i in ALL_RIGHT_HAND_KEYPOINTS]
        y1L = [y1[i] for i in ALL_LEFT_HAND_KEYPOINTS]
        x1L = [x1[i] for i in ALL_LEFT_HAND_KEYPOINTS]
        R = np.array((y1R, x1R))
        L = np.array((y1L, x1L))
        should_be = np.linalg.norm(R-L)
        self.assertAlmostEqual(should_be, response)

    def test_wrists_together(self):
        right = _get_rl_hand_keypoints(TEST_FRAMES_APART, 'r')
        left = _get_rl_hand_keypoints(TEST_FRAMES_APART, 'l')
        response = _wrists_together(right, left)
        keys = TEST_FRAMES_APART['keyframes']
        rw1 = np.array((keys[0]['x'][RIGHT_WRIST_KEYPOINT], keys[0]['y'][RIGHT_WRIST_KEYPOINT]))
        lw1 = np.array((keys[0]['x'][LEFT_WRIST_KEYPOINT], keys[0]['y'][LEFT_WRIST_KEYPOINT]))
        dist_1 = np.linalg.norm(rw1-lw1)
        rw2 = np.array((keys[1]['x'][RIGHT_WRIST_KEYPOINT], keys[1]['y'][RIGHT_WRIST_KEYPOINT]))
        lw2 = np.array((keys[1]['x'][LEFT_WRIST_KEYPOINT], keys[1]['y'][LEFT_WRIST_KEYPOINT]))
        dist_2 = np.linalg.norm(rw2-lw2)
        should_be = max(0, dist_1 - dist_2)
        self.assertEqual(should_be, response)

    def test_wrists_apart(self):
        right = _get_rl_hand_keypoints(TEST_FRAMES_APART, 'r')
        left = _get_rl_hand_keypoints(TEST_FRAMES_APART, 'l')
        response = _wrists_apart(right, left)
        keys = TEST_FRAMES_APART['keyframes']
        rw1 = np.array((keys[0]['x'][RIGHT_WRIST_KEYPOINT], keys[0]['y'][RIGHT_WRIST_KEYPOINT]))
        lw1 = np.array((keys[0]['x'][LEFT_WRIST_KEYPOINT], keys[0]['y'][LEFT_WRIST_KEYPOINT]))
        dist_1 = np.linalg.norm(rw1-lw1)
        rw2 = np.array((keys[1]['x'][RIGHT_WRIST_KEYPOINT], keys[1]['y'][RIGHT_WRIST_KEYPOINT]))
        lw2 = np.array((keys[1]['x'][LEFT_WRIST_KEYPOINT], keys[1]['y'][LEFT_WRIST_KEYPOINT]))
        dist_2 = np.linalg.norm(rw2-lw2)
        should_be = dist_2 - dist_1
        self.assertEqual(should_be, response)

    def test_zero_distance_same_gesture(self):
        result = GC._calculate_distance_between_gestures(TEST_FRAMES, TEST_FRAMES)
        self.assertEqual(0, result)

    def test_point_diff(self):
        result = _get_point_dist(2, 3, 5, 6)
        a = np.array([2, 3])
        b = np.array([5, 6])
        should_be = np.linalg.norm(a-b)
        self.assertEqual(should_be, result)

    # TODO create test frames that have a difference in y
    # def test_wrists_up(self):
    #     r = GC._get_rl_hand_keypoints(TEST_FRAMES_APART, 'r')
    #     result = GC._wrists_up(r)

    # TODO tests for:
    # vector math works out
    # clusterer gets fewer than max number of clusters
    # cluster distance is respected?
    # RL keypoints are actually correct
    # motion detections for
    #         self._palm_vert(gesture, 'l'),
    #         self._palm_horiz(gesture, 'l'),
    #         self._palm_vert(gesture, 'r'),
    #         self._palm_horiz(gesture, 'r'),
    #         #   x_oscillate,
    #         #   y_oscillate,
    #         self._wrists_up(gesture, 'r'),
    #         self._wrists_up(gesture, 'l'),
    #         self._wrists_down(gesture, 'r'),
    #         self._wrists_down(gesture, 'l'),
    #         self._max_wrist_velocity(gesture, 'r'),
    #         self._max_wrist_velocity(gesture, 'l')
    #         #   wrists_sweep,
    #         #   wrist_arc,
    #         #   r_hand_rotate,
    #         #   l_hand_rotate,
    #         #   hands_cycle


if __name__ == '__main__':
    unittest.main()
