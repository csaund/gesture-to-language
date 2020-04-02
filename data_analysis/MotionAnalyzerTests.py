import unittest
from GestureSentenceManager import *

TEST_GESTURE_IDS = [
    82775,       # both hands moving down, somewhat quickly, rock, short
    119329      #
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

BASE_KEYPOINT = [0]
RIGHT_BODY_KEYPOINTS = [1, 2, 3, 28]
LEFT_BODY_KEYPOINTS = [4, 5, 6, 7]
LEFT_HAND_KEYPOINTS = lambda x: [7] + [8 + (x * 4) + j for j in range(4)]
RIGHT_HAND_KEYPOINTS = lambda x: [28] + [29 + (x * 4) + j for j in range(4)]
ALL_RIGHT_HAND_KEYPOINTS = [3] + list(range(31, 52))
ALL_LEFT_HAND_KEYPOINTS = [6] + list(range(10, 31))
BODY_KEYPOINTS = RIGHT_BODY_KEYPOINTS + LEFT_BODY_KEYPOINTS

class TestMotionFeatures(unittest.TestCase):

    def test_get_right_hand_one_frame(self):
        should_be = [{'y': [446, 420, 416, 447, 415, 466, 443, 460, 473, 479, 432, 456, 472, 482, 432, 452, 462, 472, 427, 449, 454, 459],
                      'x': [267, 367, 343, 282, 347, 262, 357, 353, 347, 351, 342, 334, 332, 335, 330, 320, 319, 315, 319, 307, 306, 306]
                    }]
        response = GC._get_rl_hand_keypoints(TEST_FRAME, "r")
        self.assertEqual(should_be, response)

    def test_get_right_hand_multi_frame(self):
        should_be = [{'y': [446, 420, 416, 447, 415, 466, 443, 460, 473, 479, 432, 456, 472, 482, 432, 452, 462, 472, 427, 449, 454, 459],
                      'x': [267, 367, 343, 282, 347, 262, 357, 353, 347, 351, 342, 334, 332, 335, 330, 320, 319, 315, 319, 307, 306, 306]
                    },
                     {'y': [447, 421, 417, 448, 416, 467, 444, 461, 474, 480, 433, 457, 473, 483, 433, 453, 463, 473, 428, 450, 455, 460],
                      'x': [268, 368, 344, 283, 348, 263, 358, 354, 348, 352, 343, 335, 333, 336, 331, 321, 320, 316, 320, 308, 307, 307]}]
        response = GC._get_rl_hand_keypoints(TEST_FRAMES, "r")
        self.assertEqual(should_be, response)

    def test_get_left_hand_one_frame(self):
        should_be = [{'y': [418, 418, 427, 438, 459, 482, 430, 445, 468, 479, 427, 456, 471, 478, 431, 461, 475, 486, 439, 464, 479, 491],
                      'x': [378, 371, 341, 317, 297, 269, 316, 280, 298, 299, 330, 317, 318, 323, 347, 335, 335, 336, 360, 353, 351, 352]
                    }]
        response = GC._get_rl_hand_keypoints(TEST_FRAME, "l")
        self.assertEqual(should_be, response)

    def test_get_left_hand_multi_frame(self):
        should_be = [{'y': [418, 418, 427, 438, 459, 482, 430, 445, 468, 479, 427, 456, 471, 478, 431, 461, 475, 486, 439, 464, 479, 491],
                      'x': [378, 371, 341, 317, 297, 269, 316, 280, 298, 299, 330, 317, 318, 323, 347, 335, 335, 336, 360, 353, 351, 352]
                    },
                     {'y': [419, 419, 428, 439, 460, 483, 431, 446, 469, 480, 428, 457, 472, 479, 432, 462, 476, 487, 440, 465, 480, 492],
                      'x': [379, 372, 342, 318, 298, 270, 317, 281, 299, 300, 331, 318, 319, 324, 348, 336, 336, 337, 361, 354, 352, 353]
                    }]
        response = GC._get_rl_hand_keypoints(TEST_FRAMES, "l")
        self.assertEqual(should_be, response)

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)

    def test_normalize(self):
        d = [{'feature_vec':[2, 2, 2]},
             {'feature_vec':[1.5, 2, 0.5]},
             {'feature_vec':[1, 1, 1]},
             {'feature_vec':[0, 0, 0]}]
        response = GC._normalize_feature_values(d)
        print(response)

    # TODO tests for:
    # RL keypoints are actually correct
    # motion detections for
    #         self._palm_vert(gesture, 'l'),
    #         self._palm_horiz(gesture, 'l'),
    #         self._palm_vert(gesture, 'r'),
    #         self._palm_horiz(gesture, 'r'),
    #         self._max_hands_apart(gesture),
    #         self._min_hands_together(gesture),
    #         #   x_oscillate,
    #         #   y_oscillate,
    #         self._wrists_up(gesture, 'r'),
    #         self._wrists_up(gesture, 'l'),
    #         self._wrists_down(gesture, 'r'),
    #         self._wrists_down(gesture, 'l'),
    #         self._wrists_outward(gesture),
    #         self._wrists_inward(gesture),
    #         self._max_wrist_velocity(gesture, 'r'),
    #         self._max_wrist_velocity(gesture, 'l')
    #         #   wrists_sweep,
    #         #   wrist_arc,
    #         #   r_hand_rotate,
    #         #   l_hand_rotate,
    #         #   hands_cycle
    # NORMALIZING THE FEATURE VECTORS!!!!

if __name__ == '__main__':
    unittest.main()