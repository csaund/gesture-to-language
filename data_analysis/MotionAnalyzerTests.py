import unittest
from GestureSentenceManager import *

TEST_GESTURE_IDS = [
    82775,       # both hands moving down, somewhat quickly, rock, short
    119329      #
]

# 1. find and load a couple of canonical examples with keyframes
#     gesture_features = [
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
#     ]

GSM = GestureSentenceManager("rock")
GC = GSM.GestureClusterer


TEST_FRAME = {'keyframes':
                [{'y': [247, 242, 387, 446, 260, 425, 418, 151, 127, 131, 418, 427, 438, 459, 482, 430, 445, 468, 479, 427, 456, 471, 478, 431, 461, 475, 486, 439, 464, 479, 491, 420, 416, 447, 415, 466, 443, 460, 473, 479, 432, 456, 472, 482, 432, 452, 462, 472, 427, 449, 454, 459],
                  'x': [326, 199, 160, 267, 449, 499, 378, 327, 305, 350, 371, 341, 317, 297, 269, 316, 280, 298, 299, 330, 317, 318, 323, 347, 335, 335, 336, 360, 353, 351, 352, 367, 343, 282, 347, 262, 357, 353, 347, 351, 342, 334, 332, 335, 330, 320, 319, 315, 319, 307, 306, 306]}]}

TEST_FRAME = {'keyframes':
                [{'y': [247, 242, 387, 446, 260, 425, 418, 151, 127, 131, 418, 427, 438, 459, 482, 430, 445, 468, 479,
                        427, 456, 471, 478, 431, 461, 475, 486, 439, 464, 479, 491, 420, 416, 447, 415, 466, 443, 460,
                        473, 479, 432, 456, 472, 482, 432, 452, 462, 472, 427, 449, 454, 459],
                  'x': [326, 199, 160, 267, 449, 499, 378, 327, 305, 350, 371, 341, 317, 297, 269, 316, 280, 298, 299,
                        330, 317, 318, 323, 347, 335, 335, 336, 360, 353, 351, 352, 367, 343, 282, 347, 262, 357, 353,
                        347, 351, 342, 334, 332, 335, 330, 320, 319, 315, 319, 307, 306, 306]},
                 {'y': [245, 243, 385, 446, 260, 425, 418, 151, 127, 131, 418, 427, 438, 459, 482, 430, 445, 468, 479,
                        427, 456, 471, 478, 431, 461, 475, 486, 439, 464, 479, 491, 420, 416, 447, 415, 466, 443, 460,
                        473, 479, 432, 456, 472, 482, 432, 452, 462, 472, 427, 449, 454, 459],
                  'x': [126, 399, 260, 267, 449, 499, 378, 327, 305, 350, 371, 341, 317, 297, 269, 316, 280, 298, 299,
                        330, 317, 318, 323, 347, 335, 335, 336, 360, 353, 351, 352, 367, 343, 282, 347, 262, 357, 353,
                        347, 351, 342, 334, 332, 335, 330, 320, 319, 315, 319, 307, 306, 306]},
                 ]}

class TestMotionFeatures(unittest.TestCase):

    def get_body_range_one_frame(self):
        should_be = [{'y': [247, 242, 387],
                      'x': [326, 199, 160]}]
        response = GC._get_keypoints_body_range(TEST_FRAME, 0, 3)
        self.assertEqual(should_be, response)

    def get_body_range_multi_frame(self):
        should_be = [{'y': [247, 242, 387],
                      'x': [326, 199, 160]},
                     {'y': [245, 243, 385],
                      'x': [126, 399, 260]}]
        response = GC._get_keypoints_body_range(TEST_FRAME, 0, 3)
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

if __name__ == '__main__':
    unittest.main()