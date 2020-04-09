import os
import subprocess
import numpy as np
from PIL import Image
from matplotlib import cm, pyplot as plt
from GestureClusterer import _get_rl_hand_keypoints
LEFT_HAND_KEYPOINTS = lambda x: [10] + [11 + (x * 4) + j for j in range(4)]  # CHECK THESE
RIGHT_HAND_KEYPOINTS = lambda x: [31] + [32 + (x * 4) + j for j in range(4)]   # CHECK THESE
# https://github.com/amirbar/speech2gesture/blob/master/common/pose_plot_lib.py
from GestureMovementHelpers import BASE_KEYPOINT, RIGHT_BODY_KEYPOINTS, LEFT_BODY_KEYPOINTS
LINE_WIDTH_CONST = 1.5


def draw_hand(hand_keys, frame):
    keys = hand_keys[frame]
    xs = np.array(keys['x'])
    ys = np.array(keys['y'])
    for i in range(5):
        _kp = [0] + [(i*4)+j for j in range(1, 5)]
        plt.plot(xs[_kp], ys[_kp])
    plt.show()


def draw_hands(gesture, frame):
    right_hand = _get_rl_hand_keypoints(gesture, 'r')
    left_hand = _get_rl_hand_keypoints(gesture, 'l')
    draw_hand(right_hand, frame)
    draw_hand(left_hand, frame)


def draw_arms(gesture, frame):
    k = gesture['keyframes']
    keys = k[frame]
    xs = np.array(keys['x'])
    ys = np.array(keys['y'])
    r_keypoints = np.array(BASE_KEYPOINT + RIGHT_BODY_KEYPOINTS)
    l_keypoints = np.array(BASE_KEYPOINT + LEFT_BODY_KEYPOINTS)
    plt.plot(xs[r_keypoints], ys[r_keypoints])
    plt.plot(xs[l_keypoints], ys[l_keypoints])


def draw_pose_custom(gesture, frame, img=None, img_width=1280, img_height=720, output=None, title=None, title_x=1,
                     alpha_img=0.5, fig=None, show=True):
    if fig is None:
        plt.close("all")
        fig = plt.figure(figsize=(6, 4))

    plt.axis('off')

    if img != None:
        img = Image.open(img)
        img_width, img_height = img.size
    else:
        from PIL import Image
        img = Image.new(mode='RGB', size=(img_width, img_height), color='white')

    plt.imshow(img, alpha=alpha_img)
    draw_arms(gesture, frame)
    draw_hands(gesture, frame)
    ax = fig.get_axes()[0]
    ax.set_xlim(0, img_width)
    ax.set_ylim(img_height, 0)
    if title:
        plt.title(title, x=title_x)

    if show:
        plt.show()

    if output:
        plt.savefig(output)
        plt.close()


def save_video(g, output_fn,temp_folder='temp_gesture_plots',  delete_tmp=True):
    if not (os.path.exists(temp_folder)):
        os.makedirs(temp_folder)

    #if not (os.path.exists(os.path.dirname(output_fn))):
    #    os.makedirs(temp_folder)

    output_fn_pattern = os.path.join(temp_folder, '%04d.jpg')
    keys = g['keyframes']

    for j in range(len(keys)):
        draw_pose_custom(g, j, output=output_fn_pattern % j, show=False)
        plt.close()

    create_mute_video_from_images(output_fn, temp_folder)
    if delete_tmp:
        subprocess.call('rm -R "%s"' % (temp_folder), shell=True)


def create_mute_video_from_images(output_fn, temp_folder):
    '''
    :param output_fn: output video file name
    :param temp_folder: contains images in the format 0001.jpg, 0002.jpg....
    :return:
    '''
    subprocess.call('ffmpeg -loglevel panic -r 30000/2002 -f image2 -i "%s" -r 30000/1001 "%s" -y' % (
        os.path.join(temp_folder, '%04d.jpg'), output_fn), shell=True)
