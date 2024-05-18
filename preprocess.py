# python3 and python2
import glob
import os
from functools import partial
from multiprocessing import Pool

import cv2
import numpy as np


def align_2p(img, left_eye, right_eye):
    """
    Aligns a face image based on the positions of the left and right eyes.

    Args:
    - img (numpy.ndarray): Input face image.
    - left_eye (tuple): Coordinates of the left eye (x, y).
    - right_eye (tuple): Coordinates of the right eye (x, y).

    Returns:
    - numpy.ndarray: Aligned face image.
    """
    width = 256  # Target width of the aligned face image
    eye_width = 70  # Expected width between the eyes in the aligned image

    # Initialize an identity transformation matrix
    transform = np.matrix([
        [1, 0, left_eye[0]],
        [0, 1, left_eye[1]],
        [0, 0, 1]
    ], dtype='float')

    # Calculate the angle of rotation based on the line connecting the left and right eyes
    th = np.pi + -np.arctan2(left_eye[1] - right_eye[1], left_eye[0] - right_eye[0])

    # Apply rotation transformation to the matrix
    transform *= np.matrix([
        [np.cos(th), np.sin(th), 0],
        [-np.sin(th), np.cos(th), 0],
        [0, 0, 1]
    ], dtype='float')

    # Calculate the scale factor based on the distance between the left and right eyes
    scale = np.sqrt((left_eye[1] - right_eye[1]) ** 2 + (left_eye[0] - right_eye[0]) ** 2) / eye_width

    # Apply scaling transformation to the matrix
    transform *= np.matrix([
        [scale, 0, 0],
        [0, scale, 0],
        [0, 0, 1]
    ], dtype='float')

    # Translate the aligned image to the center of the output image
    transform *= np.matrix([
        [1, 0, -(width - eye_width) / 2],
        [0, 1, -width / 2.42],  # Adjusted value for vertical alignment
        [0, 0, 1]
    ], dtype='float')

    # Invert the transformation matrix
    transform = np.linalg.inv(transform)

    # Apply the transformation to the input image using warpAffine
    aligned_img = cv2.warpAffine(img, transform[:2], (width, width))

    return aligned_img


def align_face_2p(img, landmarks):
    """
    Aligns a face image based on two facial landmarks representing the positions of the left and right eyes.

    Args:
    - img (numpy.ndarray): Input face image.
    - landmarks (list or tuple): List or tuple containing the coordinates of two facial landmarks.
                                 The expected format is [(x1, y1), (x2, y2)].

    Returns:
    - numpy.ndarray: Aligned face image.
    """
    # Extract the coordinates of the left and right eye landmarks
    left_eye = (landmarks[0], landmarks[1])
    right_eye = (landmarks[2], landmarks[3])

    # Call the align_2p function to perform alignment based on the two landmarks
    aligned_img = align_2p(img, left_eye, right_eye)

    return aligned_img

# average landmarks
mean_face_lm5p = np.array([
    [-0.17607, -0.172844],  # left eye pupil
    [0.1736, -0.17356],  # right eye pupil
    [-0.00182, 0.0357164],  # nose tip
    [-0.14617, 0.20185],  # left mouth corner
    [0.14496, 0.19943],  # right mouth corner
])


def _get_align_5p_mat23_size_256(lm):
    # Define the width of the output image (256 pixels)
    width = 256

    # Copy the mean face landmarks
    mf = mean_face_lm5p.copy()

    # Calculate the ratio for scaling the landmarks based on the output image size
    ratio = 70.0 / (256.0 * 0.34967)
    # Magic number 0.34967 compensates for scaling from average landmarks

    # Calculate the vertical distance between the left eye pupil and the nose tip in the output image
    left_eye_pupil_y = mf[0][1]
    ratioy = (left_eye_pupil_y * ratio + 0.5) * (1 + 1.42)
    # In an aligned face image, the ratio between the vertical distances from eye to the top and bottom is 1:1.42

    # Scale and translate the mean face landmarks to the output image size
    mf[:, 0] = (mf[:, 0] * ratio + 0.5) * width
    mf[:, 1] = (mf[:, 1] * ratio + 0.5) * width / ratioy

    # Calculate means of the mean face landmarks and input landmarks
    mx = mf[:, 0].mean()
    my = mf[:, 1].mean()
    dmx = lm[:, 0].mean()
    dmy = lm[:, 1].mean()

    # Compute parameters for affine transformation
    ux = mf[:, 0] - mx
    uy = mf[:, 1] - my
    dux = lm[:, 0] - dmx
    duy = lm[:, 1] - dmy
    c1 = (ux * dux + uy * duy).sum()
    c2 = (ux * duy - uy * dux).sum()
    c3 = (dux ** 2 + duy ** 2).sum()
    a = c1 / c3
    b = c2 / c3

    # Scale factors
    kx = 1
    ky = 1

    # Compute additional transformation parameters
    s = c3 / (c1 ** 2 + c2 ** 2)
    ka = c1 * s
    kb = c2 * s

    # Build the affine transformation matrix
    transform = np.zeros((2, 3))
    transform[0][0] = kx * a
    transform[0][1] = kx * b
    transform[0][2] = mx - kx * a * dmx - kx * b * dmy
    transform[1][0] = -ky * b
    transform[1][1] = ky * a
    transform[1][2] = my - ky * a * dmy + ky * b * dmx

    return transform

def get_align_5p_mat23(lm5p, size):
    """Align a face given 5 facial landmarks of
    left_eye_pupil, right_eye_pupil, nose_tip, left_mouth_corner, right_mouth_corner

    :param lm5p: nparray of (5, 2), 5 facial landmarks,

    :param size: an integer, the output image size. The face is aligned to the mean face

    :return: a affine transformation matrix of shape (2, 3)
    """
    mat23 = _get_align_5p_mat23_size_256(lm5p.copy())
    mat23 *= size / 256
    return mat23


def align_given_lm5p(img, lm5p, size):
    """
    Aligns a face image based on given 5-point facial landmarks and target size.

    Args:
    - img (numpy.ndarray): Input face image.
    - lm5p (numpy.ndarray): 5-point facial landmarks array, reshaped to a 5x2 array.
    - size (int): Target size for the aligned image (both width and height).

    Returns:
    - numpy.ndarray: Aligned face image.
    """
    # Calculate the affine transformation matrix using the given landmarks and size
    mat23 = get_align_5p_mat23(lm5p, size)

    # Apply the affine transformation to the input image to align it
    aligned_img = cv2.warpAffine(img, mat23, (size, size))

    return aligned_img


def align_face_5p(img, landmarks):
    """
    Aligns a face image based on 5-point facial landmarks using a fixed target size.

    Args:
    - img (numpy.ndarray): Input face image.
    - landmarks (list): List of 5-point facial landmarks.

    Returns:
    - numpy.ndarray: Aligned face image.
    """
    # Reshape the landmarks array into a 5x2 array
    lm5p = np.array(landmarks).reshape((5, 2))

    # Call align_given_lm5p function to perform alignment with a target size of 256x256 pixels
    aligned_img = align_given_lm5p(img, lm5p, 256)

    return aligned_img


def work(data_dir, out_dir, landmarks, i):
    """
    Process a single image from the CelebA dataset.

    Args:
    - data_dir (str): Path to the directory containing the CelebA dataset.
    - out_dir (str): Path to the output directory where processed images will be saved.
    - landmarks (list): List of landmarks corresponding to each image.
    - i (int): Index of the image to process.

    Returns:
    - int: Return code indicating the success of the processing (0 indicates success).
    """
    # Construct the source and destination image file paths
    src_imname = os.path.join(data_dir, 'data', '{:06d}.jpg'.format(i + 1))
    des_imname = os.path.join(out_dir, '{:06d}.jpg'.format(i + 1))

    # Read the image using OpenCV
    img = cv2.imread(src_imname)

    # Perform face alignment using the landmarks
    aligned_img = align_face_5p(img, landmarks[i])

    # Write the aligned image to the output directory
    cv2.imwrite(des_imname, aligned_img)

    # Return 0 to indicate success
    return 0


def main(data_dir, out_dir, thread_num):
    """
    Process images and landmarks from the CelebA dataset.

    Args:
    - data_dir (str): Path to the directory containing the CelebA dataset.
    - out_dir (str): Path to the output directory where processed images will be saved.
    - thread_num (int): Number of threads to use for parallel processing.

    Returns:
    - None
    """

    # Check if the output directory exists; if not, create it
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Read landmarks from the annotation file
    with open(os.path.join(data_dir, 'list_landmarks_celeba.txt'), 'r') as f:
        # Extract landmarks for each image and convert them to integers
        landmarks = [list(map(int, x.split()[1:11])) for x in f.read().strip().split('\n')[2:]]

    # Get the list of image files in the data directory
    im_list = glob.glob(os.path.join(data_dir, 'data/*.jpg'))

    # Initialize a multiprocessing Pool with the specified number of threads
    pool = Pool(thread_num)

    # Define a partial function to pass additional arguments to the work function
    partial_work = partial(work, data_dir, out_dir, landmarks)

    # Map the work function to process each image in parallel
    pool.map(partial_work, range(len(im_list)))

    # Close the pool of worker processes
    pool.close()

    # Wait for all worker processes to complete
    pool.join()


if __name__ == '__main__':
    '''
        The CUDA_VISIBLE_DEVICES environment variable is used to specify 
        which GPU devices are visible to CUDA-enabled applications. 
        By setting it to an empty string, you effectively disable CUDA support, 
        meaning that the script will run on CPU instead of GPU.
    '''
    os.environ["CUDA_VISIBLE_DEVICES"] = ''
    main('./datasets/celebA/', './datasets/celebA/align_5p/', 30)