import numpy as np
import PIL
import PIL.Image
import dlib
import cv2

predictor = dlib.shape_predictor("../image_utils/shape_predictor_68_face_landmarks.dat")


def get_landmark(pil_img):
    """
    :param pil_img: PIL image of the image which we want to find faces in
    :return: list of landmarks of the faces in the image
    """
    lm_list = []
    detector = dlib.get_frontal_face_detector()
    np_img = np.array(pil_img)
    dets = detector(np_img, 1)

    if (len(dets)) == 0:
        raise ValueError('No faces were found, skipping...')

    for d in dets:
        shape = predictor(np_img, d)
        t = list(shape.parts())
        a = []
        for tt in t:
            a.append([tt.x, tt.y])
        lm = np.array(a)
        lm_list.append(lm)

    return lm_list


def align_face(img_orginal):
    """
    :param img: a PIL image of the image we want to extract faces from
    :return 1: list of type PIL Image of the faces in the image
    :return 2: a list of numpy arrays, every array include 4 points of the face image location in the original image
    """

    imgs_list = []
    points_list = []
    lm_list = get_landmark(img_orginal)

    for lm in lm_list:
        img = img_orginal
        lm_eye_left = lm[36: 42]  # left-clockwise
        lm_eye_right = lm[42: 48]  # left-clockwise
        lm_mouth_outer = lm[48: 60]  # left-clockwise

        # Calculate auxiliary vectors.
        eye_left = np.mean(lm_eye_left, axis=0)
        eye_right = np.mean(lm_eye_right, axis=0)
        eye_avg = (eye_left + eye_right) * 0.5
        eye_to_eye = eye_right - eye_left
        mouth_left = lm_mouth_outer[0]
        mouth_right = lm_mouth_outer[6]
        mouth_avg = (mouth_left + mouth_right) * 0.5
        eye_to_mouth = mouth_avg - eye_avg

        # Choose oriented crop rectangle.
        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
        y = np.flipud(x) * [-1, 1]
        c = eye_avg + eye_to_mouth * 0.1
        quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
        qsize = np.hypot(*x) * 2

        output_size = 256

        # Shrink.
        shrink = int(np.floor(qsize / output_size * 0.5))
        if shrink > 1:
            rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
            img = img.resize(rsize, PIL.Image.ANTIALIAS)
            quad /= shrink
            qsize /= shrink

        # Crop.
        border = max(int(np.rint(qsize * 0.1)), 3)
        crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
                int(np.ceil(max(quad[:, 1]))))
        crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]),
                min(crop[3] + border, img.size[1]))

        if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
            img = img.crop(crop)
            quad -= crop[0:2]

        # Transform.
        quad += 0.5
        img = img.transform((output_size, output_size), PIL.Image.QUAD, (quad).flatten(), PIL.Image.BILINEAR)
        imgs_list.append(img)
        points_list.append(quad + crop[0:2])

    # Return aligned images and points.
    return imgs_list, points_list


def embed_faces(img, imgs_list, points_list):
    """
    :param img: the orginal image in IMAGE (PIL) format
    :param imgs_list: list of PIL images of all the faces in the image
    :param points_list: a list of numpy arrays, every array include 4 points of the face image location in the original image
    :return: Image (PIL) format with the embeded faces from the list
    """
    img = np.array(img.convert('RGB'))

    for face, points in zip(imgs_list, points_list):
        face = np.array(face)

        # define the four points in the face image (counterclockwise from top-left)
        pts_face = np.array([[0, 0], [0, face.shape[0]], [face.shape[1], face.shape[0]], [face.shape[1], 0]],
                            dtype=np.float32)
        pts_target = np.array(points, dtype=np.float32)

        # compute the perspective transform matrix
        M = cv2.getPerspectiveTransform(pts_face, pts_target)

        # warp the photo using the transform matrix
        warped = cv2.warpPerspective(face, M, (img.shape[1], img.shape[0]))
        mask = np.zeros_like(img)

        # fixing aliasing by making the mask a little smaller and creating mask
        b = 3
        pts_target[pts_target < 0] = 0
        pts_target[(pts_target[:, 0] > img.shape[1]), 0] = img.shape[1]
        pts_target[(pts_target[:, 1] > img.shape[0]), 1] = img.shape[0]
        border_fix = np.array([[b, b], [b, -b], [-b, -b], [-b, b]])

        cv2.fillPoly(mask, np.int32([pts_target + border_fix]), (255, 255, 255))

        # combine the warped photo and the target image using bitwise operations
        result_face = cv2.bitwise_and(mask, warped)
        mask = cv2.bitwise_not(mask)
        result_img = cv2.bitwise_and(img, mask)
        img = result_img + result_face

    return PIL.Image.fromarray(img)

