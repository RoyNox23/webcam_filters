import numpy as np
import cv2
import matplotlib.pyplot as plt
from CVLibrary import *


def print_instructions():
    print("Press 0 to remove filters.")
    print("Press 1 to show a pencil like sketch video.")
    print("Press 2 to show a cartoonified video.")
    print("Press 3 to show a face detection and smoothing video.")
    print("Press 4 to show a face detection sunglass patching video.")
    print("Press ESC to end the program")


def pencilSketch(in_image):
    gray_image  = cv2.cvtColor(in_image, cv2.COLOR_BGR2GRAY)
    gauss_blur  = cv2.GaussianBlur(gray_image, (5, 5), 0, 0)
    laplacian   = cv2.Laplacian(gauss_blur, cv2.CV_32F, ksize = 5, scale = 1, delta = 0)
    normalized  = (cv2.normalize(laplacian, dst  = laplacian,
                                 alpha = 0, beta = 1,
                                 norm_type = cv2.NORM_MINMAX,
                                 dtype     = cv2.CV_32F) * 255).astype(np.uint8)
    vals, counts = np.unique(normalized, return_counts = True)
    thresh_val = vals[np.argmax(counts)]
    ret, thresh = cv2.threshold(normalized, thresh_val + 3, 255, cv2.THRESH_BINARY)
    return cv2.cvtColor(cv2.bitwise_not(thresh), cv2.COLOR_GRAY2BGR)


def cartoonify(in_image, arguments = 0):
    return cv2.bitwise_and(cv2.bilateralFilter(in_image, 15, 80, 80), pencilSketch(in_image))


def faceTracking(in_image, in_tracker):
    track_ok, bboxes = in_tracker.update(in_image)
    return track_ok, bboxes


def face_detect(in_image, in_model = None, in_params = None):
    image_gray = cv2.cvtColor(in_image, cv2.COLOR_BGR2GRAY)
    for neigh in range(in_params["faceNeighborsMin"], in_params["faceNeighborsMax"], in_params["neighborStep"]):
        faces = in_model.detectMultiScale(image_gray, 1.2, neigh)
        if neigh == in_params["faceNeighborsMax"] - 1:
            image_clone = np.copy(in_image)
    return faces, image_clone


def faceLocation(in_image, in_model, in_params, in_tracker, in_track_type):
    boxes, image = face_detect(in_image, in_model, in_params)
    for bbox in boxes:
        in_tracker.add(createLegacyTrackerByName(in_track_type), image, tuple(bbox))
    if len(boxes) > 0:
        return True, boxes
    else:
        return False, boxes


def face_filtering(in_image, in_bbox, in_filter_params):
    x, y, h, w = in_bbox
    face = in_image[y: y + h, x: x + w]
    hsv_face = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
    face_patch = hsv_face[int(2 * h / 5): int(4 * h / 5), int(w / 5): int(4 * w / 5)]
    hist = cv2.calcHist([face_patch], [0], None, [180], [0, 180])
    hue_bin = np.argmax(hist)
    if hue_bin - 20 < 0:
        low_hue = 0
    else:
        low_hue = hue_bin - 20
    if hue_bin + 20 >= 180:
        up_hue = 179
    else:
        up_hue = hue_bin + 20
    lower_patch = np.array([low_hue, 30, 30],  dtype = "int")
    upper_patch = np.array([up_hue, 220, 220], dtype = "int")
    face_range = cv2.inRange(hsv_face, lower_patch, upper_patch)
    face_mask  = cv2.merge([face_range, face_range, face_range])
    face_back  = cv2.bitwise_not(face_mask)
    new_face   = cv2.bitwise_and(face_mask, face).copy()
    new_back   = cv2.bitwise_and(face_back, face).copy()
    dim = in_filter_params["dimension"]
    s   = in_filter_params["sigma"]
    new_face   = cv2.bilateralFilter(new_face, dim, s, s)
    return cv2.add(new_face, new_back)


def shades_mask(in_image):
    hsv_sung = cv2.cvtColor(in_image[:, :, :3], cv2.COLOR_BGR2HSV)
    lower_patch = np.array([80, 120, 120],  dtype = "int")
    upper_patch = np.array([130, 255, 255], dtype = "int")
    sg_range = cv2.inRange(hsv_sung, lower_patch, upper_patch)
    return sg_range


def sunglass_filter(in_image, in_bbox, in_shades, in_frame):
    x, y, h, w = in_bbox
    face = in_image[y: y + h, x: x + w]
    face_patch   = face[int(1 * h / 5): int(3 * h / 5), int(w / 18): int(17 * w / 18)].copy()
    new_dim      = (int(17 * w / 18) - int(w / 18), int(3 * h / 5) - int(1 * h / 5))
    rsz_shades   = cv2.resize(in_shades, new_dim, interpolation = cv2.INTER_AREA)
    rsz_frame    = cv2.resize(in_frame, new_dim, interpolation = cv2.INTER_AREA)
    sg_back_1    = cv2.bitwise_not(cv2.merge([rsz_frame[:, :, 3], rsz_frame[:, :, 3], rsz_frame[:, :, 3]]))
    masked_patch = cv2.bitwise_and(face_patch, sg_back_1)
    frame_patch  = cv2.add(masked_patch, rsz_frame[:, :, :3])
    frame_patch_2 = cv2.addWeighted(frame_patch, 1, rsz_shades[:, :, :3], 0.85, 0)
    face[int(1 * h / 5): int(3 * h / 5), int(w / 18): int(17 * w / 18)] = frame_patch_2
    in_image[y: y + h, x: x + w] = face
    return in_image


def createLegacyTrackerByName(in_trackerType):
    tracker_types = ["BOOSTING", "MIL", "KCF", "TLD", "MEDIANFLOW", "GOTURN", "CSRT", "MOSSE"]
    if in_trackerType   == tracker_types[0]:
        return cv2.legacy.TrackerBoosting_create()
    elif in_trackerType == tracker_types[1]:
        return cv2.legacy.TrackerMIL_create()
    elif in_trackerType == tracker_types[2]:
        return cv2.legacy.TrackerKCF_create()
    elif in_trackerType == tracker_types[3]:
        return cv2.legacy.TrackerTLD_create()
    elif in_trackerType == tracker_types[4]:
        return cv2.legacy.TrackerMedianFlow_create()
    elif in_trackerType == tracker_types[5]:
        return cv2.legacy.TrackerGOTURN_create()
    elif in_trackerType == tracker_types[6]:
        return cv2.legacy.TrackerCSRT_create()
    elif in_trackerType == tracker_types[7]:
        return cv2.legacy.TrackerMOSSE_create()
    else:
        print("Incorrect tracker name.")
        print("Available trackers are:")
        for trk in tracker_types:
            print(trk)
        return None


if __name__ == "__main__":
    model_name = "./model/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(model_name)
    haar_params = {
        "faceNeighborsMin": 2,
        "faceNeighborsMax": 3,
        "neighborStep"    : 1
    }

    trackerType = "MEDIANFLOW"

    sg_image      = cv2.imread("./images/sunglass.png", cv2.IMREAD_UNCHANGED)
    sg_shade_mask = shades_mask(sg_image)
    sg_frame_mask = cv2.bitwise_xor(sg_image[:, :, 3], sg_shade_mask)
    sg_shades     = cv2.bitwise_and(sg_image, cv2.merge([sg_shade_mask, sg_shade_mask,
                                                         sg_shade_mask, sg_shade_mask]))
    sg_frame      = cv2.bitwise_and(sg_image, cv2.merge([sg_frame_mask, sg_frame_mask,
                                                         sg_frame_mask, sg_frame_mask]))

    filter_params = {"dimension": 11,
                     "sigma":     15}

    window_name = "WebCam Video"
    cv2.namedWindow(window_name)
    wc      = cv2.VideoCapture(0)
    k       = 0
    flag    = 0
    success = True
    count   = 0

    print_instructions()

    while True:
        ok, frame = wc.read()
        k = cv2.waitKey(10)
        if not ok or k == 27:
            cv2.destroyWindow(window_name)
            wc.release()
            break
        elif k == 48:
            flag = 0
        elif k == 49:
            flag = 1
        elif k == 50:
            flag = 2
        elif k == 51:
            flag = 3
        elif k == 52:
            flag = 4

        if flag == 0:
            cv2.imshow(window_name, frame)

        elif flag == 1:
            cv2.imshow(window_name, pencilSketch(frame))

        elif flag == 2:
            cv2.imshow(window_name, cartoonify(frame))

        elif flag == 3 or flag == 4:
            if k == 51 or not success or count % 24 == 0:
                multiTracker = cv2.legacy.MultiTracker_create()
                success, bboxes = faceLocation(frame, faceCascade, haar_params, multiTracker, trackerType)
                count = 1
            else:
                success, bboxes = faceTracking(frame, multiTracker)
                count += 1
            if success and flag == 3:
                for bx in bboxes.astype(int):
                    if bx[0] < 0:
                        bx[0] = 0
                    if bx[1] < 0:
                        bx[1] = 0
                    frame[bx[1]: bx[1] + bx[3], bx[0]: bx[0] + bx[2]] = face_filtering(frame, bx, filter_params)
            if success and flag == 4:
                for bx in bboxes.astype(int):
                    if bx[0] < 0:
                        bx[0] = 0
                    if bx[1] < 0:
                        bx[1] = 0
                    frame = sunglass_filter(frame, bx, sg_shades, sg_frame)
            cv2.imshow(window_name, frame)




