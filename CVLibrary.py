import cv2
import numpy as np
import math
import os


def affine_transformation(in_frame, dst_dim,
                          in_s = np.ones(2, np.float32),
                          in_t = np.zeros(2, np.float32),
                          in_theta = 0.0,
                          in_rc = np.zeros(2, np.float32),
                          in_sh = np.zeros(2, np.float32)):
    """in_s:     scaling factor
       in_t:     translation factor
       in_theta: degrees of rotation
       in_rc:    rotation center
       in_sh:    shear amount"""
    in_theta = in_theta * np.pi / 180.0
    print(in_t)
    As = np.float32(
        [
            [in_s[0] * np.cos(in_theta),    in_sh[0] + np.sin(in_theta),
             in_t[0] + in_rc[0] * (1.0 - np.cos(in_theta)) - in_rc[1] * np.sin(in_theta)],
            [in_sh[1] - np.sin(in_theta),   in_s[1] * np.cos(in_theta),
             in_t[1] + in_rc[0] * np.sin(in_theta) + in_rc[1] * (1.0 - np.cos(in_theta))]
        ])
    print(As)
    return cv2.warpAffine(in_frame, As, dst_dim, None, flags = cv2.INTER_LINEAR)


def brightness_modify(in_image, in_offset, in_type = "uint8"):
    if in_image is not None:
        if in_type == "uint8":
            if 0 <= in_offset <= 255:
                return cv2.add(in_image, np.ones(in_image.shape, dtype = "uint8") * in_offset)
            else:
                print("Offset value must be in range [0, 255].")
                return None
        elif in_type == "float32":
            if 0 <= in_offset <= 1:
                return cv2.add(in_image, np.ones(in_image.shape, dtype = "float32") * in_offset)
            else:
                print("Offset value must be in range [0, 1].")
                return None
        else:
            print("Wrong image data type.")
            return None
    else:
        print("Image is None.")
        return None


def concat_images(in_image_first, in_image_second):
    if in_image_first is not None and in_image_second is not None:
        first_type  = in_image_first.dtype
        second_type = in_image_second.dtype
        if first_type != second_type:
            if first_type == "uint8":
                converted = np.uint8(in_image_second * 255)
            else:
                converted = np.float32(in_image_second) / 255
            return cv2.hconcat([in_image_first, converted])
        else:
            return cv2.hconcat([in_image_first, in_image_second])
    else:
        print("At least one image is None.")


def contrast_enhancement(in_image, in_scale_factor = 0.0, out_data_type = "uint8"):
    if 0.0 < in_scale_factor < 1.0:
        if out_data_type == "uint8":
            return np.uint8(np.clip(in_image * (1 + in_scale_factor), 0, 255))
        elif out_data_type == "float32":
            return np.clip((in_image * (1 + in_scale_factor)) / 255.0, 0, 1)
        else:
            print("Not a valid out_data_type. Please specify 'uint8' or 'float32'.")
            return None
    else:
        print("Not a valid in_scale_factor. Value must be in the range [0, 1).")
        return None


def copy_video(in_video, out_video):
    while in_video.isOpened():
        read, frame = in_video.read()
        if read:
            out_video.write(frame)
        else:
            in_video.release()
            out_video.release()


def create_colors():
    return {
        "red": (0, 0, 255),
        "blue": (255, 0, 0),
        "green": (0, 255, 0),
        "yellow": (0, 255, 255),
        "cyan": (255, 255, 0),
        "magenta": (255, 0, 255),
        "black": (0, 0, 0),
        "white": (255, 255, 255)
    }


def createTrackerByName(in_trackerType):
    tracker_types = ["BOOSTING", "MIL", "KCF", "TLD", "MEDIANFLOW", "GOTURN", "CSRT", "MOSSE"]
    if in_trackerType   == tracker_types[0]:
        return cv2.TrackerBoosting_create()
    elif in_trackerType == tracker_types[1]:
        return cv2.TrackerMIL_create()
    elif in_trackerType == tracker_types[2]:
        return cv2.TrackerKCF_create()
    elif in_trackerType == tracker_types[3]:
        return cv2.TrackerTLD_create()
    elif in_trackerType == tracker_types[4]:
        return cv2.TrackerMedianFlow_create()
    elif in_trackerType == tracker_types[5]:
        return cv2.TrackerGOTURN_create()
    elif in_trackerType == tracker_types[6]:
        return cv2.TrackerCSRT_create()
    elif in_trackerType == tracker_types[7]:
        return cv2.TrackerMOSSE_create()
    else:
        print("Incorrect tracker name.")
        print("Available trackers are:")
        for trk in tracker_types:
            print(trk)
        return None


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


def display_image(in_frame       = None,    in_name = "Image", in_channels = False,
                  in_window_type = "fixed", in_wait = 0,       in_destroy  = True):
    window_types = {
        "fixed":   cv2.WINDOW_AUTOSIZE,
        "dynamic": cv2.WINDOW_NORMAL
    }
    type_exists = False
    for key in window_types.keys():
        if key == in_window_type:
            type_exists = True
            break
    if in_frame is None:
        print("There is no image to display")
    elif not type_exists:
        print("Wrong window type.\nPossible values: {fixed, dynamic}.")
    elif in_wait < 0:
        print("Negative time values are not allowed.")
    else:
        cv2.namedWindow(in_name, window_types[in_window_type])
        if in_channels:
            cv2.imshow(in_name, cv2.hconcat(cv2.split(in_frame)))
        else:
            cv2.imshow(in_name, in_frame)
        k = cv2.waitKey(in_wait)
        if in_wait != 0:
            return k
        if in_destroy:
            cv2.destroyWindow(in_name)


def display_image_features(in_image):
    print("Data type: {}".format(in_image.dtype))
    print("Object type: {}".format(type(in_image)))
    print("Image dimensions: {}".format(in_image.shape))
    print("Image max value: {}  ".format(in_image.max()))
    print("Image min value: {}\n".format(in_image.min()))


def display_video(in_video, in_name = "Video", in_wait = 25):
    cv2.namedWindow(in_name)
    cv2.namedWindow(in_name)
    while in_video.isOpened():
        read, frame = in_video.read()
        if read:
            cv2.imshow(in_name, frame)
            cv2.waitKey(in_wait)
        else:
            in_video.release()


def fixBorder(in_frame):
    n = in_frame.shape
    T = cv2.getRotationMatrix2D((n[1] / 2, n[0] / 2), 0, 1.04)
    frame = cv2.warpAffine(in_frame, T, (n[1], n[0]))
    return frame


def flann_feature_match(in_flann, in_des1, in_des2, in_k = 2):
    matches = in_flann.knnMatch(np.float32(in_des1),
                                np.float32(in_des2),
                                k = in_k)
    bestMatches = []
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            bestMatches.append(m)
            print(m.distance, n.distance, 0.7 * n.distance)
    return bestMatches


def get_image_features(in_image):
    return {
        "type":   in_image.dtype,
        "format": type(in_image),
        "shape":  in_image.shape,
        "max":    in_image.max(),
        "min":    in_image.min()
    }


def get_video_features(in_video):
    return {
        "position": in_video.get(cv2.CAP_PROP_POS_MSEC),
        "width":    int(in_video.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height":   int(in_video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps":      in_video.get(cv2.CAP_PROP_FPS),
        "fourcc":   in_video.get(cv2.CAP_PROP_FOURCC),
        "frames": int(in_video.get(cv2.CAP_PROP_FRAME_COUNT))
    }


def HogPrepareData(in_data):
    featureVectorLength = len(in_data[0])
    features = np.float32(in_data).reshape(-1, featureVectorLength)
    return features


def HogCompute(in_hog, in_data):
    hogData = []
    for image in in_data:
        hogData.append(in_hog.compute(image))
    return hogData


def hough_lines(in_image, in_min_thresh = 50, in_max_thresh = 200, in_voters = 100, in_min_length = 10, in_max_gap = 250):
    image_gray = cv2.cvtColor(in_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(image_gray, in_min_thresh, in_max_thresh)
    return cv2.HoughLinesP(edges, 1, np.pi / 180, in_voters,
                           minLineLength = in_min_length, maxLineGap = in_max_gap)


def hough_circles(in_image, in_blur = False, in_blur_thresh = 5, in_image_scale = 1, in_centers_distance = 50,
                  in_param1 = 450, in_param2 = 10, in_min_radius = 30, in_max_radius = 40):
    image_gray = cv2.cvtColor(in_image, cv2.COLOR_BGR2GRAY)
    if in_blur:
        image_gray = cv2.medianBlur(image_gray, in_blur_thresh)
    return cv2.HoughCircles(image_gray, cv2.HOUGH_GRADIENT, in_image_scale, in_centers_distance,
                            param1 = in_param1, param2 = in_param2,
                            minRadius = in_min_radius, maxRadius = in_max_radius)


def mouse_draw_circle(action, x, y, flags, userdata):
    if action == cv2.EVENT_LBUTTONDOWN:
        userdata["center"] = [(x, y)]
        cv2.circle(userdata["image"], userdata["center"][0], 1, (255, 255, 0), 2, cv2.LINE_AA)
    elif action == cv2.EVENT_LBUTTONUP:
        userdata["circumference"] = [(x, y)]
        radius = math.sqrt(math.pow(userdata["center"][0][0] - userdata["circumference"][0][0], 2) +
                           math.pow(userdata["center"][0][1] - userdata["circumference"][0][1], 2))
        cv2.circle(userdata["image"], userdata["center"][0], int(radius), (0, 255, 0), 2, cv2.LINE_AA)


def movingAverage(in_curve, in_radius):
    window_size = 2 * in_radius + 1
    f = np.ones(window_size) / window_size
    curve_pad = np.lib.pad(in_curve, (in_radius, in_radius), "edge")
    curve_smoothed = np.convolve(curve_pad, f, mode = "same")
    curve_smoothed = curve_smoothed[in_radius: -in_radius]
    return curve_smoothed


def smooth(in_trajectory, in_SMOOTHING_RADIUS):
    smoothed_trajectory = np.copy(in_trajectory)
    for i in range(3):
        smoothed_trajectory[:, i] = movingAverage(in_trajectory[:, i],
                                                  in_radius = in_SMOOTHING_RADIUS)
    return smoothed_trajectory


def svmInit(in_c, in_gamma, in_kernel):
    model = cv2.ml.SVM_create()
    model.setGamma(in_gamma)
    model.setC(in_c)
    model.setKernel(in_kernel)
    model.setType(cv2.ml.SVM_C_SVC)
    model.setTermCriteria((cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 1e-3))
    return model


def svmTrain(in_model, in_samples, in_responses):
    in_model.train(in_samples, cv2.ml.ROW_SAMPLE, in_responses)
    return in_model


def svmPredict(in_model, in_samples):
    return in_model.predict(in_samples)[1]


def svmEvaluate(in_model, in_samples, in_labels):
    labels = in_labels[:, np.newaxis]
    predictions = svmPredict(in_model, in_samples)
    correct = np.sum((labels == predictions))
    errors  = (labels != predictions).mean()
    print('label -- 1:{}, -1:{}'.format(np.sum(predictions == 1),
                                        np.sum(predictions == -1)))
    return correct, errors * 100

