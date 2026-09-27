import os
import configparser
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import roc_curve, auc, det_curve
from matplotlib import cm

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# 대상 이미지 경로 불러오기 #


def get_image_path(root_folder):
    image_path = []
    valid_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')
    for root, _, files in os.walk(root_folder):
        root = os.path.normpath(root).replace("\\", "/")
        for file in files:
            if file.lower().endswith(valid_extensions):
                file_path = os.path.join(root, file).replace("\\", "/")
                image_path.append(file_path)
    return image_path

# Config.ini 파일 읽어오기 #


def select_config(file_name):
    config = configparser.ConfigParser()
    config.read(file_name, encoding='UTF-8')

    # SIFT Parameters #
    nfeatures = config.getint('SIFT', 'nfeatures')
    nOctaveLayers = config.getint('SIFT', 'nOctaveLayers')
    contrastThreshold = config.getfloat('SIFT', 'contrastThreshold')
    edgeThreshold = config.getfloat('SIFT', 'edgeThreshold')
    sigma = config.getfloat('SIFT', 'sigma')

    sift_parameters = [nfeatures, nOctaveLayers,
                       contrastThreshold, edgeThreshold, sigma]

    # Matching Parameter #
    ratio_test = config.getfloat('ratio', 'ratio_test')

    # Matching Threshold #
    threshold = config.getint('match', 'threshold')

    # Resizing Parameter #
    resize_size = config.getint('resize', 'size')
    return sift_parameters, ratio_test, threshold, resize_size

# SIFT #
# sift_parameters : config file parameters #
# SIFT 키포인트 및 디스크립터 추출 #


def compute_sift(image, sift_parameters):
    sift = cv2.SIFT_create(sift_parameters[0], sift_parameters[1],
                           sift_parameters[2], sift_parameters[3], sift_parameters[4])
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors

# Keypoint Extraction and Save as pickle file #


def keypoint_extraction(img_files, resize_size, sift_parameters, ratio_test, save_dir='.'):
    os.makedirs(save_dir, exist_ok=True)
    keypoint_filename = os.path.join(save_dir, f'resize_{str(resize_size)}_{str(sift_parameters[0])}_{str(sift_parameters[1])}_{str(sift_parameters[2])}_{str(ratio_test)}.pkl')
    key_des_list = []
    for i, img in enumerate(img_files):
        print(img)
        image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        if resize_size != 0:
            qh, qw = image.shape[:2]
            if qh > resize_size:
                qratio = qh/qw
                nqh, nqw = resize_size, int(resize_size/qratio)
                image = cv2.resize(image, (nqw, nqh),
                                   interpolation=cv2.INTER_AREA)
        keypoints, descriptors = compute_sift(image, sift_parameters)
        keypoint_data = []
        for kp in keypoints:
            keypoint_data.append([
                kp.pt,
                kp.size,
                kp.angle,
                kp.response,
                kp.octave,
                kp.class_id
            ])
        key_des_list.append([keypoint_data, descriptors])

    # Save keypoint file as pickle file #
    with open(keypoint_filename, 'wb') as f:
        pickle.dump(key_des_list, f)

    return keypoint_filename

# Key point matching #


def match_descriptors(des1, des2, ratio_test, k=2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=k)
    key_matches = []
    for match_set in matches:
        if len(match_set) >= 2 and match_set[0].distance < ratio_test * match_set[1].distance:
            key_matches.append(match_set[0])
    return key_matches

# Convert Pickle Data into Keypoint #


def keypoints_from_data(keypoint_data):
    keypoints = []
    for kp_data in keypoint_data:
        pt = tuple(kp_data[0])
        size = kp_data[1]
        angle = kp_data[2]
        response = kp_data[3]
        octave = int(kp_data[4])
        class_id = int(kp_data[5])
        kp = cv2.KeyPoint(x=pt[0], y=pt[1], size=size, angle=angle, response=response,
                          octave=octave, class_id=class_id)
        keypoints.append(kp)
    return keypoints

# Make Score Matrix by Matching Result #


def score_matrix(keypoint_file_path, ratio_test, resize_size, sift_parameters, save_dir='.'):
    with open(keypoint_file_path, 'rb') as f:
        loaded_list = pickle.load(f)
    num_images = len(loaded_list)

    # Make Score Directory #
    # ex) dir_path = "폴더 경로"
    # if not os.path.exists(dir_path):
    # os.mkdir(dir_path)

    keypoints_list, descriptors_list = [], []
    for keypoint_data, descriptors in loaded_list:
        keypoints_list.append(keypoints_from_data(keypoint_data))
        descriptors_list.append(descriptors)

    # Score Matrix Init #
    score_matrix = np.zeros((num_images, num_images))

    # Matching #
    for i in range(num_images):
        for j in range(num_images):
            if i == j:
                score_matrix[i, j] = 1
            else:
                key_matches = match_descriptors(
                    descriptors_list[i], descriptors_list[j], ratio_test)
                match_count = len(key_matches)
                if match_count > 4:  # Least 4 matches
                    src_pts = np.float32(
                        [keypoints_list[i][m.queryIdx].pt for m in key_matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32(
                        [keypoints_list[j][m.trainIdx].pt for m in key_matches]).reshape(-1, 1, 2)
                    _, mask = cv2.findHomography(
                        src_pts, dst_pts, cv2.RANSAC, 10.0)
                    inliers = mask.ravel().tolist()
                    # inlier_ratio = inliers.count(1) / len(keypoints_list[i])
                    score_matrix[i, j] = inliers.count(1)
                else:
                    score_matrix[i, j] = 0

    # Save Score Matrix as numpy file #
    os.makedirs(save_dir, exist_ok=True)
    score_matrix_path = os.path.join(save_dir, f'{str(resize_size)}_{str(sift_parameters[0])}_{str(sift_parameters[1])}_{str(sift_parameters[2])}_{str(ratio_test)}_.npy')
    np.save(score_matrix_path, score_matrix)
    return score_matrix_path

# Load Score Matrix Files #


def list_npy_files(directory):
    files = os.listdir(directory)
    npy_files = [os.path.join(directory, file).replace('\\', '/')
                 for file in files if file.endswith('.npy')]
    return npy_files

# Draw the distribution of both genuine and impost scores #


def both_score(genuine_scores, imposter_scores, score_file_name, save_dir='.'):
    g_mean_score = np.mean(genuine_scores)
    g_std_deviation = np.std(genuine_scores)
    g_dist = stats.norm(loc=g_mean_score, scale=g_std_deviation)
    gx = np.linspace(g_dist.ppf(0.001), g_dist.ppf(0.999), 100)

    i_mean_score = np.mean(imposter_scores)
    i_std_deviation = np.std(imposter_scores)
    i_dist = stats.norm(loc=i_mean_score, scale=i_std_deviation)
    ix = np.linspace(i_dist.ppf(0.001), i_dist.ppf(0.999), 100)

    plt.figure(figsize=(10, 6))

    plt.plot(gx, g_dist.pdf(gx), 'b-', lw=2,
             label=f'Genuine Distribution (mean={g_mean_score:.4f}, std={g_std_deviation:.4f})')
    plt.axvline(g_mean_score, color='blue', linestyle='dashed',
                linewidth=1.5, label='Genuine Mean')
    plt.axvline(g_mean_score + g_std_deviation*2, color='green',
                linestyle='dashed', linewidth=1.5, label='Genuine Mean 95.4%')
    plt.axvline(g_mean_score - g_std_deviation*2, color='orange',
                linestyle='dashed', linewidth=1.5, label='Genuine Mean 95.4%')

    plt.plot(ix, i_dist.pdf(ix), 'r-', lw=2,
             label=f'Imposter Distribution (mean={i_mean_score:.4f}, std={i_std_deviation:.4f})')
    plt.axvline(i_mean_score, color='red', linestyle='dashed',
                linewidth=1.5, label='Imposter Mean')
    plt.axvline(i_mean_score + i_std_deviation*2, color='purple',
                linestyle='dashed', linewidth=1.5, label='Imposter Mean 95.4%')
    plt.axvline(i_mean_score - i_std_deviation*2, color='brown',
                linestyle='dashed', linewidth=1.5, label='Imposter Mean 95.4%')

    plt.scatter(genuine_scores, np.full_like(genuine_scores, 0.02),
                color='blue', alpha=0.6, label='Genuine Scores')
    plt.scatter(imposter_scores, np.full_like(imposter_scores, 0.025),
                color='red', alpha=0.6, label='Imposter Scores')

    plt.xlabel('Score')
    plt.ylabel('Probability Density')
    plt.title(f'Scores_{score_file_name}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'genuine_vs_imposter_{score_file_name}.jpg'))
    plt.show()

# Draw ROC curve #


def draw_entire_roc_curve(scores, filenames, dir_path):
    colormap = cm.get_cmap('tab10', len(scores))
    score_num = len(scores)
    plt.figure()
    y_true = np.array([1] * len(scores[0][0]) + [0] * len(scores[0][1]))
    i = 0
    target_fpr = 0.1
    for (genuine_scores, imposter_scores), filename in zip(scores, filenames):
        y_scores = np.array(genuine_scores + imposter_scores)
        color = colormap(i / score_num)
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        closest_index = np.argmin(np.abs(fpr-target_fpr))
        threshold_at_target_fpr = thresholds[closest_index]
        tpr_at_target_fpr = tpr[closest_index]
        print(
            f'Model_{filename}: FPR {target_fpr} 일 때의 threshold 값: {threshold_at_target_fpr}, TPR 값 : {tpr_at_target_fpr}')

        plt.plot(fpr, tpr, color=color, lw=1.5,
                 label=f'Model_{filename} (AUC = {roc_auc:.4f})')
        i = i+1

    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'(ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid()
    plt.savefig(f'{dir_path}/Entire_ROC.jpg')
    plt.show()

# Draw DET curve #


def draw_entire_det_curve(scores, params, dir_path):
    colormap = cm.get_cmap('tab10', len(scores))
    score_num = len(scores)
    plt.figure()
    y_true = np.array([1] * len(scores[0][0]) + [0] * len(scores[0][1]))
    i = 0
    for (genuine_scores, imposter_scores), params in zip(scores, parameters):
        y_scores = np.array(genuine_scores + imposter_scores)
        color = colormap(i / score_num)
        fpr, fnr, thresholds = det_curve(y_true, y_scores)
        det_auc = auc(fpr, fnr)
        plt.plot(fpr, fnr, color=color, lw=1.5,
                 label=f'Model_{params[0]}_{params[1]}_{params[2]}_{params[3]} (AUC = {det_auc:.4f})')
        i = i+1

    plt.xlabel('False Positive Rate')
    plt.ylabel('False Negative Rate')
    plt.title(f'(DET) Curve')
    plt.legend()
    plt.grid()
    plt.savefig(f'{dir_path}/Entire_DET.jpg')
    plt.show()

# Call both_score function to make distribution graph #


def score_graph(score_matrigx, score_file_name, indices_path, save_dir='.'):
    loaded_arr = np.load(indices_path, allow_pickle=True)
    indices = loaded_arr.tolist()
    num_images = len(indices)

    genuine_scores, imposter_scores = [], []
    genuine_indices = []
    for i in range(num_images):
        genuine_list = indices[i]
        imposter_list = list(range(num_images))
        imposter_list = list(
            filter(lambda gx: gx not in indices[i]+[i], imposter_list))
        # genuine scores #
        for gi in genuine_list:
            genuine_scores.append(score_matrigx[i, gi])
            genuine_indices.append((i, gi))
        # imposter scores #
        for ii in imposter_list:
            imposter_scores.append(score_matrigx[i, ii])

    both_score(genuine_scores, imposter_scores, score_file_name, save_dir=save_dir)
    return genuine_scores, imposter_scores


# For문의 형태로 변형 시 #
# 여러 파라미터 조합에 대한 Keypoint, Score Matrix 저장 #
# 여러 파라미터 조합에 대한 ROC, DET 커브를 한 번에 확인할 수 있음 #
if __name__ == "__main__":
    # Image Folder Path #
    # Config.ini path #
    image_folder_path = input("Image Folder Path: ")
    config_file_path = input("Config File Path: ")
    base_dir = input("Output Base Directory (keypoints/scores/graph will be created here): ").strip() or "."
    indices_path = input("Indices File Path (.npy): ").strip()

    # Image Files #
    image_files = get_image_path(image_folder_path)
    image_files.sort()

    # Read Config.ini file #
    sift_parameters, ratio_test, threshold, resize_size = select_config(
        config_file_path)

    # Call function for keypoint extraction and saving pickle file #
    keypoint_file_path = keypoint_extraction(
        image_files, resize_size, sift_parameters, ratio_test,
        save_dir=os.path.join(base_dir, "keypoints"))

    # Call function for building score matrix #
    score_matrix_path = score_matrix(
        keypoint_file_path, ratio_test, resize_size, sift_parameters,
        save_dir=os.path.join(base_dir, "scores"))

    score_dir = os.path.dirname(score_matrix_path)
    score_list = list_npy_files(score_dir)
    dir_path = os.path.join(base_dir, "graph")
    os.makedirs(dir_path, exist_ok=True)

    scores, parameters = [], []
    filenames = []
    for score_file_name in score_list:
        print(score_file_name)
        file_name = score_file_name.split("/")[-1]
        # Resize_size, Nfeatures, Num of Octave Layers, Contrast Threshold, Ratio Test #
        rs, nf, no, ct, ratio = file_name.split("_")[0:5]
        params = (rs, nf, no, ct, ratio)
        loaded_score = np.load(score_file_name)
        genuine, imposter = score_graph(loaded_score, file_name, indices_path,
                                        save_dir=dir_path)
        scores.append((genuine, imposter))
        parameters.append(params)
        filenames.append(file_name)
    draw_entire_roc_curve(scores, filenames, dir_path)
    draw_entire_det_curve(scores, parameters, dir_path)
