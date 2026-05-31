import os
import cv2
import numpy as np
import tempfile
import shutil
import time
import configparser
import pickle

# Load Images from Folder #
# If true => Whole Folder # 
# Else => Read one image #
def load_images_from_folder(folder_or_file):
    images = []
    if os.path.isdir(folder_or_file):
        folder = folder_or_file
        for filename in os.listdir(folder):
            img_path = os.path.join(folder, filename)
            file_data = np.fromfile(img_path, np.uint8)
            img = cv2.imdecode(file_data, cv2.IMREAD_COLOR)
            if img is not None:
                images.append((filename, img))
    else:
        # Input is a single image file
        img_path = folder_or_file
        filename = os.path.basename(img_path)
        file_data = np.fromfile(img_path, np.uint8)
        img = cv2.imdecode(file_data, cv2.IMREAD_COLOR)
        if img is not None:
            images.append((filename, img))
    
    return images

# SIFT #
# sift_parameters : from config file #
# SIFT Keypoint, Descriptor Extraction #
def compute_sift(image, sift_parameters):    
    sift = cv2.SIFT_create(nfeatures=sift_parameters[0], nOctaveLayers=sift_parameters[1], contrastThreshold=sift_parameters[2], edgeThreshold=sift_parameters[3], sigma=sift_parameters[4])
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors

# Key point matching #
# knnMatching (k=2 default) #
# Matching Function  #
# ####################################### #  
def match_descriptors(des1, des2, ratio_test, k=2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=k)
    key_matches = []
    for match_set in matches:
        # ratio test #
        if len(match_set) >= k and match_set[0].distance < ratio_test * match_set[1].distance:
            key_matches.append(match_set[0])
    return key_matches
# ######################################## #

# Draw matching points on image #
# Save Matching Result #
def draw_matches(query_image, query_keypoints, comparison_image, comparison_keypoints, key_matches, output_path):
    match_img = cv2.drawMatches(query_image, query_keypoints, comparison_image, comparison_keypoints, key_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
        temp_file_path = tmp_file.name
        cv2.imwrite(temp_file_path, match_img)
    shutil.move(temp_file_path, output_path)

# Ranking, Matching and Visualization #
# Call SIFT #
# Call Matching Function #
def rank_images_and_visualize(query_image, source_images, save_path, sift_parameters, ratio_test, threshold, resize_size):
    max_matches = 0

    if resize_size!=0:
        qh, qw  = query_image.shape[:2]
        if qh>resize_size:
            qratio = qh/qw
            nqh, nqw = resize_size, int(resize_size/qratio)
            query_image = cv2.resize(query_image, (nqw, nqh), interpolation=cv2.INTER_AREA)
    # Query Image Keypoint 및 Descriptor 추출 #   
    query_keypoints, query_descriptors = compute_sift(query_image, sift_parameters)

    scores = []
    matching = []

    # Source Image Keypoint 및 Descriptor 추출 #
    # 추출한 descriptor로 매칭 #
    for filename, image in source_images:
        if resize_size!=0:
            oh, ow = image.shape[:2]
            if oh>resize_size:
                nratio = oh/ow
                nh, nw = resize_size, int(resize_size/nratio)
                image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
        keypoints, descriptors = compute_sift(image, sift_parameters)

        # Query Descriptors와 Source Descriptors 매칭 #
        # ########################################## #
        key_matches = match_descriptors(query_descriptors, descriptors, ratio_test)
        # ########################################## #

        match_count = len(key_matches)
        if match_count > 4:  
            src_pts = np.float32([query_keypoints[m.queryIdx].pt for m in key_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypoints[m.trainIdx].pt for m in key_matches]).reshape(-1, 1, 2)
            _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 10.0)
            inliers = mask.ravel().tolist()
            # Threshold 이상 매칭이 된 경우만 저장 #
            # ################################## #
            if inliers.count(1)>=threshold:
                inlier_matches = [m for i, m in enumerate(key_matches) if inliers[i] == 1]
                matching.append((image, keypoints, inlier_matches, inliers.count(1)))
                scores.append((filename, inliers.count(1)))
                if inliers.count(1) > max_matches:
                    max_matches = inliers.count(1)
            # ################################## #
    
    # 매칭 결과 저장 #        
    # Ranking을 위해서 scores, matching 결과 정렬 #
    scores.sort(key=lambda x: x[1], reverse=True)
    matching.sort(key=lambda x: x[3], reverse=True)

    # 이미지에 매칭 포인트 그리고 이미지 저장 #
    for rank, match in enumerate(matching):
        output_path = os.path.join(save_path, f"rank{rank+1:05d}_matches_{scores[rank][0]}")
        draw_matches(query_image, query_keypoints, match[0], match[1], match[2], output_path)
        warping_images(query_image, match[0], query_keypoints, match[1], match[2], rank+1, scores[rank][0], save_path, alpha=0.5)
    return scores, max_matches

# Align and warp(project) images on query image #
def warping_images(img1, img2, kp1, kp2, matches, rank, filename, save_path, alpha):
    # Find keypoint coordinate #
    matches = sorted(matches, key=lambda x : x.distance)
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)
    
    # Compute Homography matrix #
    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 10.0, maxIters=40000)
    # Align image #
    green_color = (0, 255, 0)
    im2h, im2w = img2.shape[:2]
    corners = np.float32([[0, 0], [0, im2h - 1], [im2w - 1, im2h - 1], [im2w - 1, 0]]).reshape(-1, 1, 2)

    height, width = img1.shape[:2]
    aligned_img = cv2.warpPerspective(img2, H, (width, height))

    # Alpha blending #
    blended_img = cv2.addWeighted(aligned_img, alpha, img1, 0.5, 0)

    # Compute Intersection Area #
    transformed_corners = cv2.perspectiveTransform(corners, H)
    img1_corners = np.float32([[0, 0], [0, height - 1], [width - 1, height - 1], [width - 1, 0]]).reshape(-1, 1, 2)
    intersect_area, intersect_corners = cv2.intersectConvexConvex(np.int32(transformed_corners), np.int32(img1_corners))
    intersect_corners = np.round(intersect_corners).astype(np.int32)
    align_path = os.path.join(save_path, f"rank{rank:05d}_align_{filename}")
    if intersect_area > 0:
        # Draw the intersection area on aligned image #
        blended_img_with_intersection = blended_img.copy()
        cv2.polylines(blended_img_with_intersection, [intersect_corners], True, green_color, thickness=5)

        # Gray Filter on unoverlapped region #
        mask = np.zeros(blended_img_with_intersection.shape[:2], dtype=np.uint8)
        cv2.fillConvexPoly(mask, intersect_corners, 255)
        gray = np.full_like(blended_img_with_intersection, [128, 128, 128], dtype=np.uint8)
        non_intersecting_region = mask != 255
        if not np.all(non_intersecting_region==False):
            blended_img_with_intersection[non_intersecting_region] = cv2.addWeighted(blended_img_with_intersection[non_intersecting_region], 0.5, gray[non_intersecting_region], 0.5, 0)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            temp_file_path = tmp_file.name
            cv2.imwrite(temp_file_path, blended_img_with_intersection)
        shutil.move(temp_file_path, align_path)
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            temp_file_path = tmp_file.name
            cv2.imwrite(temp_file_path, blended_img)
        shutil.move(temp_file_path, align_path)


# Read Config.ini file #
def select_config(file_name):
   config = configparser.ConfigParser()
   config.read(file_name)

   # SIFT Parameters #
   nfeatures = config.getint('SIFT', 'nfeatures')
   nOctaveLayers = config.getint('SIFT', 'nOctaveLayers')
   contrastThreshold = config.getfloat('SIFT', 'contrastThreshold')
   edgeThreshold = config.getfloat('SIFT', 'edgeThreshold')
   sigma = config.getfloat('SIFT', 'sigma')

   sift_parameters = [nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma]

   # Matching Parameter #
   ratio_test = config.getfloat('ratio', 'ratio_test')

   # Matching Threshold #
   threshold = config.getint('match', 'threshold')

   # Resizing Parameter #
   resize_size = config.getint('resize', 'size')
   return sift_parameters, ratio_test, threshold, resize_size

if __name__=="__main__": 

   # 1
   # Query Image 1 장과 Source Image 1 장을 불러와 매칭 해보기 #
   # Query Image, Source Image, Save Path, Config Path 읽기 #
   query_image_path = input("Query Image Path: ")
   source_image_path = input("Source Image Path: ")
   save_path = input("Save Path: ")
   config_path = input("Config Path: ")

   # 2
   # Config.ini 파일의 파라미터를 변경해가며 최적의 파라미터 찾기 #
   # SIFT Parameter를 Config 파일로 받아오기 #
   sift_parameters, ratio_test, threshold, resize_size = select_config(config_path)    

   # Save Folder에 Query Image 저장하기 #
   current_time = time.localtime()
   folder_name = time.strftime("%Y-%m-%d_%H-%M-%S", current_time)
   if query_image_path and source_image_path:
      if not os.path.exists(save_path):
         os.makedirs(save_path)
      save_folder = f"{save_path}/{folder_name}"
      if not os.path.exists(save_folder):
         os.makedirs(save_folder)
      file_data = np.fromfile(query_image_path, np.uint8)
      # Query Image 읽어오기 #
      query_image = cv2.imdecode(file_data, cv2.IMREAD_COLOR)
      with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
         temp_file_path = tmp_file.name
         cv2.imwrite(temp_file_path, query_image)
      shutil.move(temp_file_path, os.path.join(save_folder, 'query_image.jpg'))

      # Source Image 불러오기 #
      source_images = load_images_from_folder(source_image_path)

      # Ranking and Visualization #
      ranked_images, max_matches = rank_images_and_visualize(query_image, source_images, save_folder, sift_parameters, ratio_test, threshold, resize_size)


      # Save Folder에 매칭을 통해 파악한 유사 이미지 저장하기 #
      for rank, (filename, match_count) in enumerate(ranked_images):
         if os.path.isdir(source_image_path):
            file_data = np.fromfile(os.path.join(source_image_path, filename), np.uint8)
            img = cv2.imdecode(file_data, cv2.IMREAD_COLOR)
         else:
            file_data = np.fromfile(source_image_path, np.uint8)
            img = cv2.imdecode(file_data, cv2.IMREAD_COLOR)
         with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            temp_file_path = tmp_file.name
            cv2.imwrite(temp_file_path, img)
         shutil.move(temp_file_path, os.path.join(save_folder,f"rank{rank+1:05d}_{filename}"))
         text = f"Rank {rank+1}: {filename}, Score: {(100*match_count/max_matches):.2f} ({match_count} matches))\n"
         print(text)

