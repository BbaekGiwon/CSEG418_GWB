import numpy as np
import cv2
from scipy.ndimage import convolve

# =======================
# 엣지 검출 주요 변수 설정
# =======================
LOW_THRESHOLD = 30      # 약한 엣지 임계값
HIGH_THRESHOLD = 100    # 강한 엣지 임계값
KERNEL_SIZE = 3         # 가우시안 커널 크기
SIGMA = 1.0             # 가우시안 커널 표준편차

# =======================
# 함수 정의
# =======================
# 1. Gaussian Filter (흐림)


def gaussian_kernel(size: int, sigma: float):
    # """가우시안 필터 커널을 생성합니다."""
    x, y = np.meshgrid(np.arange(size), np.arange(size))
    center = (size - 1) / 2
    kernel = np.exp(-((x-center)**2 + (y-center)**2) /
                    (2 * sigma**2))  # 가우시안 필터 생성 -> 가우시안 분포 적용
    return kernel / np.sum(kernel)  # 정규화


def gaussian_blur(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    # """이미지에 가우시안 블러를 적용합니다."""
    return convolve(image.astype(float), kernel.astype(float))

# 2. Gradient Calculation (엣지 강도와 방향)


def gradient_magnitude_direction(image: np.ndarray):
    # """이미지에서 그라디언트의 강도와 방향을 계산합니다."""
    # x축 sobel filter
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    # y축 sobel filter
    sobel_y = np.array([[-1, -2, -1], [0,  0,  0], [1,  2,  1]])

    grad_x = convolve(image.astype(float), sobel_x)
    grad_y = convolve(image.astype(float), sobel_y)

    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    direction = (np.arctan2(grad_y, grad_x) * 180 /
                 np.pi) % 360  # 방향을 0~360도로 변환

    return magnitude, direction

# 3. Non-Maximum Suppression (비최대 억제)


def non_maximum_suppression(magnitude: np.ndarray, direction: np.ndarray) -> np.ndarray:
    # """비최대 억제를 적용하여 엣지를 얇게 만듭니다."""
    height, width = magnitude.shape
    output = np.zeros_like(magnitude)

    for y in range(1, height-1):
        for x in range(1, width-1):
            angle = direction[y, x]
            neighbors = []

            # 방향에 따라 비교할 픽셀 선택
            if (angle < 22.5) or (angle >= 157.5):
                neighbors = [magnitude[y, x-1], magnitude[y, x+1]]  # 수평
            elif (22.5 <= angle < 67.5):
                neighbors = [magnitude[y-1, x+1],
                             magnitude[y+1, x-1]]  # 오른쪽 대각선
            elif (67.5 <= angle < 112.5):
                neighbors = [magnitude[y-1, x], magnitude[y+1, x]]  # 수직
            elif (112.5 <= angle < 157.5):
                neighbors = [magnitude[y-1, x-1],
                             magnitude[y+1, x+1]]  # 왼쪽 대각선

            # 중앙 픽셀이 양 옆보다 크다면 유지
            if magnitude[y, x] > max(neighbors):  # neighbor보다 중앙이 큰 경우 -> edge
                output[y, x] = magnitude[y, x]
            else:
                output[y, x] = 0

    return output

# 4. Edge Tracking by Hysteresis (이중 임계값으로 경계 추적)


def edge_tracking_by_hysteresis(image: np.ndarray, low_threshold: float, high_threshold: float) -> np.ndarray:
    # """이중 임계값을 이용해 엣지를 추적합니다."""
    strong_edges = image > high_threshold
    weak_edges = (image >= low_threshold) & (image <= high_threshold)

    final_edges = np.copy(strong_edges)

    height, width = image.shape
    for y in range(1, height-1):
        for x in range(1, width-1):
            if weak_edges[y, x]:
                # linking, 주위에 강한 edge가 있으면 final_edges에 포함시킨다
                if np.any(strong_edges[y-1:y+2, x-1:x+2]):
                    final_edges[y, x] = True
                else:
                    final_edges[y, x] = False

    return final_edges.astype(np.uint8) * 255  # 0과 255로 변환

# Canny Edge Detector


def canny_edge_detection(image: np.ndarray, low_threshold: float, high_threshold: float, kernel_size: int, sigma: float):

    # 1. Gaussian Blur
    kernel = gaussian_kernel(kernel_size, sigma)
    blurred_image = gaussian_blur(image, kernel)

    # 2. Gradient 계산
    magnitude, direction = gradient_magnitude_direction(blurred_image)

    # 3. Non-Maximum Suppression
    suppressed_image = non_maximum_suppression(magnitude, direction)

    # 4. Edge Tracking by Hysteresis
    final_edges = edge_tracking_by_hysteresis(
        suppressed_image, low_threshold, high_threshold)

    return final_edges


# =======================
# 메인 실행 코드
# =======================
if __name__ == "__main__":
    img = cv2.imread("lena.jpg", cv2.IMREAD_GRAYSCALE)  # 이미지 경로 수정
    if img is None:
        print("이미지 파일을 찾을 수 없습니다.")
    else:
        edges = canny_edge_detection(
            img, LOW_THRESHOLD, HIGH_THRESHOLD, KERNEL_SIZE, SIGMA)
        cv2.imwrite("canny_edges.jpg", edges)  # 결과를 파일로 저장
        cv2.imshow("Canny Edges", edges)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
