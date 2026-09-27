# Computer Vision Assignments

2024년 2학기 컴퓨터비전 과제 모음입니다. 영상 회전, Canny 에지 검출, SIFT 기반 이미지 매칭을 구현했습니다.

| 과제 | 내용 | 주요 코드 |
| --- | --- | --- |
| HW1 | Forward / backward mapping과 보간을 이용한 영상 회전 | [main.cpp](hw1/Image_rotation_exercise/Image_Rotation/main.cpp) |
| HW2 | Gaussian blur → Sobel gradient → non-maximum suppression → hysteresis | [Python 코드](hw2/canny%20edge/canny%20edge/project2_20201586_baek_giwon.py) |
| HW3 | SIFT 특징점, ratio test, RANSAC 기반 검색·정합 및 ROC/DET 평가 | [매칭](hw3/computer_vision.py), [평가](hw3/computer_vision_nbyn.py) |

## Python 환경

저장소 루트에서 실행합니다. `requirements.txt`는 기존 `sps.yaml`의 주요 라이브러리 버전을 사용하며 Python 3.10 환경을 기준으로 설치하도록 구성했습니다. 전체 실행 검증이 완료된 환경은 아닙니다.

```powershell
py -3.10 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

기존 `hw3/sps.yaml`은 당시 Windows / Python 3.8 환경 기록입니다.

## HW1: Image Rotation

`hw1/Image_rotation_exercise/Image_Rotation.sln`을 Visual Studio에서 엽니다. 프로젝트의 도구 집합은 `v143`이고, OpenCV 설정은 **2.4.13 / x86 / vc12**용입니다. 해당 라이브러리와 현재 컴파일러의 호환성 확인 또는 프로젝트 설정 변경이 필요합니다.

OpenCV SDK는 Git 추적에서 제외되어 있습니다. 기존 `.props`를 사용할 경우 `hw1/Image_rotation_exercise/OPENCV` 아래에 `include`와 `x86/vc12/{lib,bin}`을 배치하고 DLL 경로를 설정합니다. 다른 OpenCV 버전을 쓸 경우 include/lib 경로와 링크 라이브러리 이름도 변경해야 합니다.

실행 작업 디렉터리는 `hw1/Image_rotation_exercise/Image_Rotation`으로 설정합니다. 코드가 이 디렉터리의 `lena.jpg`를 읽으며, 결과 창에서 키를 눌러 다음 단계로 진행합니다.

## HW2: Canny Edge Detection

저장소 루트에서 다음과 같이 실행합니다.

```powershell
cd "hw2/canny edge/canny edge"
../../../.venv/Scripts/python project2_20201586_baek_giwon.py
```

같은 폴더의 `lena.jpg`를 읽고 `canny_edges.jpg`를 저장하며 결과 창을 표시합니다. 임계값과 Gaussian 커널 설정은 코드 상단에서 변경합니다. 과제 제출 당시 구현을 보존하고 있으며 OpenCV의 Canny와 동일한 출력을 보장하지 않습니다. 특히 방향을 0–360도로 계산하지만 억제 구간은 0–180도 기준이고, hysteresis는 강한 에지에 바로 인접한 약한 에지만 연결합니다.

## HW3: SIFT Image Matching

저장소 루트에서 실행합니다.

```powershell
.venv/Scripts/python hw3/computer_vision.py
```

입력 예시는 아래와 같습니다. 프롬프트에는 경로를 따옴표 없이 입력합니다.

```text
Query Image Path: hw3/database/001_2024-MO03125.jpg
Source Image Path: hw3/database
Save Path: outputs/matching
Config Path: hw3/confg.ini
```

설정 파일 이름은 기존 코드의 `confg.ini`를 유지했습니다. 결과는 시간별 하위 폴더에 저장됩니다. 예시는 질의 이미지 자신도 검색 대상으로 포함하므로, 실제 검색 평가에서는 질의 이미지를 대상 폴더에서 분리해야 합니다.

### N × N 비교 및 ROC / DET

```powershell
.venv/Scripts/python hw3/computer_vision_nbyn.py
```

이미지 폴더, 설정 파일, 출력 폴더, 정답 인덱스 `.npy` 경로를 차례로 입력합니다. 정답 인덱스는 저장소에 포함되어 있지 않으므로 별도로 준비해야 합니다. 정렬된 이미지 목록의 각 이미지마다 동일 대상으로 간주하는 다른 이미지의 인덱스 목록을 담아야 합니다. 원본 과제 폴더의 `indices.npy`를 사용할 경우 이미지 목록과 순서가 일치하는지 먼저 확인합니다.

기존 실험 그래프는 [graphs](hw3/graphs)에 보관되어 있습니다. 특징점이 없는 이미지나 homography 추정 실패 등 일부 입력은 원래 코드에서 처리하지 않으므로, 모든 입력에 대한 실행을 보장하지 않습니다.

## 저장소 구성과 보존 범위

- `hw3/`: HW3 실행 안내의 기준 버전과 실험 그래프.
- `hw3/hw3/hw3/`: 별도 제출 버전. 기준 버전과 코드 차이가 있어 보존했습니다.
- `hw1/`, `hw2/`: 과제 코드, 입력 이미지 및 기존 문서. HW2의 중첩 제출 폴더도 보존했습니다.
- 빌드 결과, OpenCV SDK, 개인 IDE 설정, Python 캐시, 생성된 점수 및 출력 폴더는 Git 추적 대상에서 제외합니다.

원래 과제의 알고리즘을 보존한 학습용 저장소입니다. 별도의 오픈소스 라이선스는 아직 지정하지 않았으며, 포함된 과제 문서와 이미지에 일괄적인 재배포 라이선스를 부여하지 않습니다.
