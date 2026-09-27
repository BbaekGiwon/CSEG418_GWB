# 기존 제출본과의 차이

최상위 `computer_vision_ver`를 기준으로 HW3를 `hw3/`에 통합했습니다. 이 폴더에는 기존 `hw3/computer_vision_ver/computer_vision_ver`에서 내용이 달랐던 설정과 이미지 두 장만 보관합니다.

| 설정 | 통합 기본값 (`hw3/confg.ini`) | 기존 제출본 (`confg.ini`) |
| --- | --- | --- |
| nfeatures | 4000 | 0 |
| contrastThreshold | 0.16 | 0.05 |
| resize size | 5000 | 200 |

`database/004_2024-MO07520.jpg`와 `database/009_2024-MO06413.jpg`는 기본 데이터베이스의 동명 이미지와 바이트 내용이 다릅니다. 어느 쪽이 더 적절한지는 판정하지 않았으며, 기본 데이터베이스와 분리해 원본을 보존했습니다. 제출본 데이터로 실험하려면 `hw3/database`를 별도 작업 폴더에 복사한 뒤 이 두 파일로 교체하세요. 이 폴더의 두 장만으로 전체 데이터베이스 평가를 실행하지 마세요.

Python 코드는 하나로 통합했습니다. `computer_vision.py`의 실행 구문은 두 버전이 같았습니다. `computer_vision_nbyn.py`는 기준 버전의 UTF-8 설정 파일 읽기와 이미지 경로 출력이 추가된 형태를 유지했습니다. 기존 제출 코드는 Git 커밋 `4ab5216`의 `hw3/computer_vision_ver/computer_vision_ver/`에서 확인할 수 있습니다.

동일했던 이미지 42장과 환경 설정, 정답 인덱스는 중복을 제거했습니다. 기준 버전의 그래프, 발표 자료, 보고서 및 로컬 점수 파일은 `hw3/`에 보존했습니다.
