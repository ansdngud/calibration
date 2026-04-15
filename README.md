ChArUco 보드 기반 Eye-to-Hand 캘리브레이션 도구.
외부 고정 카메라(RealSense)와 FR5 로봇 간의 좌표 변환 행렬(T_cam2base)을 계산한다.
End-Effector에 ChArUco 보드를 고정한 뒤, 로봇을 다양한 포즈로 이동시키며 카메라 이미지와 로봇 TCP 포즈를 동시에 수집한다.
cali.py는 데이터 수집 + 자체 캘리브레이션 (메인 스크립트)
Tsai-Lenz.py는 OpenCV의 Tsai-Lenz 알고리즘을 사용하여 Eye-to-Hand 환경에서 카메라와 로봇 베이스 간의 변환 행렬을 계산하는 유틸리티
pgo2.py는 tsai-lenz.py에서 도출된 초기 캘리브레이션 결과를 바탕으로 Pose Graph Optimization (PGO) 기법을 적용하여 최종 변환 행렬을 최적화하며, 데이터 수집 과정에서 발생할 수 있는 노이즈나 오차(Outlier)를 MAD 필터링으로 제거하여 매우 정밀한 캘리브레이션 결과를 얻을 수 있다.

eyeinhand.py는 카메라가 로봇의 엔드이펙터(EE)에 부착되어 있고, 캘리브레이션 타겟(ChArUco 보드)이 외부 작업 공간에 고정된 eye-in-hand 환경을 위한 자동 캘리브레이션 툴.
