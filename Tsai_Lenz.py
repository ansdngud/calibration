'''
#!/usr/bin/env python3
"""
Tsai-Lenz 알고리즘을 사용한 Eye-to-Hand 캘리브레이션 유틸리티

올바른 Tsai-Lenz 알고리즘을 사용하여 카메라에서 로봇 베이스로의 변환 행렬을 계산합니다.
"""

import json
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, Optional, List, Dict
from datetime import datetime


def load_calibration_data(data_dir: str) -> List[Dict]:
    """
    캘리브레이션 데이터 로드
    
    Args:
        data_dir: 캘리브레이션 데이터가 저장된 디렉토리 경로
        
    Returns:
        프레임 데이터 리스트. 각 프레임은 다음 키를 포함:
        - timestamp: 타임스탬프
        - T_target2cam: 타겟에서 카메라로의 변환 행렬 (4x4)
        - T_base2ee: 로봇 베이스에서 end-effector로의 변환 행렬 (4x4)
    """
    data_dir = Path(data_dir)
    frames = []
    
    # ChArUco 파일 찾기
    charuco_files = sorted(data_dir.glob('*_charuco.json'))
    
    for charuco_file in charuco_files:
        timestamp = charuco_file.stem.replace('_charuco', '')
        
        # Load charuco data
        try:
            with open(charuco_file, 'r') as f:
                charuco_data = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load {charuco_file}: {e}")
            continue
        
        # Load robot pose
        pose_file = data_dir / 'poses' / f"{timestamp}_pose.json"
        if not pose_file.exists():
            print(f"Warning: Pose file not found for {timestamp}")
            continue
        
        try:
            with open(pose_file, 'r') as f:
                pose_data = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load {pose_file}: {e}")
            continue
        
        # Parse charuco data
        rvec_target2cam = np.array(charuco_data['rvec_target2cam']).reshape(3, 1)
        tvec_target2cam = np.array(charuco_data['tvec_target2cam']).reshape(3, 1)
        
        R_target2cam = cv2.Rodrigues(rvec_target2cam)[0]
        T_target2cam = np.eye(4)
        T_target2cam[:3, :3] = R_target2cam
        T_target2cam[:3, 3:4] = tvec_target2cam
        
        # Parse robot pose data
        R_base2ee = np.array(pose_data['R_base2ee'])
        t_base2ee = np.array(pose_data['t_base2ee']).reshape(3, 1)
        T_base2ee = np.eye(4)
        T_base2ee[:3, :3] = R_base2ee
        T_base2ee[:3, 3:4] = t_base2ee
        
        frame = {
            'timestamp': timestamp,
            'T_target2cam': T_target2cam,
            'T_base2ee': T_base2ee,
            'rvec_target2cam': rvec_target2cam,
            'tvec_target2cam': tvec_target2cam
        }
        
        frames.append(frame)
    
    print(f"Loaded {len(frames)} frames from {data_dir}")
    return frames


def solve_tsai_lenz(frames: List[Dict]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Tsai-Lenz 알고리즘을 사용하여 Cam-to-Base 변환 행렬 계산
    
    표준 Hand-Eye 방정식: A_i @ X = X @ B_i
    여기서 상대 변환을 사용:
    - A_i: 로봇의 상대 변환 (첫 포즈에서 i번째 포즈로)
    - B_i: 카메라의 상대 변환 (첫 포즈에서 i번째 포즈로)
    - X: 카메라 → 로봇 베이스 변환 (고정)
    
    Args:
        frames: load_calibration_data()로 로드한 프레임 데이터 리스트
        
    Returns:
        (R_cam2base, t_cam2base) 튜플:
        - R_cam2base: 카메라에서 베이스로의 회전 행렬 (3x3)
        - t_cam2base: 카메라에서 베이스로의 병진 벡터 (3,)
        실패 시 (None, None) 반환
    """
    if len(frames) < 3:
        print(f"Error: 최소 3개 이상의 프레임이 필요합니다. 현재 {len(frames)}개")
        return None, None
    
    print(f"Tsai-Lenz 알고리즘으로 Eye-to-Hand 캘리브레이션 수행 (총 {len(frames)}개 포즈)")
    
    # 첫 번째 포즈를 기준으로 설정
    T_base2ee_0 = frames[0]['T_base2ee']
    T_target2cam_0 = frames[0]['T_target2cam']
    T_cam2target_0 = np.linalg.inv(T_target2cam_0)
    
    # 상대 변환 계산을 위한 배열
    A_rotations = []  # 상대 회전 (로봇)
    A_translations = []  # 상대 병진 (로봇)
    B_rotations = []  # 상대 회전 (카메라)
    B_translations = []  # 상대 병진 (카메라)
    
    # 첫 번째 포즈는 항등 변환
    A_rotations.append(np.eye(3))
    A_translations.append(np.zeros(3))
    B_rotations.append(np.eye(3))
    B_translations.append(np.zeros(3))
    
    # 나머지 포즈들에 대해 상대 변환 계산
    for i in range(1, len(frames)):
        T_base2ee_i = frames[i]['T_base2ee']
        T_target2cam_i = frames[i]['T_target2cam']
        T_cam2target_i = np.linalg.inv(T_target2cam_i)
        
        # 상대 변환 계산: 기준 포즈에서 현재 포즈로의 변환
        A = np.linalg.inv(T_base2ee_0) @ T_base2ee_i  # 로봇의 상대 변환
        B = np.linalg.inv(T_cam2target_0) @ T_cam2target_i  # 카메라의 상대 변환
        
        A_rotations.append(A[:3, :3])
        A_translations.append(A[:3, 3])
        B_rotations.append(B[:3, :3])
        B_translations.append(B[:3, 3])
    
    try:
        # OpenCV의 Tsai-Lenz 알고리즘 사용
        R_cam2base, t_cam2base = cv2.calibrateHandEye(
            A_rotations, A_translations,
            B_rotations, B_translations,
            method=cv2.CALIB_HAND_EYE_TSAI
        )
        
        print(f"✅ Tsai-Lenz 알고리즘 완료")
        print(f"R_cam2base:\n{R_cam2base}")
        print(f"t_cam2base: {t_cam2base.flatten()}")
        
        return R_cam2base, t_cam2base.flatten()
        
    except Exception as e:
        print(f"❌ Tsai-Lenz 알고리즘 실패: {e}")
        return None, None


def save_calibration_result(
    R_cam2base: np.ndarray,
    t_cam2base: np.ndarray,
    output_path: str,
    num_poses: int = None,
    data_dir: str = None
) -> bool:
    """
    캘리브레이션 결과를 JSON 파일로 저장
    
    Args:
        R_cam2base: 카메라에서 베이스로의 회전 행렬 (3x3)
        t_cam2base: 카메라에서 베이스로의 병진 벡터 (3,)
        output_path: 출력 파일 경로 (상대 경로 또는 절대 경로)
        num_poses: 사용된 포즈 개수 (선택적)
        data_dir: 데이터 디렉토리 (output_path가 상대 경로일 경우 사용)
        
    Returns:
        저장 성공 여부
    """
    try:
        # 전체 변환 행렬 구성
        T_cam2base = np.eye(4)
        T_cam2base[:3, :3] = R_cam2base
        T_cam2base[:3, 3] = t_cam2base
        
        # 결과 데이터 구성
        result = {
            "R_cam2base": R_cam2base.tolist(),
            "t_cam2base": t_cam2base.tolist(),
            "T_cam2base": T_cam2base.tolist(),
            "method": "Tsai-Lenz (cv2.calibrateHandEye with CALIB_HAND_EYE_TSAI)",
            "timestamp": datetime.now().isoformat(),
            "description": "Eye-to-Hand 캘리브레이션 결과 (카메라 → 로봇 베이스 변환)"
        }
        
        if num_poses is not None:
            result["num_poses"] = num_poses
        
        # 출력 경로 결정
        if Path(output_path).is_absolute():
            output_file = Path(output_path)
        else:
            if data_dir is None:
                output_file = Path(output_path)
            else:
                output_file = Path(data_dir) / output_path
        
        # 디렉토리 생성
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 파일 저장
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"✅ 캘리브레이션 결과 저장 완료: {output_file}")
        return True
        
    except Exception as e:
        print(f"❌ 결과 저장 실패: {e}")
        return False


def calibrate_cam2base(
    data_dir: str,
    output_path: str = "cam2base_tl.json"
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    데이터를 로드하고 Tsai-Lenz 알고리즘으로 Cam-to-Base 변환 행렬을 계산하여 저장
    
    Args:
        data_dir: 캘리브레이션 데이터 디렉토리
        output_path: 출력 JSON 파일 경로 (기본값: "cam2base_tl.json")
        
    Returns:
        (R_cam2base, t_cam2base) 튜플, 실패 시 (None, None)
    """
    print("="*80)
    print("Tsai-Lenz Eye-to-Hand 캘리브레이션")
    print("="*80)
    
    # 데이터 로드
    frames = load_calibration_data(data_dir)
    
    if len(frames) < 3:
        print(f"❌ 충분한 데이터가 없습니다. 최소 3개 이상의 프레임이 필요합니다.")
        return None, None
    
    # Tsai-Lenz 알고리즘 실행
    R_cam2base, t_cam2base = solve_tsai_lenz(frames)
    
    if R_cam2base is None or t_cam2base is None:
        print("❌ 캘리브레이션 실패")
        return None, None
    
    # 결과 저장
    save_calibration_result(
        R_cam2base, t_cam2base, output_path,
        num_poses=len(frames), data_dir=data_dir
    )
    
    return R_cam2base, t_cam2base


def main():
    """메인 실행 함수 (예제)"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python Tsai_Lenz.py <data_dir> [output_path]")
        print("Example: python Tsai_Lenz.py /home/ros/llm_robot/data/Calibration/Eye-to-Hand2")
        sys.exit(1)
    
    data_dir = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "cam2base_tl.json"
    
    R_cam2base, t_cam2base = calibrate_cam2base(data_dir, output_path)
    
    if R_cam2base is not None:
        print("\n" + "="*80)
        print("캘리브레이션 완료!")
        print("="*80)
    else:
        print("\n" + "="*80)
        print("캘리브레이션 실패!")
        print("="*80)
        sys.exit(1)


if __name__ == "__main__":
    main()
'''


#!/usr/bin/env python3
"""
Tsai-Lenz 알고리즘을 사용한 Eye-to-Hand 캘리브레이션 유틸리티
"""

import json
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, Optional, List, Dict
from datetime import datetime


def load_calibration_data(data_dir: str) -> List[Dict]:
    data_dir = Path(data_dir)
    frames = []

    charuco_files = sorted(data_dir.glob('*_charuco.json'))

    for charuco_file in charuco_files:
        timestamp = charuco_file.stem.replace('_charuco', '')

        try:
            with open(charuco_file, 'r') as f:
                charuco_data = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load {charuco_file}: {e}")
            continue

        pose_file = data_dir / 'poses' / f"{timestamp}_pose.json"
        if not pose_file.exists():
            print(f"Warning: Pose file not found for {timestamp}")
            continue

        try:
            with open(pose_file, 'r') as f:
                pose_data = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load {pose_file}: {e}")
            continue

        # ✅ 수정: cam2target 원본값에서 직접 역변환
        r_ct = np.array(charuco_data['rvec_cam2target']).reshape(3)
        t_ct = np.array(charuco_data['tvec_cam2target']).reshape(3)
        R_ct, _ = cv2.Rodrigues(r_ct)
        R_tc = R_ct.T
        t_tc = (-R_ct.T @ t_ct).reshape(3, 1)

        T_target2cam = np.eye(4)
        T_target2cam[:3, :3] = R_tc
        T_target2cam[:3, 3:4] = t_tc

        R_base2ee = np.array(pose_data['R_base2ee'])
        t_base2ee = np.array(pose_data['t_base2ee']).reshape(3, 1)
        T_base2ee = np.eye(4)
        T_base2ee[:3, :3] = R_base2ee
        T_base2ee[:3, 3:4] = t_base2ee

        frame = {
            'timestamp': timestamp,
            'T_target2cam': T_target2cam,
            'T_base2ee': T_base2ee,
        }
        frames.append(frame)

    print(f"Loaded {len(frames)} frames from {data_dir}")
    return frames


def solve_tsai_lenz(frames: List[Dict]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if len(frames) < 3:
        print(f"Error: 최소 3개 이상의 프레임이 필요합니다. 현재 {len(frames)}개")
        return None, None

    print(f"Tsai-Lenz 알고리즘으로 Eye-to-Hand 캘리브레이션 수행 (총 {len(frames)}개 포즈)")

    T_base2ee_0    = frames[0]['T_base2ee']
    T_target2cam_0 = frames[0]['T_target2cam']
    T_cam2target_0 = np.linalg.inv(T_target2cam_0)

    A_rotations    = [np.eye(3)]
    A_translations = [np.zeros(3)]
    B_rotations    = [np.eye(3)]
    B_translations = [np.zeros(3)]

    for i in range(1, len(frames)):
        T_base2ee_i    = frames[i]['T_base2ee']
        T_target2cam_i = frames[i]['T_target2cam']
        T_cam2target_i = np.linalg.inv(T_target2cam_i)

        A = np.linalg.inv(T_base2ee_0) @ T_base2ee_i
        B = np.linalg.inv(T_cam2target_0) @ T_cam2target_i

        A_rotations.append(A[:3, :3])
        A_translations.append(A[:3, 3])
        B_rotations.append(B[:3, :3])
        B_translations.append(B[:3, 3])

    try:
        R_cam2base, t_cam2base = cv2.calibrateHandEye(
            A_rotations, A_translations,
            B_rotations, B_translations,
            method=cv2.CALIB_HAND_EYE_TSAI
        )
        print(f"✅ Tsai-Lenz 완료")
        print(f"R_cam2base:\n{R_cam2base}")
        print(f"t_cam2base: {t_cam2base.flatten()}")
        print(f"t_cam2base (mm): {t_cam2base.flatten() * 1000}")
        return R_cam2base, t_cam2base.flatten()

    except Exception as e:
        print(f"❌ Tsai-Lenz 실패: {e}")
        return None, None


def save_calibration_result(R_cam2base, t_cam2base, output_path, num_poses=None, data_dir=None):
    try:
        T_cam2base = np.eye(4)
        T_cam2base[:3, :3] = R_cam2base
        T_cam2base[:3, 3]  = t_cam2base

        result = {
            "R_cam2base": R_cam2base.tolist(),
            "t_cam2base": t_cam2base.tolist(),
            "T_cam2base": T_cam2base.tolist(),
            "method": "Tsai-Lenz (cv2.calibrateHandEye with CALIB_HAND_EYE_TSAI)",
            "timestamp": datetime.now().isoformat(),
            "description": "Eye-to-Hand 캘리브레이션 결과 (카메라 → 로봇 베이스 변환)"
        }
        if num_poses is not None:
            result["num_poses"] = num_poses

        if Path(output_path).is_absolute():
            output_file = Path(output_path)
        else:
            output_file = Path(data_dir) / output_path if data_dir else Path(output_path)

        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)

        print(f"✅ 결과 저장 완료: {output_file}")
        return True

    except Exception as e:
        print(f"❌ 결과 저장 실패: {e}")
        return False


def calibrate_cam2base(data_dir, output_path="cam2base_tl.json"):
    print("="*80)
    print("Tsai-Lenz Eye-to-Hand 캘리브레이션")
    print("="*80)

    frames = load_calibration_data(data_dir)
    if len(frames) < 3:
        print("❌ 데이터 부족")
        return None, None

    R_cam2base, t_cam2base = solve_tsai_lenz(frames)
    if R_cam2base is None:
        print("❌ 캘리브레이션 실패")
        return None, None

    save_calibration_result(R_cam2base, t_cam2base, output_path,
                            num_poses=len(frames), data_dir=data_dir)
    return R_cam2base, t_cam2base


def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage: python Tsai_Lenz.py <data_dir> [output_path]")
        sys.exit(1)

    data_dir    = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "cam2base_tl.json"

    R_cam2base, t_cam2base = calibrate_cam2base(data_dir, output_path)

    if R_cam2base is not None:
        print("\n" + "="*80)
        print("캘리브레이션 완료!")
        print("="*80)
    else:
        print("\n캘리브레이션 실패!")
        sys.exit(1)


if __name__ == "__main__":
    main()