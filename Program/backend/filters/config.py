from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class FilterConfig:
    """Streaming filter thresholds and memory limits."""

    ## 블러처리 임계치
    blur_threshold: float = 80.0
    ## 과노출 임계치
    white_thr: int = 245
    black_thr: int = 10
    white_ratio_thr: float = 0.28
    black_ratio_thr: float = 0.30
    
    ## 중복제거 임계치
    #- 사진이 얼마나 비슷하면 중복으로 볼 것인지?
    #- 값 크게 : 중복 제거 강함
    #- 값 작게 : 중복 제거 약함
    hash_dist_thr: int = 0      ## **
    
    #- 사진이 애매한 경우 2차 검증(NCC)용
    #- refine_band : 애매한 구간 폭
    #- refine_ncc_thr : 정밀검사 기준점
    refine_band: int = 0
    refine_ncc_thr: float = 0.5          ## **

    #- 같은 객체(트랙)에서 몇 프레임 이내를 중복 검사 할지
    #- 값 크게 : 중복 제거 강함
    #- 값 작게 : 중복 제거 약함
    frame_gap_thr: int = 10      ## **
    
    #- 같은 트랙 중복 기록 유지 시간
    track_ttl_frames: int = 3000

    #- 전역 중복 기록 유지 범위
    global_ttl_frames: int = 3000
    global_max_entries: int = 20000
    
    #- 전역 중복 후보 탐색 범위/속도
    global_bucket_prefix_bits: int = 16
    compare_neighbor_buckets: bool = True

    thumb_size: int = 32



## 실제로 임계치 건드는 곳 :
## hash_dist_thr, frame_gap_thr, refine_ncc_thr