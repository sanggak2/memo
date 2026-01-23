# memo
오픈 메모장

```
import time
import argparse
import os
import cv2
import numpy as np
import psutil
import csv
from hailo_platform import (HEF, VDevice, HailoStreamInterface, ConfigureParams, InputVStreamParams, OutputVStreamParams, FormatType)

# -------------------------------------------------
# 1. Hailo Wrapper Class (수정됨)
# -------------------------------------------------
class HailoInference:
    def __init__(self, hef_path):
        self.hef_path = hef_path
        self.target = VDevice()
        self.hef = HEF(hef_path)

        # 네트워크 그룹 설정
        self.configure_params = ConfigureParams.create_from_hef(hef=self.hef, interface=HailoStreamInterface.PCIe)
        self.network_groups = self.target.configure(self.hef, self.configure_params)
        self.network_group = self.network_groups[0]

        # 입력/출력 스트림 매개변수 설정
        self.input_vstream_params = InputVStreamParams.make(self.network_group, format_type=FormatType.FLOAT32)
        self.output_vstream_params = OutputVStreamParams.make(self.network_group, format_type=FormatType.FLOAT32)

        # 입력 정보 가져오기
        self.input_vstream_infos = self.hef.get_input_vstream_infos()
        self.input_shape = self.input_vstream_infos[0].shape
        self.input_name = self.input_vstream_infos[0].name
        print(f"[Init] Model Input Shape: {self.input_shape} / Name: {self.input_name}")

    def get_input_shape(self):
        return self.input_shape

    def run(self, video_path, log_path):
        # -------------------------------------------------
        # 핵심 수정: activate() 안에서 모든 작업을 수행
        # -------------------------------------------------
        with self.network_group.activate(self.network_group_params): 
            # 파이프라인 생성
            with self.network_group.create_infer_pipeline(self.input_vstream_params, self.output_vstream_params) as pipeline:
                
                # 비디오 로드
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    raise RuntimeError(f"Could not open video: {video_path}")

                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                print(f"[INFO] Video Info: {width}x{height}, {total_frames} frames")

                # 워밍업 (더미 데이터로 10회) - 시동 예열
                print("[SYSTEM] Warming up (10 frames)...")
                dummy_data = {self.input_name: np.zeros(self.input_shape, dtype=np.float32)}
                for _ in range(10):
                    pipeline.infer(dummy_data)
                print("[SYSTEM] Warmup Done. Starting Benchmark...")

                frame_count = 0
                
                # CSV 파일 준비
                with open(log_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Frame', 'Pre_ms', 'Infer_ms', 'Post_ms', 'Total_ms', 'FPS', 'CPU_%', 'Mem_%'])

                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break

                        frame_count += 1
                        
                        # 1. Pre-processing (Resize & Normalize)
                        t0 = time.time()
                        resized = cv2.resize(frame, (self.input_shape[1], self.input_shape[0]))
                        input_data = resized.astype(np.float32) / 255.0
                        input_data = np.expand_dims(input_data, axis=0) # (1, H, W, C)
                        input_dict = {self.input_name: input_data}
                        t1 = time.time()

                        # 2. Inference (NPU)
                        # activate 상태이므로 여기서 infer 호출 가능
                        output = pipeline.infer(input_dict)
                        t2 = time.time()

                        # 3. Post-processing (Dummy for benchmark)
                        # 실제 후처리는 복잡하므로 시간 측정용 더미 연산만 수행
                        _ = output 
                        t3 = time.time()

                        # 시간 계산
                        pre_time = (t1 - t0) * 1000
                        infer_time = (t2 - t1) * 1000
                        post_time = (t3 - t2) * 1000
                        total_time = (t3 - t0) * 1000
                        fps = 1000.0 / total_time if total_time > 0 else 0

                        # 시스템 상태
                        cpu_usage = psutil.cpu_percent()
                        mem_usage = psutil.virtual_memory().percent

                        # 로그 기록 및 출력
                        writer.writerow([frame_count, f"{pre_time:.2f}", f"{infer_time:.2f}", f"{post_time:.2f}", f"{total_time:.2f}", f"{fps:.2f}", cpu_usage, mem_usage])
                        
                        if frame_count % 10 == 0:
                            print(f"Frame {frame_count}/{total_frames} | FPS: {fps:.2f} | Infer: {infer_time:.2f}ms")

                cap.release()
                print(f"[INFO] Benchmark Finished. Log saved to {log_path}")

    # activate 파라미터가 필요없는 버전일 경우를 대비해 속성 추가
    @property
    def network_group_params(self):
        from hailo_platform import ConfigureParams
        return self.configure_params

# -------------------------------------------------
# Main Execution
# -------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='yolop.hef', help='Path to .hef model')
    parser.add_argument('--video', default='NonDureong.mp4', help='Path to input video')
    args = parser.parse_args()

    # 결과 저장 폴더 생성
    os.makedirs("logs", exist_ok=True)
    log_filename = f"logs/{time.strftime('%Y%m%d_%H%M%S')}_RPi5_Hailo_yolop.csv"
    
    print(f"[INFO] Target Video: {args.video}")
    print(f"[INFO] Target Model: {args.model}")

    try:
        hailo_app = HailoInference(args.model)
        hailo_app.run(args.video, log_filename)
    except Exception as e:
        print(f"[ERROR] {e}")
```

Error
```
[INFO] Target Video: NonDureong.mp4
[INFO] Target Model: yolop.hef
[Init] Loading HEF: yolop.hef
[Init] Model Input Shape: (640, 640, 3) / Name: yolop/input_layer1
[INFO] Log file created: logs/20260123_051004_RPi5_Hailo_yolop.csv
[SYSTEM] Warming up (10 frames)...[HailoRT] [error] CHECK failed - Trying to write to vstream yolop/input_layer1 before its network group is activated
[HailoRT] [error] CHECK_SUCCESS failed with status=HAILO_NETWORK_GROUP_NOT_ACTIVATED(69)
[HailoRT] [error] Failed waiting for threads with status HAILO_NETWORK_GROUP_NOT_ACTIVATED(69)
[HailoRT] [error] Failed waiting for threads with status HAILO_NETWORK_GROUP_NOT_ACTIVATED(69)
[HailoRT] [error] Failed waiting for threads with status HAILO_NETWORK_GROUP_NOT_ACTIVATED(69)
[HailoRT] [error] Failed waiting for threads with status HAILO_NETWORK_GROUP_NOT_ACTIVATED(69)
Traceback (most recent call last):
  File "/usr/local/lib/python3.10/dist-packages/hailo_platform/pyhailort/pyhailort.py", line 974, in infer
    self._infer_pipeline.infer(input_data, output_buffers, batch_size)
hailo_platform.pyhailort._pyhailort.HailoRTStatusException: 69

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/workspace/benchmark.py", line 189, in <module>
    run_experiment(args.model, args.video)
  File "/workspace/benchmark.py", line 126, in run_experiment
    hailo_wrapper.infer(dummy)
  File "/workspace/benchmark.py", line 101, in infer
    return self.pipeline.infer(input_dict)
  File "/usr/local/lib/python3.10/dist-packages/hailo_platform/pyhailort/pyhailort.py", line 972, in infer
    with ExceptionWrapper():
  File "/usr/local/lib/python3.10/dist-packages/hailo_platform/pyhailort/pyhailort.py", line 122, in __exit__
    self._raise_indicative_status_exception(value)
  File "/usr/local/lib/python3.10/dist-packages/hailo_platform/pyhailort/pyhailort.py", line 172, in _raise_indicative_status_exception
    raise self.create_exception_from_status(error_code) from libhailort_exception
hailo_platform.pyhailort.pyhailort.HailoRTNetworkGroupNotActivatedException: Network group is not activated
Segmentation fault (core dumped)


```
