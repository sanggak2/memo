# memo
오픈 메모장

```
import cv2
import time
import argparse
import sys
import os
import numpy as np
import csv
from datetime import datetime
import psutil

# =========================================================================
# [설정] 사용자 파일 경로 지정 (여기만 수정하면 됩니다)
# =========================================================================
DEFAULT_VIDEO_PATH = "/workspace/NonDureong.mp4"
DEFAULT_HEF_PATH   = "/workspace/yolop.hef"
# =========================================================================

# Hailo 라이브러리 임포트
try:
    from hailo_platform import (HEF, VDevice, HailoStreamInterface, InferVStreams, 
                                ConfigureParams, InputVStreamParams, OutputVStreamParams, FormatType)
except ImportError:
    print("[CRITICAL] 'hailo-platform' 라이브러리가 없습니다. (pip install hailo_platform)")
    sys.exit(1)

def get_system_temp():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            return int(f.read().strip()) / 1000.0
    except:
        return -1

class StandaloneLogger:
    def __init__(self, model_path, video_path):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        
        if not os.path.exists("logs"):
            os.makedirs("logs")

        self.filename = f"logs/{timestamp}_RPi5_Hailo_{model_name}.csv"
        self.file = open(self.filename, 'w', newline='')
        self.writer = csv.writer(self.file)

        # 논문용 헤더 (엑셀에서 열어보기 좋게)
        self.writer.writerow([
            'Frame_ID', 'Timestamp', 'FPS', 'E2E_Latency_ms', 'Inference_Time_ms',
            'CPU_Usage_%', 'CPU_Freq_MHz', 'Memory_Usage_%', 'Temperature_C', 'Model'
        ])
        print(f"[INFO] Log file created: {self.filename}")

    def log(self, frame_id, fps, e2e_latency, inf_time, cpu_freq, temp, model):
        now_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        cpu_usage = psutil.cpu_percent(interval=None)
        mem_usage = psutil.virtual_memory().percent
        
        self.writer.writerow([
            frame_id, now_time, f"{fps:.2f}", f"{e2e_latency:.2f}", f"{inf_time:.2f}",
            cpu_usage, f"{cpu_freq:.1f}", mem_usage, temp, model
        ])
        self.file.flush()

    def close(self):
        self.file.close()

class HailoWrapper:
    def __init__(self, hef_path):
        if not os.path.exists(hef_path):
            raise FileNotFoundError(f"HEF file not found: {hef_path}")
            
        print(f"[Init] Loading HEF: {hef_path}")
        self.hef = HEF(hef_path)
        self.target = VDevice()
        
        # PCIe 인터페이스 설정
        self.configure_params = ConfigureParams.create_from_hef(
            self.hef, interface=HailoStreamInterface.PCIe)
        self.network_groups = self.target.configure(self.hef, self.configure_params)
        self.network_group = self.network_groups[0]
        
        self.input_vparams = InputVStreamParams.make_from_network_group(
            self.network_group, quantized=False, format_type=FormatType.FLOAT32)
        self.output_vparams = OutputVStreamParams.make_from_network_group(
            self.network_group, quantized=False, format_type=FormatType.FLOAT32)
        
        # 입력 레이어 정보 자동 추출
        input_vstream_info = self.hef.get_input_vstream_infos()[0]
        self.input_name = input_vstream_info.name
        self.input_shape = input_vstream_info.shape 
        self.height = self.input_shape[0]
        self.width = self.input_shape[1]
        
        print(f"[Init] Model Input Shape: {self.input_shape} / Name: {self.input_name}")

        self.pipeline = InferVStreams(self.network_group, self.input_vparams, self.output_vparams)
        self.pipeline.__enter__()

    def infer(self, input_array):
        # Dictionary 형태로 변환 (Hailo API 요구사항)
        input_dict = {self.input_name: input_array}
        return self.pipeline.infer(input_dict)

    def close(self):
        self.pipeline.__exit__(None, None, None)

def run_experiment(model_path, video_path):
    if not os.path.exists(video_path):
        print(f"[CRITICAL] Video file not found: {video_path}")
        sys.exit(1)

    print(f"[INFO] Target Video: {video_path}")
    print(f"[INFO] Target Model: {model_path}")

    try:
        hailo_wrapper = HailoWrapper(model_path)
    except Exception as e:
        print(f"[ERROR] Failed to init Hailo: {e}")
        sys.exit(1)
        
    cap = cv2.VideoCapture(video_path)
    logger = StandaloneLogger(model_path, video_path)

    print("[SYSTEM] Warming up (10 frames)...", end="", flush=True)
    for _ in range(10):
        dummy = np.zeros((1, hailo_wrapper.height, hailo_wrapper.width, 3), dtype=np.float32)
        hailo_wrapper.infer(dummy)
    print(" [Ready]")

    print(f"[INFO] Starting Benchmark Loop...")
    frame_id = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret: 
                print("[INFO] End of video file.")
                break

            # 1. E2E 시작
            t_start_e2e = time.perf_counter()

            # 2. 전처리 (Resize & Normalization)
            resized = cv2.resize(frame, (hailo_wrapper.width, hailo_wrapper.height))
            input_data = np.expand_dims(resized, axis=0).astype(np.float32)

            # 3. 추론 (NPU)
            t_start_inf = time.perf_counter()
            _ = hailo_wrapper.infer(input_data)
            t_end_inf = time.perf_counter()
            
            # 4. E2E 종료
            t_end_e2e = time.perf_counter()
            
            # 지표 계산
            inf_time = (t_end_inf - t_start_inf) * 1000
            e2e_latency = (t_end_e2e - t_start_e2e) * 1000
            fps = 1000.0 / e2e_latency if e2e_latency > 0 else 0
            
            temp = get_system_temp()
            try:
                cpu_freq = psutil.cpu_freq().current
            except:
                cpu_freq = 0

            # 로그 기록
            logger.log(frame_id, fps, e2e_latency, inf_time, cpu_freq, temp, model_path)

            if frame_id % 30 == 0:
                print(f"[{frame_id}] FPS:{fps:.1f} | E2E:{e2e_latency:.1f}ms (Infer:{inf_time:.1f}ms) | Temp:{temp}C", flush=True)
            frame_id += 1

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    except Exception as e:
        print(f"\n[ERROR] Runtime error: {e}")
    finally:
        cap.release()
        hailo_wrapper.close()
        logger.close()
        print(f"[INFO] Experiment Done. Logs saved in 'logs' folder.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 기본값을 사용자가 지정한 파일로 고정
    parser.add_argument('--model', type=str, default=DEFAULT_HEF_PATH, help='Path to .hef file')
    parser.add_argument('--video', type=str, default=DEFAULT_VIDEO_PATH, help='Path to video file')
    args = parser.parse_args()
    
    run_experiment(args.model, args.video)
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
