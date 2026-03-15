# memo
오픈 메모장

동기식
```
import time
import argparse
import os
import cv2
import numpy as np
import psutil
import csv
import threading
import gc
import sys
from datetime import datetime

from hailo_platform import (HEF, VDevice, HailoStreamInterface, ConfigureParams, 
                            InputVStreamParams, OutputVStreamParams, FormatType, InferVStreams)

# -------------------------------------------------
# 1. System Monitor (Threaded)
# -------------------------------------------------
class SystemMonitor(threading.Thread):
    def __init__(self, interval=0.5):
        super().__init__()
        self.interval = interval
        self.running = True
        self.daemon = True
        self.stats = {"cpu": 0.0, "temp": 0.0}

    def run(self):
        while self.running:
            try:
                self.stats["cpu"] = psutil.cpu_percent(interval=None)
                try:
                    with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                        self.stats["temp"] = int(f.read().strip()) / 1000.0
                except:
                    self.stats["temp"] = 0.0
                time.sleep(self.interval)
            except:
                break

    def stop(self):
        self.running = False

# -------------------------------------------------
# 2. Benchmark Function
# -------------------------------------------------
def run_final_benchmark_rpi(model_path, video_path):
    # [최적화] 프로세스 우선순위 격상
    try:
        pid = os.getpid()
        param = os.sched_param(99)
        os.sched_setscheduler(pid, os.SCHED_FIFO, param)
        print("[INFO] Process Priority: REAL-TIME (FIFO 99)")
    except Exception as e:
        print(f"[WARN] 우선순위 설정 실패: {e}")

    # Hailo Init
    print("[INIT] Loading Hailo HEF...")
    target = VDevice()
    hef = HEF(model_path)

    configure_params = ConfigureParams.create_from_hef(hef=hef, interface=HailoStreamInterface.PCIe)
    network_groups = target.configure(hef, configure_params)
    network_group = network_groups[0]

    input_vstream_params = InputVStreamParams.make(network_group, format_type=FormatType.UINT8)
    output_vstream_params = OutputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)

    input_vstream_infos = hef.get_input_vstream_infos()
    input_shape = input_vstream_infos[0].shape
    input_name = input_vstream_infos[0].name
    in_h, in_w = input_shape[0], input_shape[1]
    
    print(f"[INIT] Model Input Shape: {in_h}x{in_w}")

    cap = cv2.VideoCapture(os.path.abspath(video_path))
    if not cap.isOpened(): raise RuntimeError("Video Open Failed")

    monitor = SystemMonitor(interval=0.5)
    monitor.start()

    # 로그 버퍼 (Infer_Lat와 E2E_Lat 두 개 저장)
    MAX_FRAMES = 5000
    # (Frame_ID, Timestamp, FPS, Infer_Lat, E2E_Lat, CPU, Temp)
    log_buffer = [None] * MAX_FRAMES 
    frame_id = 0

    try:
        with network_group.activate():
            with InferVStreams(network_group, input_vstream_params, output_vstream_params) as pipeline:
                
                print(f"[INFO] 🚀 FINAL RPi+Hailo Benchmark Started (Dual Latency Mode)")

                # 버퍼 비우기 & 워밍업
                for _ in range(30): cap.read()
                dummy_data = {input_name: np.zeros((1, in_h, in_w, 3), dtype=np.uint8)}
                for _ in range(50): pipeline.infer(dummy_data)
                
                gc.disable()
                print("[INFO] Measurement Started!")
                
                start_time_global = time.time()

                while True:
                    ret, frame = cap.read()
                    if not ret: break

                    # [A] E2E 시작 시간 측정
                    t_now = time.time()
                    t_e2e_start = time.perf_counter()

                    # 1. CPU Preprocessing
                    resized_frame = cv2.resize(frame, (in_w, in_h))
                    input_data = np.expand_dims(resized_frame, axis=0)
                    infer_request = {input_name: input_data}

                    # [B] Inference 시작 시간 측정
                    t_infer_start = time.perf_counter()
                    
                    # 2. Inference (PCIe Send -> NPU -> PCIe Recv)
                    output = pipeline.infer(infer_request)
                    
                    # [C] Inference 종료 시간 측정
                    t_infer_end = time.perf_counter()

                    # 3. CPU Postprocessing (Minimal)
                    _ = list(output.values())[0]

                    # [D] E2E 종료 시간 측정
                    t_e2e_end = time.perf_counter()

                    # 계산
                    infer_latency = (t_infer_end - t_infer_start) * 1000.0
                    e2e_latency = (t_e2e_end - t_e2e_start) * 1000.0
                    fps = 1000.0 / e2e_latency if e2e_latency > 0 else 0

                    log_buffer[frame_id] = (
                        frame_id,
                        t_now,
                        fps,
                        infer_latency, # 순수 추론+전송
                        e2e_latency,   # 시스템 전체
                        monitor.stats["cpu"],
                        monitor.stats["temp"]
                    )

                    if frame_id % 200 == 0:
                         print(f"[{frame_id}] FPS: {fps:.1f} | Infer: {infer_latency:.2f}ms | E2E: {e2e_latency:.2f}ms")

                    frame_id += 1
                    if frame_id >= MAX_FRAMES: break

    except KeyboardInterrupt: pass
    except Exception as e: print(f"[ERROR] {e}")
    finally:
        gc.enable()
        monitor.stop()
        total_dur = time.time() - start_time_global
        cap.release()

        print(f"\n[INFO] Saving data...")
        os.makedirs("logs", exist_ok=True)
        log_path = f"logs/bench_rpi_dual_{datetime.now().strftime('%m%d_%H%M')}.csv"

        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            # 헤더 수정: Latency가 두 개로 나뉨
            writer.writerow(["Frame_ID", "Unix_Time", "Timestamp", "System_FPS", "Infer_Latency_ms", "E2E_Latency_ms", "CPU_Usage_Percent", "Temp_C"])

            for i in range(frame_id):
                fid, ts, fps, inf_lat, e2e_lat, cpu, tmp = log_buffer[i]
                ts_str = datetime.fromtimestamp(ts).strftime("%H:%M:%S.%f")
                writer.writerow([fid, f"{ts:.6f}", ts_str, f"{fps:.2f}", f"{inf_lat:.2f}", f"{e2e_lat:.2f}", cpu, tmp])

        print(f"[RESULT] Avg System FPS: {frame_id / total_dur:.2f}")
        print(f"[RESULT] Saved to {log_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='yolop_raw.hef', help='Path to .hef model') 
    parser.add_argument('--video', default='NonDureong.mp4', help='Input video')
    args = parser.parse_args()

    if os.path.exists(args.model) and os.path.exists(args.video):
        run_final_benchmark_rpi(args.model, args.video)
    else:
        print("[ERROR] File not found.")
```

비동기식
```
import time
import argparse
import os
import cv2
import numpy as np
import psutil
import csv
import threading
import queue
from datetime import datetime
from hailo_platform import VDevice, FormatType, HailoSchedulingAlgorithm

# -------------------------------------------------
# 1. System Monitor (기존 유지)
# -------------------------------------------------
class SystemMonitor(threading.Thread):
    def __init__(self, interval=0.5):
        super().__init__()
        self.interval = interval
        self.running = True
        self.daemon = True
        self.stats = {"cpu": 0.0, "temp": 0.0}

    def run(self):
        while self.running:
            try:
                self.stats["cpu"] = psutil.cpu_percent(interval=None)
                with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                    self.stats["temp"] = int(f.read().strip()) / 1000.0
                time.sleep(self.interval)
            except: break
    def stop(self): self.running = False

# -------------------------------------------------
# 2. Benchmark Class (HailoRT 4.23.0 전용)
# -------------------------------------------------
class HailoAsyncBenchmark:
    def __init__(self, model_path):
        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        
        self.target = VDevice(params)
        self.infer_model = self.target.create_infer_model(model_path)
        
        # 입력/출력 포맷 명시적 설정
        self.infer_model.inputs[0].set_format_type(FormatType.UINT8)
        self.input_shape = self.infer_model.inputs[0].shape
        
        if len(self.input_shape) == 4:
            self.in_h, self.in_w = self.input_shape[1], self.input_shape[2]
        else:
            self.in_h, self.in_w = self.input_shape[0], self.input_shape[1]
            
        self.input_name = self.infer_model.inputs[0].name

        self.output_info = []
        for output in self.infer_model.outputs:
            output.set_format_type(FormatType.FLOAT32)
            self.output_info.append({
                "name": output.name,
                "shape": output.shape
            })
            
        print(f"[INIT] Model has 1 input and {len(self.infer_model.outputs)} outputs.")

        self.log_queue = queue.Queue()
        self.stop_event = threading.Event()

    def run_benchmark(self, video_path, max_frames=2000):
        cap = cv2.VideoCapture(os.path.abspath(video_path))
        monitor = SystemMonitor()
        monitor.start()
        results = []
        
        print(f"[INFO] Starting 4.23.0 Async Benchmark...")
        
        with self.infer_model.configure() as configured_infer_model:
            start_time_total = time.time()

            ongoing_bindings = []
            for i in range(max_frames):
                ret, frame = cap.read()
                if not ret: break

                # 1. 전처리
                t_start = time.perf_counter()
                resized = cv2.resize(frame, (self.in_w, self.in_h))
                input_data = np.expand_dims(resized, axis=0)

                # 2. 비동기 추론 요청 (Bindings 사용)
                bindings = configured_infer_model.create_bindings()
                bindings.input(self.input_name).set_buffer(input_data)

                output_buffers = {}
                for out_info in self.output_info:
                    # NPU가 값을 채워넣을 빈 FLOAT32 배열 할당
                    empty_buffer = np.empty(out_info["shape"], dtype=np.float32)
                    bindings.output(out_info["name"]).set_buffer(empty_buffer)
                    output_buffers[out_info["name"]] = empty_buffer

                def get_callback(f_id, t_st, current_binding):
                    def cb(completion_info):
                        if completion_info.exception:
                            print(f"[ERROR] Inference failed: {completion_info.exception}")
                        else:
                            t_end = time.perf_counter()
                            e2e_lat = (t_end - t_st) * 1000.0
                            self.log_queue.put((f_id, e2e_lat))
                        
                        if current_binding in ongoing_bindings:
                            ongoing_bindings.remove(current_binding)
                    return cb

                # GC 방지 리스트에 추가
                ongoing_bindings.append(bindings)

                # 비동기 실행 (Non-blocking)
                configured_infer_model.run_async([bindings], get_callback(i, t_start, bindings))
                # 로그 큐에서 데이터 가끔 비워주기 (메모리 관리)
                while not self.log_queue.empty():
                    results.append(self.log_queue.get())

            # 모든 작업 완료 대기
            self.target.wait_for_idle() 
            total_duration = time.time() - start_time_total

        # 종료 및 로깅
        monitor.stop()
        cap.release()

        while not self.log_queue.empty():
            results.append(self.log_queue.get())

        self._save_logs(results, total_duration, monitor.stats)

    def _save_logs(self, results, duration, stats):
        avg_fps = len(results) / duration
        print(f"\n[RESULT] Avg Throughput: {avg_fps:.2f} FPS")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='yolop_raw.hef')
    parser.add_argument('--video', default='NonDureong.mp4')
    args = parser.parse_args()

    bench = HailoAsyncBenchmark(args.model)
    bench.run_benchmark(args.video)
```

Error
```
[INIT] Model has 1 input and 5 outputs.
[INFO] Starting 4.23.0 Async Benchmark...
[HailoRT] [error] CHECK failed - Trying to get buffer as view for 'model/conv57', while it is not configured as view
Traceback (most recent call last):
  File "/usr/local/lib/python3.10/dist-packages/hailo_platform/pyhailort/pyhailort.py", line 3351, in run_async
    cpp_job = self._configured_infer_model.run_async(
hailo_platform.pyhailort._pyhailort.HailoRTStatusException: 6

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/workspace/async-benchmark.py", line 138, in <module>
    bench.run_benchmark(args.video)
  File "/workspace/async-benchmark.py", line 109, in run_benchmark
    configured_infer_model.run_async([bindings], get_callback(i, t_start, bindings))
  File "/usr/local/lib/python3.10/dist-packages/hailo_platform/pyhailort/pyhailort.py", line 3350, in run_async
    with ExceptionWrapper():
  File "/usr/local/lib/python3.10/dist-packages/hailo_platform/pyhailort/pyhailort.py", line 122, in __exit__
    self._raise_indicative_status_exception(value)
  File "/usr/local/lib/python3.10/dist-packages/hailo_platform/pyhailort/pyhailort.py", line 172, in _raise_indicative_status_exception
    raise self.create_exception_from_status(error_code) from libhailort_exception
hailo_platform.pyhailort.pyhailort.HailoRTInvalidOperationException: Invalid operation. See hailort.log for more information

```
