# memo
오픈 메모장



선행설정
CPU GOVERNOR -> PERFORMANCE
```
# CPU 0~3의 거버너를 performance로 일괄 변경
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# 확인하는 명령어 (performance라고 4줄 뜨면 성공)
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```
시간동기화
```
t
```
Frame_ID,Unix_Time,Timestamp,System_FPS,Infer_Latency_ms,E2E_Latency_ms,CPU_Usage_Percent,Temp_C

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

ANCHORS = [
    [[3, 9], [5, 11], [4, 20]],       # Stride 8
    [[7, 18], [6, 39], [12, 31]],     # Stride 16
    [[19, 50], [38, 81], [68, 157]]   # Stride 32
]
STRIDES = [8, 16, 32]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def decode_detections(det_outputs, conf_thres=0.4):
    boxes, scores, class_ids = [], [], []
    for i, pred in enumerate(det_outputs):
        stride = STRIDES[i]
        anchor = np.array(ANCHORS[i])
        bs, h, w, ch = pred.shape
        pred = pred.reshape(bs, h, w, 3, 6)
        pred = sigmoid(pred)
        
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        grid = np.stack((grid_x, grid_y), axis=2).reshape(1, h, w, 1, 2)

        pred_xy = (pred[..., 0:2] * 2.0 - 0.5 + grid) * stride
        anchor_broadcast = anchor.reshape(1, 1, 1, 3, 2)
        pred_wh = (pred[..., 2:4] * 2.0) ** 2 * anchor_broadcast
        
        pred_conf = pred[..., 4]
        pred_cls = pred[..., 5]
        final_score = pred_conf * pred_cls
        
        mask = final_score > conf_thres
        if not np.any(mask): continue
            
        valid_xy = pred_xy[mask]
        valid_wh = pred_wh[mask]
        valid_scores = final_score[mask]
        
        x1y1 = valid_xy - valid_wh / 2
        valid_boxes = np.concatenate([x1y1, valid_wh], axis=1)
        
        boxes.extend(valid_boxes.tolist())
        scores.extend(valid_scores.tolist())
        class_ids.extend([0] * len(valid_scores))
    return boxes, scores, class_ids

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
    with VDevice() as target:
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
        MAX_FRAMES = 10000
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

                        # 원본 이미지의 크기를 가져와 가로/세로 비율 계산
                        h0, w0 = frame.shape[:2]
                        rx = w0 / in_w
                        ry = h0 / in_h

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
                        try:
                            det_8 = output['model/conv57']
                            det_16 = output['model/conv65']
                            det_32 = output['model/conv72']
                            da_seg = output['model/ne_activation_activation1']
                            ll_seg = output['model/ne_activation_activation2']
                        except KeyError:
                            vals = list(output.values())
                            det_outs = [v for v in vals if len(v.shape) == 4 and v.shape[3] == 18]
                            seg_outs = [v for v in vals if len(v.shape) == 4 and v.shape[3] == 2]
                            det_outs.sort(key=lambda x: x.shape[1], reverse=True) 
                            det_8, det_16, det_32 = det_outs[0], det_outs[1], det_outs[2]
                            da_seg, ll_seg = seg_outs[0], seg_outs[1]

                        # 3-1. Drivable & Lane Line Mask 처리
                        da_diff = da_seg[0][..., 1] - da_seg[0][..., 0]
                        da_mask = np.zeros_like(da_diff, dtype=np.uint8)
                        da_mask[da_diff > 0.0] = 1 
                        ll_mask = np.argmax(ll_seg[0], axis=-1).astype(np.uint8)

                        # 3-2. Object Detection 디코딩 및 NMS
                        boxes, scores, class_ids = decode_detections([det_8, det_16, det_32], conf_thres=0.4)
                        final_boxes = []
                        for b in boxes:
                            x1, y1, w, h = b
                            # [수정된 부분] letterbox 공식(dw, dh, r) 대신 단순 비율(rx, ry)을 곱해 좌표 복원
                            x1 = x1 * rx
                            y1 = y1 * ry
                            w = w * rx
                            h = h * ry
                            final_boxes.append([int(x1), int(y1), int(w), int(h)])
                        
                        _ = cv2.dnn.NMSBoxes(final_boxes, scores, score_threshold=0.4, nms_threshold=0.45)
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
    parser.add_argument('--video', default='Video.mp4', help='Input video')
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
cv2.setNumThreads(1) # OpenCV가 스레드를 과도하게 생성하여 GIL과 충돌하는 것 방지
import numpy as np
import psutil
import csv
import threading
import queue
from datetime import datetime
from hailo_platform import VDevice, FormatType, HailoSchedulingAlgorithm

# -------------------------------------------------
# 0. YOLOP Post-Processing Constants & Utils
# -------------------------------------------------
ANCHORS = [
    [[3, 9], [5, 11], [4, 20]],       # Stride 8
    [[7, 18], [6, 39], [12, 31]],     # Stride 16
    [[19, 50], [38, 81], [68, 157]]   # Stride 32
]
STRIDES = [8, 16, 32]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def decode_detections(det_outputs, conf_thres=0.4):
    boxes, scores, class_ids = [], [], []
    for i, pred in enumerate(det_outputs):
        stride = STRIDES[i]
        anchor = np.array(ANCHORS[i])
        bs, h, w, ch = pred.shape
        pred = pred.reshape(bs, h, w, 3, 6)
        pred = sigmoid(pred)
        
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        grid = np.stack((grid_x, grid_y), axis=2).reshape(1, h, w, 1, 2)

        pred_xy = (pred[..., 0:2] * 2.0 - 0.5 + grid) * stride
        anchor_broadcast = anchor.reshape(1, 1, 1, 3, 2)
        pred_wh = (pred[..., 2:4] * 2.0) ** 2 * anchor_broadcast
        
        pred_conf = pred[..., 4]
        pred_cls = pred[..., 5]
        final_score = pred_conf * pred_cls
        
        mask = final_score > conf_thres
        if not np.any(mask): continue
            
        valid_xy = pred_xy[mask]
        valid_wh = pred_wh[mask]
        valid_scores = final_score[mask]
        
        x1y1 = valid_xy - valid_wh / 2
        valid_boxes = np.concatenate([x1y1, valid_wh], axis=1)
        
        boxes.extend(valid_boxes.tolist())
        scores.extend(valid_scores.tolist())
        class_ids.extend([0] * len(valid_scores))
    return boxes, scores, class_ids

# -------------------------------------------------
# 1. System Monitor
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
            except: 
                print("/sys/class/thermal/thermal_zone0/temp MATTER")
                break
    def stop(self): self.running = False

# -------------------------------------------------
# 2. Async Benchmark (Paper-Grade)
# -------------------------------------------------
class HailoAsyncBenchmark:
    def __init__(self, model_path):
        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        
        self.target = VDevice(params)
        self.infer_model = self.target.create_infer_model(model_path)
        
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
            self.output_info.append({"name": output.name, "shape": output.shape})
            
        print(f"[INIT] Model Inputs: 1, Outputs: {len(self.infer_model.outputs)}")

        self.POOL_SIZE = 15 # 논문 멘트: "We empirically selected POOL_SIZE=15 for optimal throughput"
        self.NUM_WORKERS = 3  

        self.buffer_pool = queue.Queue(maxsize=self.POOL_SIZE)
        self.postprocess_queue = queue.Queue(maxsize=self.POOL_SIZE)
        self.log_queue = queue.Queue()
        self.monitor = SystemMonitor()

    def postprocess_worker(self):
        """CPU에서 BBox 디코딩 및 NMS를 수행하는 워커 스레드"""
        while True:
            item = self.postprocess_queue.get()
            if item is None: # 종료 신호
                self.postprocess_queue.task_done()
                break
                
            f_id, t_e2e_st, t_inf_st, t_inf_end, rx, ry, bindings, input_buf, output_bufs = item

            try:
                # --- 1. 텐서 파싱 ---
                try:
                    det_8 = output_bufs['model/conv57']
                    det_16 = output_bufs['model/conv65']
                    det_32 = output_bufs['model/conv72']
                    da_seg = output_bufs['model/ne_activation_activation1']
                    ll_seg = output_bufs['model/ne_activation_activation2']
                except KeyError:
                    try:
                        det_8 = next(v for k, v in output_bufs.items() if 'conv57' in k)
                        det_16 = next(v for k, v in output_bufs.items() if 'conv65' in k)
                        det_32 = next(v for k, v in output_bufs.items() if 'conv72' in k)
                        da_seg = next(v for k, v in output_bufs.items() if 'activation1' in k)
                        ll_seg = next(v for k, v in output_bufs.items() if 'activation2' in k)
                    except StopIteration:
                        print(f"\n[FATAL ERROR] 출력 텐서를 찾을 수 없습니다! 실제 딕셔너리 구조:")
                        for k, v in output_bufs.items():
                            print(f"  - Key: '{k}' | Shape: {v.shape}")
                        os._exit(1)
                
                if len(det_8.shape) == 3:
                    det_8 = np.expand_dims(det_8, axis=0)
                    det_16 = np.expand_dims(det_16, axis=0)
                    det_32 = np.expand_dims(det_32, axis=0)
                if len(da_seg.shape) == 3:
                    da_seg = np.expand_dims(da_seg, axis=0)
                    ll_seg = np.expand_dims(ll_seg, axis=0)

                # --- 2. NCHW -> NHWC 방어 코드 ---
                if len(det_8.shape) == 4 and det_8.shape[1] == 18: 
                    det_8 = np.transpose(det_8, (0, 2, 3, 1))
                    det_16 = np.transpose(det_16, (0, 2, 3, 1))
                    det_32 = np.transpose(det_32, (0, 2, 3, 1))
                if len(da_seg.shape) == 4 and da_seg.shape[1] == 2:
                    da_seg = np.transpose(da_seg, (0, 2, 3, 1))
                    ll_seg = np.transpose(ll_seg, (0, 2, 3, 1))

                # --- 3. Drivable & Lane Line Mask (증발했던 로직 복구!) ---
                da_diff = da_seg[0][..., 1] - da_seg[0][..., 0]
                da_mask = np.zeros_like(da_diff, dtype=np.uint8)
                da_mask[da_diff > 0.0] = 1 
                ll_mask = np.argmax(ll_seg[0], axis=-1).astype(np.uint8)

                # --- 4. BBox 디코딩 및 NMS (증발했던 로직 복구!) ---
                boxes, scores, class_ids = decode_detections([det_8, det_16, det_32], conf_thres=0.4)
                final_boxes = []
                for b in boxes:
                    x1, y1, w, h = b
                    final_boxes.append([int(x1 * rx), int(y1 * ry), int(w * rx), int(h * ry)])
                
                _ = cv2.dnn.NMSBoxes(final_boxes, scores, score_threshold=0.4, nms_threshold=0.45)

            except Exception as e:
                print(f"[WARN] Postprocess Error on frame {f_id}: {e}")

            # [최적화 3] E2E Latency 측정 종료 시점
            t_e2e_end = time.perf_counter()
            t_now = time.time()
            
            # 정확하게 분리된 Latency 계산
            infer_lat = (t_inf_end - t_inf_st) * 1000.0  # (PCIe 송신 + NPU + PCIe 수신)
            e2e_lat = (t_e2e_end - t_e2e_st) * 1000.0    # (CPU 전처리 + Infer + CPU 후처리)
            fps_e2e = 1000.0 / e2e_lat if e2e_lat > 0 else 0

            # 로깅 큐에 저장 (스레드가 섞일 수 있으므로 frame_id 필수 포함)
            self.log_queue.put((
                f_id, t_now, fps_e2e, infer_lat, e2e_lat, 
                self.monitor.stats["cpu"], self.monitor.stats["temp"]
            ))

            if f_id % 200 == 0:
                print(f"[{f_id}] Infer: {infer_lat:.2f}ms | E2E: {e2e_lat:.2f}ms")

            # 🌟 [핵심] 처리가 완전히 끝난 버퍼만 반납하여 NPU와 CPU 간 메모리 충돌 차단
            self.buffer_pool.put((bindings, input_buf, output_bufs))
            self.postprocess_queue.task_done()

    def run_benchmark(self, video_path, max_frames=10000):
        # [최적화] 프로세스 우선순위 격상 (OS 스케줄러 간섭 최소화)
        try:
            os.sched_setscheduler(os.getpid(), os.SCHED_FIFO, os.sched_param(99))
            print("[INFO] Process Priority: REAL-TIME (FIFO 99)")
        except: pass

        cap = cv2.VideoCapture(os.path.abspath(video_path))
        if not cap.isOpened(): raise RuntimeError("Video Open Failed")

        self.monitor.start()
        
        # 다중 후처리 워커 스레드 시작
        workers = []
        for _ in range(self.NUM_WORKERS):
            t = threading.Thread(target=self.postprocess_worker, daemon=True)
            t.start()
            workers.append(t)
        
        print(f"[INFO] Started Async Benchmark (Workers: {self.NUM_WORKERS}, Pool: {self.POOL_SIZE})")
        
        with self.infer_model.configure() as configured_infer_model:
            start_time_global = time.time()

            # 버퍼 풀 사전 할당
            for _ in range(self.POOL_SIZE):
                bindings = configured_infer_model.create_bindings()
                input_buf = np.empty(self.input_shape, dtype=np.uint8)
                bindings.input(self.input_name).set_buffer(input_buf)
                output_bufs = {}
                for out_info in self.output_info:
                    out_buf = np.empty(out_info["shape"], dtype=np.float32)
                    bindings.output(out_info["name"]).set_buffer(out_buf)
                    output_bufs[out_info["name"]] = out_buf
                self.buffer_pool.put((bindings, input_buf, output_bufs))

            frame_id = 0
            while frame_id < max_frames:
                ret, frame = cap.read()
                if not ret: break

                # ⏱️ [E2E 시작]
                t_e2e_st = time.perf_counter()

                h0, w0 = frame.shape[:2]
                rx, ry = w0 / self.in_w, h0 / self.in_h

                resized = cv2.resize(frame, (self.in_w, self.in_h))
                input_data = np.expand_dims(resized, axis=0)

                # 풀에서 버퍼 획득 (모든 버퍼가 후처리 중이면 대기)
                bindings, input_buf, output_bufs = self.buffer_pool.get()
                np.copyto(input_buf, input_data)

                # ⏱️ [순수 Infer 시작] (PCIe 전송 직전)
                t_inf_st = time.perf_counter()

                # 콜백 함수 (가장 가벼운 형태로 유지)
                def get_callback(f_id, t_e2e, t_inf_s, r_x, r_y, current_binding, curr_in, curr_outs):
                    def cb(completion_info):
                        # ⏱️ [순수 Infer 종료] (가속기 연산 완료 직후)
                        t_inf_end = time.perf_counter()
                        
                        if completion_info.exception:
                            print(f"[ERROR] Inference failed: {completion_info.exception}")
                            self.buffer_pool.put((current_binding, curr_in, curr_outs))
                        else:
                            # 후처리 큐로 모든 시간값과 데이터 일괄 이관
                            self.postprocess_queue.put((f_id, t_e2e, t_inf_s, t_inf_end, r_x, r_y, current_binding, curr_in, curr_outs))
                    return cb

                job = configured_infer_model.run_async(
                    [bindings], 
                    get_callback(frame_id, t_e2e_st, t_inf_st, rx, ry, bindings, input_buf, output_bufs)
                )
                frame_id += 1

            # NPU 파이프라인 정리 대기
            if 'job' in locals(): job.wait(10000)
            
            # 후처리 큐가 다 비워질 때까지 대기 후 스레드 종료 신호
            self.postprocess_queue.join()
            for _ in range(self.NUM_WORKERS):
                self.postprocess_queue.put(None)
            for t in workers:
                t.join()
                
            total_dur = time.time() - start_time_global

        self.monitor.stop()
        cap.release()

        # 다중 스레드 특성상 로그 순서가 뒤섞일 수 있으므로 정렬
        logs = []
        while not self.log_queue.empty():
            logs.append(self.log_queue.get())
        logs.sort(key=lambda x: x[0]) # frame_id 기준으로 정렬

        self._save_logs(logs, frame_id, total_dur)

    def _save_logs(self, logs, total_frames, duration):
        os.makedirs("logs", exist_ok=True)
        log_path = f"logs/bench_rpi_async_paper_{datetime.now().strftime('%m%d_%H%M')}.csv"

        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Frame_ID", "Unix_Time", "Timestamp", "System_FPS", "Infer_Latency_ms", "E2E_Latency_ms", "CPU_Usage_Percent", "Temp_C"])

            for log in logs:
                fid, ts, fps, inf_lat, e2e_lat, cpu, tmp = log
                ts_str = datetime.fromtimestamp(ts).strftime("%H:%M:%S.%f")
                writer.writerow([fid, f"{ts:.6f}", ts_str, f"{fps:.2f}", f"{inf_lat:.2f}", f"{e2e_lat:.2f}", cpu, tmp])

        global_e2e_fps = total_frames / duration if duration > 0 else 0
        avg_hdh_lat = np.mean([log[3] for log in logs]) if logs else 0
        
        print(f"\n[RESULT] Saved to {log_path}")
        print(f"[RESULT] Avg HDH Latency (NPU Round-trip): {avg_hdh_lat:.2f} ms")
        print(f"[RESULT] Global System Throughput (E2E)  : {global_e2e_fps:.2f} FPS")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='yolop_raw.hef')
    parser.add_argument('--video', default='Video.mp4')
    args = parser.parse_args()

    if os.path.exists(args.model) and os.path.exists(args.video):
        bench = HailoAsyncBenchmark(args.model)
        bench.run_benchmark(args.video, max_frames=10000)
    else:
        print("[ERROR] Model or Video file not found.")
```

```
import time
import argparse
import os
import cv2
cv2.setNumThreads(1)
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

            POOL_SIZE = 10  # 동시에 유지할 최대 비동기 파이프라인 개수
            buffer_pool = queue.Queue(maxsize=POOL_SIZE)

            for _ in range(POOL_SIZE):
                bindings = configured_infer_model.create_bindings()
                input_buf = np.empty(self.input_shape, dtype=np.uint8)
                bindings.input(self.input_name).set_buffer(input_buf)
                output_bufs = {}
                for out_info in self.output_info:
                    out_buf = np.empty(out_info["shape"], dtype=np.float32)
                    bindings.output(out_info["name"]).set_buffer(out_buf)
                    output_bufs[out_info["name"]] = out_buf

                buffer_pool.put((bindings, input_buf, output_bufs))

            print(f"[INFO] Allocated {POOL_SIZE} buffer sets. Measurement Started!")

            for i in range(max_frames):
                ret, frame = cap.read()
                if not ret: break

                # 1. 전처리
                t_start = time.perf_counter()
                resized = cv2.resize(frame, (self.in_w, self.in_h))
                input_data = np.expand_dims(resized, axis=0)

                bindings, input_buf, output_bufs = buffer_pool.get()
                np.copyto(input_buf, input_data)

                def get_callback(f_id, t_nw, t_e2e_st, t_inf_st, current_binding, curr_in, curr_outs, c_cpu, c_tmp):
                    def cb(completion_info):
                        if completion_info.exception:
                            print(f"[ERROR] Inference failed: {completion_info.exception}")
                        else:
                            t_end = time.perf_counter()
                            infer_latency = (t_end - t_inf_st) * 1000.0
                            e2e_latency = (t_end - t_inf_st) * 1000.0
                            fps = 1000.0 / e2e_latency if e2e_latency > 0 else 0

                            self.log_queue.put((f_id, t_nw, fps, infer_latency, e2e_latency, c_cpu, c_tmp))
                        buffer_pool.put((current_binding, curr_in, curr_outs))
                    return cb

                # 비동기 실행
                job = configured_infer_model.run_async([bindings], get_callback(i, t_start, bindings, input_buf, output_bufs))

                # 로그 큐에서 데이터 비워주기 (메모리 관리)
                while not self.log_queue.empty():
                    results.append(self.log_queue.get())

            # 모든 작업 완료 대기
            for _ in range(POOL_SIZE):
                buffer_pool.get()
            total_duration = time.time() - start_time_total

        # 종료 및 로깅
        monitor.stop()
        cap.release()

        while not self.log_queue.empty():
            results.append(self.log_queue.get())

        self._save_logs(results, total_duration, monitor.stats)

    def _save_logs(self, results, duration):
        # 1. 비동기 콜백은 완료 순서가 뒤바뀔 수 있으므로 Frame_ID(x[0]) 기준으로 오름차순 정렬
        results.sort(key=lambda x: x[0])
        
        # 2. CSV 저장 폴더 및 파일명 지정
        os.makedirs("logs", exist_ok=True)
        log_path = f"logs/bench_rpi_async_{datetime.now().strftime('%m%d_%H%M')}.csv"
        
        # 3. 파일 작성
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Frame_ID", "Unix_Time", "Timestamp", "System_FPS", "Infer_Latency_ms", "E2E_Latency_ms", "CPU_Usage_Percent", "Temp_C"])
            
            for row in results:
                fid, ts, fps, inf_lat, e2e_lat, cpu, tmp = row
                ts_str = datetime.fromtimestamp(ts).strftime("%H:%M:%S.%f")
                writer.writerow([fid, f"{ts:.6f}", ts_str, f"{fps:.2f}", f"{inf_lat:.2f}", f"{e2e_lat:.2f}", cpu, tmp])

        avg_fps = len(results) / duration
        print(f"\n[RESULT] Avg Throughput: {avg_fps:.2f} FPS")
        print(f"[RESULT] Saved to {log_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='yolop_raw.hef')
    parser.add_argument('--video', default='NonDureong.mp4')
    args = parser.parse_args()

    bench = HailoAsyncBenchmark(args.model)
    bench.run_benchmark(args.video)
```
visualize
```
import cv2
import numpy as np
import os
import json
from hailo_platform import (
    HEF, VDevice, HailoStreamInterface, ConfigureParams,
    InputVStreamParams, OutputVStreamParams,
    FormatType, InferVStreams
)

# =====================================================
# 0. 설정 및 상수 (YOLOP 객체 탐지용)
# =====================================================
LOG_DIR = "debug_logs"
os.makedirs(LOG_DIR, exist_ok=True)

# YOLOP Anchor 설정 (Stride 8, 16, 32)
ANCHORS = [
    [[3, 9], [5, 11], [4, 20]],       # Stride 8
    [[7, 18], [6, 39], [12, 31]],     # Stride 16
    [[19, 50], [38, 81], [68, 157]]   # Stride 32
]
STRIDES = [8, 16, 32]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def save_array_stats(name, arr):
    stats = {
        "shape": arr.shape,
        "dtype": str(arr.dtype),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
    }
    with open(f"{LOG_DIR}/{name}_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

# =====================================================
# 1. 전처리 함수 (Letterbox)
# =====================================================
def letterbox(img, new_shape=(384, 640), color=(114,114,114)):
    h, w = img.shape[:2]
    nh, nw = new_shape
    r = min(nh / h, nw / w)
    new_unpad = (int(w * r), int(h * r))
    resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    dw = nw - new_unpad[0]
    dh = nh - new_unpad[1]
    dw //= 2
    dh //= 2
    padded = cv2.copyMakeBorder(
        resized,
        dh, nh - new_unpad[1] - dh,
        dw, nw - new_unpad[0] - dw,
        cv2.BORDER_CONSTANT,
        value=color
    )
    return padded, r, (dw, dh)

# =====================================================
# 2. 객체 탐지 후처리 (Decoding)
# =====================================================
def decode_detections(det_outputs, conf_thres=0.25):
    boxes = []
    scores = []
    class_ids = []

    # Hailo 출력은 보통 (1, H, W, Channels) 형태입니다. (NHWC)
    # det_outputs 순서: [det_8, det_16, det_32]
    
    for i, pred in enumerate(det_outputs):
        stride = STRIDES[i]
        anchor = np.array(ANCHORS[i]) # (3, 2)
        
        # pred shape 예시: (1, 48, 80, 18) -> Stride 8일 때 (384/8=48, 640/8=80)
        bs, h, w, ch = pred.shape
        
        # Reshape: (Batch, H, W, 3, 6) -> Anchor 3개, 정보 6개(x,y,w,h,conf,cls)
        pred = pred.reshape(bs, h, w, 3, 6)
        
        # Sigmoid 적용 (0~1 범위로 변환)
        pred = sigmoid(pred)
        
        # Grid 생성
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h)) # (H, W)
        # (H, W, 1) 형태로 변환 후 Stack -> (H, W, 2)
        grid = np.stack((grid_x, grid_y), axis=2)
        # Broadcasting을 위해 차원 확장: (1, H, W, 1, 2)
        grid = grid.reshape(1, h, w, 1, 2)

        # 좌표 복원
        # xy = (pred[..., 0:2] * 2 - 0.5 + grid) * stride
        # wh = (pred[..., 2:4] * 2) ** 2 * anchor
        
        pred_xy = (pred[..., 0:2] * 2.0 - 0.5 + grid) * stride
        
        # Anchor Broadcasting: (1, 1, 1, 3, 2)
        anchor_broadcast = anchor.reshape(1, 1, 1, 3, 2)
        pred_wh = (pred[..., 2:4] * 2.0) ** 2 * anchor_broadcast
        
        pred_conf = pred[..., 4] # Objectness
        pred_cls = pred[..., 5]  # Class score
        
        # Final Score
        final_score = pred_conf * pred_cls
        
        # Threshold Filtering
        mask = final_score > conf_thres
        
        if not np.any(mask):
            continue
            
        valid_xy = pred_xy[mask]
        valid_wh = pred_wh[mask]
        valid_scores = final_score[mask]
        
        # (cx, cy, w, h) -> (x1, y1, w, h) 좌상단 변환
        x1y1 = valid_xy - valid_wh / 2
        valid_boxes = np.concatenate([x1y1, valid_wh], axis=1)
        
        boxes.extend(valid_boxes.tolist())
        scores.extend(valid_scores.tolist())
        class_ids.extend([0] * len(valid_scores))
        
    return boxes, scores, class_ids

# =====================================================
# 3. Main App
# =====================================================
class YoloPFullApp:
    def __init__(self, hef_path):
        print("[Init] Loading HEF...")
        self.hef = HEF(hef_path)
        self.device = VDevice()

        params = ConfigureParams.create_from_hef(
            hef=self.hef,
            interface=HailoStreamInterface.PCIe
        )
        self.network_group = self.device.configure(self.hef, params)[0]

        self.input_vstream_params = InputVStreamParams.make(
            self.network_group,
            format_type=FormatType.UINT8
        )
        self.output_vstream_params = OutputVStreamParams.make(
            self.network_group,
            format_type=FormatType.FLOAT32 # 후처리를 위해 Float32로 받음
        )

        input_info = self.hef.get_input_vstream_infos()[0]
        self.input_name = input_info.name
        self.in_h, self.in_w, _ = input_info.shape

        print(f"[Init] Input Name: {self.input_name}")
        print(f"[Init] Input Shape: {input_info.shape}")

    def run(self, video_path, output_path="output_result.mp4"):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError("Video Open Failed")

        w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30

        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w0, h0))

        print("[Run] Start Inference...")

        with self.network_group.activate():
            with InferVStreams(self.network_group, self.input_vstream_params, self.output_vstream_params) as pipe:
                frame_idx = 0
                while True:
                    ret, frame = cap.read()
                    if not ret: break
                    frame_idx += 1

                    # 1. Preprocess
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img, r, (dw, dh) = letterbox(rgb, (self.in_h, self.in_w))
                    inp = np.expand_dims(img, axis=0).astype(np.uint8)

                    # 2. Inference
                    outputs = pipe.infer({self.input_name: inp})

                    # 3. Get Outputs (5 Tensors)
                    # HEF 변환 시 지정한 이름대로 가져옵니다.
                    try:
                        det_8 = outputs['model/conv57']
                        det_16 = outputs['model/conv65']
                        det_32 = outputs['model/conv72']
                        da_seg = outputs['model/ne_activation_activation1']
                        ll_seg = outputs['model/ne_activation_activation2']
                    except KeyError:
                        print("[Error] Output keys mismatch. Check HEF export names.")
                        print("Keys found:", outputs.keys())
                        break

                    # =================================================
                    # 4-1. Postprocess: Drivable Area & Lane Line
                    # =================================================
                    da = da_seg[0] # (H, W, 2)
                    ll = ll_seg[0] # (H, W, 2)

                    # Drive Area: Background vs Drivable
                    # 기존 로직: diff = drivable - background
                    da_diff = da[..., 1] - da[..., 0]
                    da_mask = np.zeros_like(da_diff, dtype=np.uint8)
                    
                    # 간단한 임계값 적용 (복잡한 column-wise 대신 속도 최적화)
                    # 필요시 기존 column 로직으로 교체 가능
                    da_mask[da_diff > 0.0] = 1 

                    # Lane Line: Argmax
                    ll_mask = np.argmax(ll, axis=-1).astype(np.uint8)

                    # Crop Padding (Letterbox 복원)
                    if dh > 0:
                        da_mask = da_mask[dh:-dh, :]
                        ll_mask = ll_mask[dh:-dh, :]
                    if dw > 0:
                        da_mask = da_mask[:, dw:-dw]
                        ll_mask = ll_mask[:, dw:-dw]

                    # Resize to Original
                    da_mask = cv2.resize(da_mask, (w0, h0), interpolation=cv2.INTER_NEAREST)
                    ll_mask = cv2.resize(ll_mask, (w0, h0), interpolation=cv2.INTER_NEAREST)

                    # =================================================
                    # 4-2. Postprocess: Object Detection
                    # =================================================
                    boxes, scores, class_ids = decode_detections([det_8, det_16, det_32], conf_thres=0.4)
                    
                    # 스케일 복원 (Original Image 좌표로 변환)
                    # Letterbox 좌표 (img) -> 원본 좌표 (frame)
                    # x_original = (x_pad - dw) / r
                    # y_original = (y_pad - dh) / r
                    
                    final_boxes = []
                    for b in boxes:
                        x1, y1, w, h = b
                        
                        # Padding 제거
                        x1 = (x1 - dw) / r
                        y1 = (y1 - dh) / r
                        w = w / r
                        h = h / r
                        
                        final_boxes.append([int(x1), int(y1), int(w), int(h)])

                    # NMS (Non-Maximum Suppression)
                    indices = cv2.dnn.NMSBoxes(final_boxes, scores, score_threshold=0.4, nms_threshold=0.45)

                    # =================================================
                    # 5. Visualization
                    # =================================================
                    overlay = frame.copy()
                    
                    # 1) 주행 영역 (초록색)
                    overlay[da_mask == 1] = (0, 255, 0)
                    
                    # 2) 차선 영역 (빨간색)
                    overlay[ll_mask == 1] = (0, 0, 255)
                    
                    # 합성
                    result = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
                    
                    # 3) 객체 박스 그리기 (파란색)
                    if len(indices) > 0:
                        for i in indices.flatten():
                            bx, by, bw, bh = final_boxes[i]
                            cv2.rectangle(result, (bx, by), (bx+bw, by+bh), (255, 0, 0), 2)
                            cv2.putText(result, f"{scores[i]:.2f}", (bx, by-5), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                    out.write(result)
                    if frame_idx % 30 == 0:
                        print(f"Processing frame {frame_idx}...", end="\r")

        cap.release()
        out.release()
        print(f"\n[Done] Result saved to {output_path}")

if __name__ == "__main__":
    app = YoloPFullApp("yolop_raw.hef")
    app.run("Video.mp4", "final_output.mp4")
```

Error
```
[WARN] 우선순위 설정 실패: [Errno 1] Operation not permitted
[INIT] Loading Hailo HEF...
[INIT] Model Input Shape: 384x640
[INFO] 🚀 FINAL RPi+Hailo Benchmark Started (Dual Latency Mode)
[HailoRT] [error] CHECK failed - UserBuffQEl2model/conv72 (D2H) failed with status=HAILO_TIMEOUT(4) (timeout=10000ms)
[HailoRT] [error] CHECK failed - UserBuffQEl1model/conv65 (D2H) failed with status=HAILO_TIMEOUT(4) (timeout=10000ms)
[HailoRT] [error] CHECK failed - UserBuffQEl3model/ne_activation_activation1 (D2H) failed with status=HAILO_TIMEOUT(4) (timeout=10000ms)
[HailoRT] [error] CHECK failed - UserBuffQEl3model/conv57 (D2H) failed with status=HAILO_TIMEOUT(4) (timeout=10000ms)
[HailoRT] [error] CHECK failed - UserBuffQEl0model/ne_activation_activation2 (D2H) failed with status=HAILO_TIMEOUT(4) (timeout=10000ms)
[HailoRT] [error] Failed waiting for threads with status HAILO_TIMEOUT(4)
[HailoRT] [error] Failed waiting for threads with status HAILO_TIMEOUT(4)
[HailoRT] [error] Failed waiting for threads with status HAILO_TIMEOUT(4)
[HailoRT] [error] Failed waiting for threads with status HAILO_TIMEOUT(4)
[HailoRT] [error] Failed waiting for threads with status HAILO_TIMEOUT(4)
[HailoRT] [error] Ioctl HAILO_FW_CONTROL failed with 19. Read dmesg log for more info
[HailoRT] [error] CHECK_SUCCESS failed with status=HAILO_DRIVER_OPERATION_FAILED(36) - Failed in fw_control
[HailoRT] [error] CHECK_SUCCESS failed with status=HAILO_DRIVER_OPERATION_FAILED(36) - Failed to send fw control
[HailoRT] [error] CHECK_SUCCESS failed with status=HAILO_DRIVER_OPERATION_FAILED(36)
[HailoRT] [error] CHECK_SUCCESS failed with status=HAILO_DRIVER_OPERATION_FAILED(36)
[HailoRT] [error] CHECK_SUCCESS failed with status=HAILO_DRIVER_OPERATION_FAILED(36) - Failed to reset context switch state machine
[HailoRT] [error] Failed deactivating core-op (status HAILO_DRIVER_OPERATION_FAILED(36))
[HailoRT] [error] Failed to deactivate low level streams with HAILO_DRIVER_OPERATION_FAILED(36)
[HailoRT] [error] Failed deactivating core-op (status HAILO_DRIVER_OPERATION_FAILED(36))
[HailoRT] [error] Failed deactivate HAILO_DRIVER_OPERATION_FAILED(36)
[ERROR] Received a timeout - hailort has failed because a timeout had occurred
[HailoRT] [error] Ioctl HAILO_FW_CONTROL failed with 19. Read dmesg log for more info
[HailoRT] [error] CHECK_SUCCESS failed with status=HAILO_DRIVER_OPERATION_FAILED(36) - Failed in fw_control
[HailoRT] [error] CHECK_SUCCESS failed with status=HAILO_DRIVER_OPERATION_FAILED(36) - Failed to send fw control
[HailoRT] [error] CHECK_SUCCESS failed with status=HAILO_DRIVER_OPERATION_FAILED(36)
[HailoRT] [error] CHECK_SUCCESS failed with status=HAILO_DRIVER_OPERATION_FAILED(36)
[HailoRT] [warning] clear configured apps ended with status HAILO_DRIVER_OPERATION_FAILED(36)
Traceback (most recent call last):
  File "/workspace/benchmark.py", line 182, in <module>
    run_final_benchmark_rpi(args.model, args.video)
  File "/workspace/benchmark.py", line 155, in run_final_benchmark_rpi
    total_dur = time.time() - start_time_global
UnboundLocalError: local variable 'start_time_global' referenced before assignment


```
