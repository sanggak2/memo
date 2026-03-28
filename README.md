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
                        
                        if len(det_8.shape) == 3:
                            det_8 = np.expand_dims(det_8, axis=0)
                            det_16 = np.expand_dims(det_16, axis=0)
                            det_32 = np.expand_dims(det_32, axis=0)
                            da_seg = np.expand_dims(da_seg, axis=0)
                            ll_seg = np.expand_dims(ll_seg, axis=0)

                            # 3) NCHW 포맷일 경우 NHWC 포맷으로 변환
                        if len(det_8.shape) == 4 and det_8.shape[1] == 18: 
                            det_8 = np.transpose(det_8, (0, 2, 3, 1))
                            det_16 = np.transpose(det_16, (0, 2, 3, 1))
                            det_32 = np.transpose(det_32, (0, 2, 3, 1))
                        if len(da_seg.shape) == 4 and da_seg.shape[1] == 2:
                            da_seg = np.transpose(da_seg, (0, 2, 3, 1))
                            ll_seg = np.transpose(ll_seg, (0, 2, 3, 1))

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
cv2.setNumThreads(1) # [변인통제] OpenCV 다중 스레드 개입 차단
import numpy as np
import psutil
import csv
import threading
import queue
import gc
from datetime import datetime
from hailo_platform import VDevice, FormatType, HailoSchedulingAlgorithm

# -------------------------------------------------
# 0. YOLOP Post-Processing Constants & Utils (동일)
# -------------------------------------------------
ANCHORS = [[[3, 9], [5, 11], [4, 20]], [[7, 18], [6, 39], [12, 31]], [[19, 50], [38, 81], [68, 157]]]
STRIDES = [8, 16, 32]

def sigmoid(x): return 1 / (1 + np.exp(-x))

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
                try:
                    with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                        self.stats["temp"] = int(f.read().strip()) / 1000.0
                except:
                    self.stats["temp"] = 0.0 # 파일 접근 실패 시 방어
                time.sleep(self.interval)
            except: break
    def stop(self): self.running = False

# -------------------------------------------------
# 2. Fair Async Benchmark
# -------------------------------------------------
class HailoAsyncBenchmark:
    def __init__(self, model_path):
        try:
            os.sched_setscheduler(os.getpid(), os.SCHED_FIFO, os.sched_param(99))
            print("[INFO] Process Priority: REAL-TIME (FIFO 99)")
        except: pass

        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        self.target = VDevice(params)
        self.infer_model = self.target.create_infer_model(model_path)
        
        self.infer_model.inputs[0].set_format_type(FormatType.UINT8)
        self.input_shape = self.infer_model.inputs[0].shape
        self.in_h, self.in_w = (self.input_shape[1], self.input_shape[2]) if len(self.input_shape) == 4 else (self.input_shape[0], self.input_shape[1])
        self.input_name = self.infer_model.inputs[0].name

        self.output_info = []
        for output in self.infer_model.outputs:
            output.set_format_type(FormatType.FLOAT32)
            self.output_info.append({"name": output.name, "shape": output.shape})

        # [변수 통제] 과도한 스위칭 방지를 위해 Worker는 1개만 둬서 Sync와 CPU 사용 패턴을 유사하게 맞춤
        self.POOL_SIZE = 10
        self.NUM_WORKERS = 1 

        self.buffer_pool = queue.Queue(maxsize=self.POOL_SIZE)
        self.postprocess_queue = queue.Queue(maxsize=self.POOL_SIZE)
        self.log_queue = queue.Queue()
        self.monitor = SystemMonitor()

    def postprocess_worker(self):
        while True:
            item = self.postprocess_queue.get()
            if item is None:
                self.postprocess_queue.task_done()
                break
                
            f_id, t_e2e_st, t_inf_st, t_inf_end, rx, ry, bindings, input_buf, output_bufs = item

            try:
                try:
                    det_8, det_16, det_32 = output_bufs['model/conv57'], output_bufs['model/conv65'], output_bufs['model/conv72']
                    da_seg, ll_seg = output_bufs['model/ne_activation_activation1'], output_bufs['model/ne_activation_activation2']
                except KeyError:
                    det_8 = next(v for k, v in output_bufs.items() if 'conv57' in k)
                    det_16 = next(v for k, v in output_bufs.items() if 'conv65' in k)
                    det_32 = next(v for k, v in output_bufs.items() if 'conv72' in k)
                    da_seg = next(v for k, v in output_bufs.items() if 'activation1' in k)
                    ll_seg = next(v for k, v in output_bufs.items() if 'activation2' in k)
                
                if len(det_8.shape) == 4 and det_8.shape[1] == 18: 
                    det_8 = np.transpose(det_8, (0, 2, 3, 1))
                    det_16 = np.transpose(det_16, (0, 2, 3, 1))
                    det_32 = np.transpose(det_32, (0, 2, 3, 1))
                if len(da_seg.shape) == 4 and da_seg.shape[1] == 2:
                    da_seg, ll_seg = np.transpose(da_seg, (0, 2, 3, 1)), np.transpose(ll_seg, (0, 2, 3, 1))

                da_diff = da_seg[0][..., 1] - da_seg[0][..., 0]
                da_mask = np.zeros_like(da_diff, dtype=np.uint8)
                da_mask[da_diff > 0.0] = 1 
                ll_mask = np.argmax(ll_seg[0], axis=-1).astype(np.uint8)

                boxes, scores, class_ids = decode_detections([det_8, det_16, det_32], conf_thres=0.4)
                final_boxes = []
                for b in boxes:
                    x1, y1, w, h = b
                    final_boxes.append([int(x1 * rx), int(y1 * ry), int(w * rx), int(h * ry)])
                _ = cv2.dnn.NMSBoxes(final_boxes, scores, score_threshold=0.4, nms_threshold=0.45)

            except Exception as e: pass

            # [공정 평가] Sync와 완전히 동일한 구간에서 E2E 종료 시간 측정
            t_e2e_end = time.perf_counter()
            t_now = time.time()
            
            infer_lat = (t_inf_end - t_inf_st) * 1000.0  # (run_async ~ callback)
            e2e_lat = (t_e2e_end - t_e2e_st) * 1000.0    # 단일 프레임 처리 속도 (Instant)
            fps_inst = 1000.0 / e2e_lat if e2e_lat > 0 else 0 

            self.log_queue.put((f_id, t_now, fps_inst, infer_lat, e2e_lat, self.monitor.stats["cpu"], self.monitor.stats["temp"]))

            if f_id % 200 == 0:
                print(f"[{f_id}] Infer: {infer_lat:.2f}ms | E2E: {e2e_lat:.2f}ms")

            self.buffer_pool.put((bindings, input_buf, output_bufs))
            self.postprocess_queue.task_done()

    def run_benchmark(self, video_path, max_frames=10000):
        cap = cv2.VideoCapture(os.path.abspath(video_path))
        if not cap.isOpened(): raise RuntimeError("Video Open Failed")

        self.monitor.start()
        
        workers = []
        for _ in range(self.NUM_WORKERS):
            t = threading.Thread(target=self.postprocess_worker, daemon=True)
            t.start()
            workers.append(t)
        
        with self.infer_model.configure() as configured_infer_model:
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

            # ---------------------------------------------------------
            # [변인통제] Sync 코드와 동일한 Warmup 절차 이식
            # ---------------------------------------------------------
            print("[INFO] Warming up pipeline...")
            for _ in range(30): cap.read()
            for _ in range(50):
                bindings, input_buf, output_bufs = self.buffer_pool.get()
                np.copyto(input_buf, np.zeros(self.input_shape, dtype=np.uint8))
                
                # [수정됨] 람다 대신 안전한 클로저 함수로 버퍼 캡처 및 반납
                def get_dummy_cb(b, i_b, o_b):
                    def cb(completion_info):
                        self.buffer_pool.put((b, i_b, o_b))
                    return cb
                
                job = configured_infer_model.run_async([bindings], get_dummy_cb(bindings, input_buf, output_bufs))
                if 'job' in locals(): job.wait(5000)
            
            # ---------------------------------------------------------
            # [변인통제] 가비지 컬렉터 끄기 (동기 코드와 환경 동일화)
            # ---------------------------------------------------------
            gc.disable()
            print("[INFO] Fair Measurement Started!")
            start_time_global = time.time()

            frame_id = 0
            while frame_id < max_frames:
                ret, frame = cap.read()
                if not ret: break

                t_e2e_st = time.perf_counter()
                h0, w0 = frame.shape[:2]
                rx, ry = w0 / self.in_w, h0 / self.in_h

                resized = cv2.resize(frame, (self.in_w, self.in_h))
                input_data = np.expand_dims(resized, axis=0)

                bindings, input_buf, output_bufs = self.buffer_pool.get()
                np.copyto(input_buf, input_data)

                t_inf_st = time.perf_counter()

                def get_callback(f_id, t_e2e, t_inf_s, r_x, r_y, current_binding, curr_in, curr_outs):
                    def cb(completion_info):
                        t_inf_end = time.perf_counter() # Callback 진입 즉시 측정
                        if completion_info.exception:
                            self.buffer_pool.put((current_binding, curr_in, curr_outs))
                        else:
                            self.postprocess_queue.put((f_id, t_e2e, t_inf_s, t_inf_end, r_x, r_y, current_binding, curr_in, curr_outs))
                    return cb

                job = configured_infer_model.run_async([bindings], get_callback(frame_id, t_e2e_st, t_inf_st, rx, ry, bindings, input_buf, output_bufs))
                frame_id += 1

            if 'job' in locals(): job.wait(10000)
            
            self.postprocess_queue.join()
            for _ in range(self.NUM_WORKERS): self.postprocess_queue.put(None)
            for t in workers: t.join()
                
            total_dur = time.time() - start_time_global

        # 측정 종료 후 GC 켜기
        gc.enable()
        self.monitor.stop()
        cap.release()

        logs = []
        while not self.log_queue.empty(): logs.append(self.log_queue.get())
        logs.sort(key=lambda x: x[0]) 

        self._save_logs(logs, frame_id, total_dur)

    def _save_logs(self, logs, total_frames, duration):
        os.makedirs("logs", exist_ok=True)
        log_path = f"logs/bench_rpi_async_fair_{datetime.now().strftime('%m%d_%H%M')}.csv"

        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Frame_ID", "Unix_Time", "Timestamp", "Real_Throughput_FPS", "HDH_Latency_ms", "E2E_Latency_ms", "CPU_Usage_Percent", "Temp_C"])

            prev_time = logs[0][1] if logs else 0
            for i, log in enumerate(logs):
                fid, ts, _, hdh_lat, e2e_lat, cpu, tmp = log
                
                if i == 0:
                    real_fps = 0.0 # 첫 프레임은 비교 대상이 없으므로 0
                else:
                    time_diff = ts - prev_time
                    real_fps = 1.0 / time_diff if time_diff > 0 else 0.0
                prev_time = ts

                writer.writerow([
                    fid, f"{ts:.6f}", datetime.fromtimestamp(ts).strftime("%H:%M:%S.%f"), 
                    f"{real_fps:.2f}", f"{hdh_lat:.2f}", f"{e2e_lat:.2f}", cpu, tmp
                ])

        global_e2e_fps = total_frames / duration if duration > 0 else 0
        avg_hdh_lat = np.mean([log[4] for log in logs]) if logs else 0
        
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
        bench.run_benchmark(args.video, max_frames=2000)
```

GStreamer 비동기
```
import time
import argparse
import os
import cv2
import numpy as np
import psutil
import csv
import threading
from datetime import datetime

# GStreamer 필수 라이브러리
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

try:
    import hailo # Hailo TAPPAS GStreamer Python API
except ImportError:
    print("[ERROR] Hailo TAPPAS Python API('hailo' module)가 설치되어 있지 않습니다.")
    print("TAPPAS 환경(hailo-rpi5-examples 등)에서 실행해주세요.")
    exit(1)

# -------------------------------------------------
# 0. YOLOP Post-Processing Constants
# -------------------------------------------------
ANCHORS = [[[3, 9], [5, 11], [4, 20]], [[7, 18], [6, 39], [12, 31]], [[19, 50], [38, 81], [68, 157]]]
STRIDES = [8, 16, 32]
def sigmoid(x): return 1 / (1 + np.exp(-x))

def decode_detections(det_outputs, conf_thres=0.4, rx=1.0, ry=1.0):
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
        
        pred_conf, pred_cls = pred[..., 4], pred[..., 5]
        final_score = pred_conf * pred_cls
        
        mask = final_score > conf_thres
        if not np.any(mask): continue
            
        valid_xy, valid_wh = pred_xy[mask], pred_wh[mask]
        valid_scores = final_score[mask]
        
        x1y1 = valid_xy - valid_wh / 2
        valid_boxes = np.concatenate([x1y1, valid_wh], axis=1)
        
        for b, s in zip(valid_boxes.tolist(), valid_scores.tolist()):
            x1, y1, bw, bh = b
            boxes.append([int(x1 * rx), int(y1 * ry), int(bw * rx), int(bh * ry)])
            scores.append(s)
            class_ids.append(0)
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
                try:
                    with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                        self.stats["temp"] = int(f.read().strip()) / 1000.0
                except: self.stats["temp"] = 0.0
                time.sleep(self.interval)
            except: break
    def stop(self): self.running = False

# -------------------------------------------------
# 2. GStreamer Benchmark App
# -------------------------------------------------
class GstHailoBenchmark:
    def __init__(self, model_path, video_path):
        Gst.init(None)
        self.frame_times = {}  # 버퍼 PTS 기준으로 프레임별 타임스탬프 기록
        self.logs = []
        self.frame_count = 0
        self.monitor = SystemMonitor()
        
        # 모델 입력 해상도 (YOLOP 기준)
        self.IN_W, self.IN_H = 640, 384
        
        # 원본 영상 해상도 파악 (비율 계산용)
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            self.orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
        else:
            self.orig_w, self.orig_h = 1920, 1080 # Default
            
        self.rx = self.orig_w / self.IN_W
        self.ry = self.orig_h / self.IN_H

        # GStreamer 파이프라인 생성 (이름표 name=... 를 붙여서 나중에 프로브를 설치합니다)
        pipeline_str = (
            f"filesrc location={video_path} ! decodebin ! "
            f"videoconvert ! "
            f"videoscale name=pre_scale ! " # 전처리 시작 지점
            f"video/x-raw,width={self.IN_W},height={self.IN_H},format=RGB ! "
            f"hailonet name=npu_infer hef-path={model_path} scheduling-algorithm=1 ! " # 추론 지점
            f"appsink name=mysink emit-signals=true max-buffers=1 drop=true"
        )
        print(f"[INFO] Pipeline: {pipeline_str}")
        self.pipeline = Gst.parse_launch(pipeline_str)

        # 검문소(Probe) 설치를 위해 파이프라인에서 엘리먼트 가져오기
        pre_elem = self.pipeline.get_by_name("pre_scale")
        infer_elem = self.pipeline.get_by_name("npu_infer")
        sink_elem = self.pipeline.get_by_name("mysink")

        # 패드(데이터 출입구)에 콜백(검문소) 부착
        pre_elem.get_static_pad("sink").add_probe(Gst.PadProbeType.BUFFER, self.probe_pre_start)
        infer_elem.get_static_pad("sink").add_probe(Gst.PadProbeType.BUFFER, self.probe_infer_start)
        infer_elem.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, self.probe_infer_end)

        # 파이썬으로 데이터가 넘어오는 최종 목적지에 콜백 연결
        sink_elem.connect("new-sample", self.on_new_sample)

    # ------------------ 패드 프로브 (타임스탬프 기록) ------------------
    def probe_pre_start(self, pad, info):
        pts = info.get_buffer().pts
        self.frame_times[pts] = {'pre_st': time.perf_counter()}
        return Gst.PadProbeReturn.OK

    def probe_infer_start(self, pad, info):
        pts = info.get_buffer().pts
        if pts in self.frame_times:
            t_now = time.perf_counter()
            self.frame_times[pts]['pre_end'] = t_now
            self.frame_times[pts]['inf_st'] = t_now
        return Gst.PadProbeReturn.OK

    def probe_infer_end(self, pad, info):
        pts = info.get_buffer().pts
        if pts in self.frame_times:
            self.frame_times[pts]['inf_end'] = time.perf_counter()
        return Gst.PadProbeReturn.OK

    # ------------------ Python Post-processing ------------------
    def on_new_sample(self, sink):
        sample = sink.emit("pull-sample")
        buffer = sample.get_buffer()
        pts = buffer.pts
        t_now = time.time()
        
        t_post_st = time.perf_counter()

        # Hailo TAPPAS API를 사용하여 버퍼에서 텐서 메타데이터 추출
        roi = hailo.get_roi_from_buffer(buffer)
        tensors = roi.get_tensors()
        
        output_bufs = {}
        for tensor in tensors:
            name = tensor.name()
            # 텐서 데이터를 numpy 배열로 변환
            np_arr = np.array(tensor)
            output_bufs[name] = np_arr

        # YOLOP 후처리 로직 (기존 코드와 동일)
        try:
            # 이름 매칭 로직 (TAPPAS는 레이어 이름이 다소 다르게 찍힐 수 있으므로 방어적 코드 작성)
            det_8 = next(v for k, v in output_bufs.items() if 'conv57' in k)
            det_16 = next(v for k, v in output_bufs.items() if 'conv65' in k)
            det_32 = next(v for k, v in output_bufs.items() if 'conv72' in k)
            da_seg = next(v for k, v in output_bufs.items() if 'activation1' in k)
            ll_seg = next(v for k, v in output_bufs.items() if 'activation2' in k)

            if len(det_8.shape) == 3: # GStreamer 텐서는 batch 차원이 빠져있을 수 있음
                det_8 = np.expand_dims(det_8, axis=0)
                det_16 = np.expand_dims(det_16, axis=0)
                det_32 = np.expand_dims(det_32, axis=0)
                da_seg = np.expand_dims(da_seg, axis=0)
                ll_seg = np.expand_dims(ll_seg, axis=0)

            boxes, scores, class_ids = decode_detections([det_8, det_16, det_32], rx=self.rx, ry=self.ry)
            _ = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.4, nms_threshold=0.45)
        except Exception as e:
            pass # 초기 Warmup 중 에러 무시

        t_post_end = time.perf_counter()

        # 측정값 수집 및 로깅
        if pts in self.frame_times:
            times = self.frame_times[pts]
            pre_lat = (times.get('pre_end', 0) - times.get('pre_st', 0)) * 1000.0
            inf_lat = (times.get('inf_end', 0) - times.get('inf_st', 0)) * 1000.0
            post_lat = (t_post_end - t_post_st) * 1000.0
            
            # 기록 완료된 프레임 타임스탬프 삭제 (메모리 누수 방지)
            del self.frame_times[pts]

            self.logs.append((self.frame_count, t_now, pre_lat, inf_lat, post_lat, 
                              self.monitor.stats["cpu"], self.monitor.stats["temp"]))

            if self.frame_count % 100 == 0:
                print(f"[{self.frame_count}] Pre: {pre_lat:.2f}ms | Infer: {inf_lat:.2f}ms | Post: {post_lat:.2f}ms")
            
            self.frame_count += 1

        return Gst.FlowReturn.OK

    def run(self, max_frames=5000):
        self.monitor.start()
        print("[INFO] GStreamer Pipeline Playing...")
        self.start_time = time.time()
        
        self.pipeline.set_state(Gst.State.PLAYING)
        
        # GStreamer 메인 루프 실행 (C언어 레벨에서 백그라운드 구동)
        loop = GLib.MainLoop()
        
        # 최대 프레임 또는 종료 이벤트를 감지하기 위한 스레드
        def check_stop():
            while self.frame_count < max_frames:
                time.sleep(1)
            print("[INFO] Max frames reached. Stopping...")
            loop.quit()
            
        threading.Thread(target=check_stop, daemon=True).start()
        
        try:
            loop.run()
        except KeyboardInterrupt:
            pass
            
        self.pipeline.set_state(Gst.State.NULL)
        self.monitor.stop()
        
        duration = time.time() - self.start_time
        self.save_logs(duration)

    def save_logs(self, duration):
        os.makedirs("logs", exist_ok=True)
        log_path = f"logs/bench_rpi_gst_{datetime.now().strftime('%m%d_%H%M')}.csv"

        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Frame_ID", "Unix_Time", "Real_FPS", "Pre_Latency_ms", "Infer_Latency_ms", "Post_Latency_ms", "CPU", "Temp"])

            prev_time = self.logs[0][1] if self.logs else 0
            for i, log in enumerate(self.logs):
                fid, ts, pre, inf, post, cpu, tmp = log
                fps = 0.0 if i == 0 else (1.0 / (ts - prev_time) if ts > prev_time else 0.0)
                prev_time = ts
                
                writer.writerow([fid, f"{ts:.6f}", f"{fps:.2f}", f"{pre:.2f}", f"{inf:.2f}", f"{post:.2f}", cpu, tmp])

        avg_fps = self.frame_count / duration if duration > 0 else 0
        print(f"\n[RESULT] Saved to {log_path}")
        print(f"[RESULT] Global Pipeline FPS: {avg_fps:.2f} FPS")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='yolop_raw.hef')
    parser.add_argument('--video', default='Video.mp4')
    args = parser.parse_args()
    
    app = GstHailoBenchmark(args.model, args.video)
    app.run(max_frames=2000)
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
