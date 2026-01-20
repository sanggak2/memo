# memo
오픈 메모장

```
import numpy as np
import cv2
import time
from hailo_platform import VDevice, HailoStreamInterface, InferVStreams, ConfigureParams, InputVStreamParams, OutputVStreamParams, FormatType, HEF

# ==========================================
# 1. 설정 및 이미지 로드
# ==========================================
hef_path = "yolov8n.hef"
image_path = "zidane.jpg"
labels = {0: "person", 56: "tie"} # 테스트용 라벨

print(f"[1] Loading & Preprocessing {image_path}...")
orig_img = cv2.imread(image_path)
if orig_img is None:
    print(f"Error: Could not read {image_path}. Check file path.")
    exit()

h, w, _ = orig_img.shape
input_img = cv2.resize(orig_img, (640, 640))
input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
input_data = np.expand_dims(input_img, axis=0).astype(np.float32)

# ==========================================
# 2. NPU 추론 실행
# ==========================================
print(f"[2] Running Inference on Hailo-8...")
params = VDevice.create_params()
prediction = None # 결과를 담을 변수 초기화

with VDevice(params) as target:
    hef = HEF(hef_path)
    configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
    network_groups = target.configure(hef, configure_params)
    network_group = network_groups[0]
    
    input_params = InputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)
    output_params = OutputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)

    with network_group.activate():
        with InferVStreams(network_group, input_params, output_params) as pipeline:
            input_info = network_group.get_input_vstream_infos()[0]
            
            # 추론 (Inference)
            raw_output = pipeline.infer({input_info.name: input_data})
            
            # ==========================================
            # [핵심 로직] 진짜 Detection Layer 찾기 (Auto-Hunt)
            # ==========================================
            print("\n[Analysis] Searching for Detection Head...")
            
            found = False
            for key, value in raw_output.items():
                # 1. 리스트 껍질 벗기기
                data = value
                while isinstance(data, list):
                    data = data[0]
                
                print(f" - Checking Layer '{key}': Shape {data.shape}")
                
                # 2. 진짜인지 검사 (YOLOv8n은 보통 8400개의 박스를 출력함)
                # (1, 84, 8400) 또는 (84, 8400) 형태를 찾아야 함
                if 8400 in data.shape:
                    print(f"   -> [FOUND!] This is the target layer.")
                    
                    # 3. Shape 정규화: (8400, 84)로 만들기
                    # 만약 (1, 84, 8400) -> (84, 8400)
                    if data.ndim == 3: data = data[0]
                    
                    # 만약 (84, 8400)이면 -> (8400, 84)로 전치
                    if data.shape[0] < data.shape[1]:
                        prediction = np.transpose(data, (1, 0))
                    else:
                        prediction = data
                        
                    found = True
                    break # 찾았으니 루프 종료
                else:
                    print("   -> (Skipping metadata/aux layer)")

            if not found:
                print("[Error] Failed to find a layer with '8400' anchors.")
                exit()

# ==========================================
# 3. 후처리 및 시각화 (Post-Processing)
# ==========================================
print(f"\n[3] Post-processing...")

# prediction은 이제 (8400, 84) 모양입니다.
# [cx, cy, w, h, class_probs...]
class_scores = prediction[:, 4:] # 뒤쪽 80개가 클래스 확률
max_scores = np.max(class_scores, axis=1) # 각 박스별 최대 확률
best_idx = np.argmax(max_scores) # 1등 인덱스
best_score = max_scores[best_idx]
class_id = np.argmax(class_scores[best_idx])

# 결과 시각화
if best_score > 0.5: # 50% 이상 확실할 때만
    box = prediction[best_idx, :4]
    cx, cy, bw, bh = box
    
    # 좌표 복원 (640 -> 원본 크기)
    x1 = int((cx - bw/2) * w / 640)
    y1 = int((cy - bh/2) * h / 640)
    x2 = int((cx + bw/2) * w / 640)
    y2 = int((cy + bh/2) * h / 640)

    label_name = labels.get(class_id, f"Class {class_id}")
    label_text = f"{label_name}: {best_score:.2f}"
    
    print(f"\n[DETECTED] ★ {label_text} at [{x1}, {y1}, {x2}, {y2}]")
    
    # 박스 그리기 (초록색)
    cv2.rectangle(orig_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.putText(orig_img, label_text, (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # 저장
    cv2.imwrite("result.jpg", orig_img)
    print(f"[Success] Saved visualization to 'result.jpg'")
    
else:
    print(f"[Fail] No object detected (Max score: {best_score:.2f})")
```

Error
```
[1] Loading & Preprocessing zidane.jpg...
[2] Running Inference on Hailo-8...
[Debug] Raw Tensor Shape: (3, 5)
[Debug] Final Prediction Shape: (5, 3)
[3] Post-processing...
Traceback (most recent call last):
  File "/workspace/testv8n/verify_yolo.py", line 65, in <module>
    max_scores = np.max(class_scores, axis=1) # 각 박스별 최대 확률
  File "/usr/local/lib/python3.10/dist-packages/numpy/core/fromnumeric.py", line 2810, in max
    return _wrapreduction(a, np.maximum, 'max', axis, None, out,
  File "/usr/local/lib/python3.10/dist-packages/numpy/core/fromnumeric.py", line 88, in _wrapreduction
    return ufunc.reduce(obj, axis, dtype, out, **passkwargs)
ValueError: zero-size array to reduction operation maximum which has no identity


```
