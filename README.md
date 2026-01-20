# memo
오픈 메모장

```
import numpy as np
import cv2
from hailo_platform import VDevice, HailoStreamInterface, InferVStreams, ConfigureParams, InputVStreamParams, OutputVStreamParams, FormatType, HEF

# ==========================================
# 1. 설정
# ==========================================
hef_path = "yolov8n.hef"
image_path = "zidane.jpg"
# NMS 모델은 보통 클래스 ID가 score에 포함되거나 별도로 나옵니다. 
# 일단 5개 속성(ymin, xmin, ymax, xmax, score)이라고 가정하고 찍어봅니다.

print(f"[1] Loading & Preprocessing {image_path}...")
orig_img = cv2.imread(image_path)
if orig_img is None: exit("Image not found")
h, w, _ = orig_img.shape

# 단순 Resize (640x640)
input_img = cv2.resize(orig_img, (640, 640))
input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
input_data = np.expand_dims(input_img, axis=0).astype(np.float32)

# ==========================================
# 2. NPU 추론
# ==========================================
print(f"[2] Running Inference on Hailo-8 (NMS Mode)...")
params = VDevice.create_params()

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
            
            # 추론
            raw_output = pipeline.infer({input_info.name: input_data})
            
            # 데이터 추출 (첫 번째 레이어)
            output_data = list(raw_output.values())[0]
            
            # 리스트 껍질 벗기기
            while isinstance(output_data, list):
                output_data = output_data[0]
                
            print(f"\n[Debug] Output Shape: {output_data.shape}")
            print(f"[Debug] Raw Data Content:\n{output_data}")

# ==========================================
# 3. 결과 해석 및 그리기 (NMS Output)
# ==========================================
# 가정: Hailo NMS 포맷은 보통 [ymin, xmin, ymax, xmax, score] 순서 (정규화된 0~1 좌표)
# 데이터가 (3, 5)라면 3개의 객체가 검출된 것.

detections = output_data
if detections.ndim == 1: # (5,) 처럼 나오면 차원 추가
    detections = np.expand_dims(detections, axis=0)

print(f"\n[3] Visualizing {len(detections)} detections...")

for i, det in enumerate(detections):
    # 값 5개 추출 (순서는 모델마다 다를 수 있으나 보통 ymin, xmin, ymax, xmax, score)
    # Hailo Defalut: ymin, xmin, ymax, xmax, score
    ymin, xmin, ymax, xmax, score = det[:5]
    
    print(f" - Det {i}: Score={score:.3f}, Box=[{xmin:.2f}, {ymin:.2f}, {xmax:.2f}, {ymax:.2f}]")

    if score > 0.5: # 신뢰도 0.5 이상만
        # 좌표 복원 (0~1 -> 원본 해상도)
        x1 = int(xmin * w)
        y1 = int(ymin * h)
        x2 = int(xmax * w)
        y2 = int(ymax * h)
        
        # 그리기
        cv2.rectangle(orig_img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        label = f"Obj {i}: {score:.2f}"
        cv2.putText(orig_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

# 저장
cv2.imwrite("result_nms.jpg", orig_img)
print(f"[Success] Saved result to 'result_nms.jpg'")
```

Error
```
[1] Loading & Preprocessing zidane.jpg...
[2] Running Inference on Hailo-8...

[Analysis] Searching for Detection Head...
 - Checking Layer 'yolov8n/yolov8_nms_postprocess': Shape (3, 5)
   -> (Skipping metadata/aux layer)
[Error] Failed to find a layer with '8400' anchors.


![result](https://github.com/user-attachments/assets/339cb64e-7875-4fe2-a067-25c92002351d)

```
