# memo
오픈 메모장

```
import numpy as np
import cv2
from hailo_platform import VDevice, HailoStreamInterface, InferVStreams, ConfigureParams, InputVStreamParams, OutputVStreamParams, FormatType, HEF

# 1. 설정
hef_path = "yolov8n.hef"
image_path = "zidane.jpg"
labels = {0: "person", 56: "tie"} # COCO 클래스 일부 (테스트용)

# 2. 이미지 전처리 (Letterbox 없이 단순 Resize for Test)
# 실제로는 비율 유지(Letterbox)를 해야 정확도가 높지만, 검증용으론 이걸로 충분합니다.
print(f"[1] Loading & Preprocessing {image_path}...")
orig_img = cv2.imread(image_path)
h, w, _ = orig_img.shape
input_img = cv2.resize(orig_img, (640, 640))
input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
input_data = np.expand_dims(input_img, axis=0).astype(np.float32) # (1, 640, 640, 3)

# 3. NPU 추론
print(f"[2] Running Inference on Hailo-8...")
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
            # Inference
            raw_output = pipeline.infer({input_info.name: input_data})
            
            # 결과 텐서 추출 (Batch, 84, 8400)
            # [수정] 무적의 Shape 처리 로직
            # 1. 딕셔너리에서 값 꺼내기
            output_data = list(raw_output.values())[0]

            # 2. 리스트 껍질이 나올 때까지 계속 벗기기 (Recursive Unwrapping)
            while isinstance(output_data, list):
                output_data = output_data[0]

            print(f"[Debug] Raw Tensor Shape: {output_data.shape}")

            # 3. 차원에 따른 자동 대응
            if output_data.ndim == 3:
                # (1, 84, 8400) -> 배치 차원 제거 -> (84, 8400)
                output_data = output_data[0]
            
            # 4. (84, 8400) -> (8400, 84)로 전치
            # 만약 이미 (8400, 84)라면 전치 생략
            if output_data.shape[0] < output_data.shape[1]: 
                prediction = np.transpose(output_data, (1, 0))
            else:
                prediction = output_data
            
            print(f"[Debug] Final Prediction Shape: {prediction.shape}")

# 4. 후처리 (간이 버전: 가장 높은 점수의 객체 찾기)
print(f"[3] Post-processing...")
class_scores = prediction[:, 4:] # 뒤쪽 80개가 클래스 확률
max_scores = np.max(class_scores, axis=1) # 각 박스별 최대 확률
best_idx = np.argmax(max_scores) # 전체 8400개 중 1등 인덱스
best_score = max_scores[best_idx]
class_id = np.argmax(class_scores[best_idx])

# 5. 결과 시각화
if best_score > 0.5: # 50% 이상 확실할 때만
    # 좌표 복원 (0~640 스케일 -> 원본 이미지 스케일)
    box = prediction[best_idx, :4]
    cx, cy, bw, bh = box
    x1 = int((cx - bw/2) * w / 640)
    y1 = int((cy - bh/2) * h / 640)
    x2 = int((cx + bw/2) * w / 640)
    y2 = int((cy + bh/2) * h / 640)

    label_text = f"{labels.get(class_id, f'Class {class_id}')}: {best_score:.2f}"
    print(f"\n[DETECTED] {label_text} at [{x1}, {y1}, {x2}, {y2}]")
    
    # 박스 그리기
    cv2.rectangle(orig_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(orig_img, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
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
