# memo
오픈 메모장

```
import numpy as np
import time
from hailo_platform import VDevice, HailoStreamInterface, InferVStreams, ConfigureParams, InputVStreamParams, OutputVStreamParams, FormatType, HEF

# 1. 모델 경로
hef_path = "yolov8n.hef"

print(f"[Init] Loading {hef_path}...")
start_time = time.time()

# 2. VDevice(NPU) 연결
params = VDevice.create_params()
with VDevice(params) as target:
    
    # [수정된 부분] HEF 파일을 클래스로 직접 로드
    hef = HEF(hef_path)
    
    # 3. 네트워크 설정 (PCIe)
    configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
    network_groups = target.configure(hef, configure_params)
    network_group = network_groups[0]
    
    # 4. 스트림 파라미터 설정
    input_params = InputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)
    output_params = OutputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)

    # 5. 추론 실행
    with InferVStreams(network_group, input_params, output_params) as pipeline:
        input_info = network_group.get_input_vstream_infos()[0]
        # 더미 데이터 (Batch, H, W, Ch)
        input_data = {
            input_info.name: np.random.random((1, 640, 640, 3)).astype(np.float32)
        }
        
        print("[Run] Starting Inference with Dummy Data...")
        
        # 워밍업
        for _ in range(3): 
            pipeline.infer(input_data)
            
        # 실제 측정
        t0 = time.time()
        output = pipeline.infer(input_data)
        dt = time.time() - t0
        
        print(f"\n[Success] Inference Logic Complete! ({dt*1000:.2f}ms)")
        for name, data in output.items():
            print(f" - Output Layer '{name}': Shape {data.shape}")

print(f"\n[Done] Total Check Time: {time.time() - start_time:.2f}s")
```
