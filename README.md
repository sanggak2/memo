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
    
    # 2.1 HEF 파일 로드
    hef = HEF(hef_path)
    
    # 3. 네트워크 설정 (PCIe) - 리소스 예약 단계
    configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
    network_groups = target.configure(hef, configure_params)
    network_group = network_groups[0]
    
    # 4. 스트림 파라미터 설정
    input_params = InputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)
    output_params = OutputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)

    # [핵심 수정] 네트워크 그룹 활성화 (Activate)
    # 하드웨어 엔진을 켜는 단계입니다. 이게 없으면 추론이 불가능합니다.
    with network_group.activate():

        # 5. 추론 실행 파이프라인
        with InferVStreams(network_group, input_params, output_params) as pipeline:
            input_info = network_group.get_input_vstream_infos()[0]
            # 더미 데이터 생성 (Batch, H, W, Ch)
            input_data = {
                input_info.name: np.random.random((1, 640, 640, 3)).astype(np.float32)
            }
            
            print("[Run] Starting Inference with Dummy Data...")
            
            # 워밍업 (Warm-up)
            for _ in range(3): 
                pipeline.infer(input_data)
                
            # 실제 성능 측정
            t0 = time.time()
            output = pipeline.infer(input_data)
            dt = time.time() - t0
            
            print(f"\n[Success] Inference Logic Complete! ({dt*1000:.2f}ms)")
            # [수정] 결과 데이터 타입 및 구조 확인 (Debug Mode)
            print("\n[Result Analysis]")
            for name, data in output.items():
                print(f"Layer: '{name}'")
                print(f" - Type: {type(data)}")
            
                # 1. 만약 리스트라면?
                if isinstance(data, list):
                    print(f" - Is List! Length: {len(data)}")
                    if len(data) > 0:
                        print(f" - Element 0 Type: {type(data[0])}")
                        # 리스트 안에 또 리스트가 있는지, 아니면 배열이 있는지 확인
                        if hasattr(data[0], 'shape'):
                             print(f" - Element 0 Shape: {data[0].shape}")
                        else:
                             print(f" - Element 0 Content: {data[0]}")
            
                # 2. 만약 넘파이 배열이라면?
                elif isinstance(data, np.ndarray):
                    print(f" - Is Numpy Array! Shape: {data.shape}")
                
                # 3. 그 외
                else:
                    print(f" - Unknown Content: {data}")

print(f"\n[Done] Total Check Time: {time.time() - start_time:.2f}s")
```

Error
```

```
