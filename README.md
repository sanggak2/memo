# memo
오픈 메모장

```
import numpy as np
import time
from hailo_platform import VDevice, HailoStreamInterface, InferVStreams, ConfigureParams, InputVStreamParams, OutputVStreamParams, FormatType

hef_path = "yolov8n.hef"

print(f"[Init] Loading {hef_path} on Hailo-8...")
start_time = time.time()

params = VDevice.create_params()
with VDevice(params) as target:
    hef = target.create_hef(hef_path)
    
    configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
    network_groups = target.configure(hef, configure_params)
    network_group = network_groups[0]
    
    input_params = InputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)
    output_params = OutputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)

    with InferVStreams(network_group, input_params, output_params) as pipeline:
        input_info = network_group.get_input_vstream_infos()[0]
        input_data = {
            input_info.name: np.random.random((1, 640, 640, 3)).astype(np.float32)
        }
        
        print("[Run] Starting Inference with Dummy Data...")
        for _ in range(3): pipeline.infer(input_data) # Warmup
            
        t0 = time.time()
        output = pipeline.infer(input_data)
        dt = time.time() - t0
        
        print(f"\n[Success] Inference Logic Complete! ({dt*1000:.2f}ms)")
        for name, data in output.items():
            print(f" - Output Layer '{name}': Shape {data.shape}")

print(f"\n[Done] Total Check Time: {time.time() - start_time:.2f}s")
```
