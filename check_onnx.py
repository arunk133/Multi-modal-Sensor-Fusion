import onnxruntime as ort

print("Inspecting pointpillars.onnx...")
sess = ort.InferenceSession("pointpillars.onnx", providers=['CPUExecutionProvider'])

print("\n--- REQUIRED INPUTS ---")
for i in sess.get_inputs():
    print(f"Name: '{i.name}' | Shape: {i.shape} | Type: {i.type}")

print("\n--- EXPECTED OUTPUTS ---")
for o in sess.get_outputs():
    print(f"Name: '{o.name}' | Shape: {o.shape} | Type: {o.type}")