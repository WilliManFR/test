from onnxsim import simplify
import onnx

model = onnx.load("models\\big\\vit_h_encoder.onnx")
model_simp, check = simplify(model)
onnx.save(model_simp, "models\\big\\vit_h_encoder_opt.onnx")
print("✅ Modèle optimisé")