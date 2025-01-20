"""
Onnxruntime test
"""
import onnxruntime as ort
import numpy as np

# Charger le modèle ONNX
onnx_model = ort.InferenceSession("models\\big\\vit_h_encoder_not_quantized\\vit_h_encoder.onnx")

# Vérifier les entrées et sorties
print("Entrées :", [i.name for i in onnx_model.get_inputs()])
print("Sorties :", [o.name for o in onnx_model.get_outputs()])

# Tester une prédiction
input_data = np.random.randn(1, 3, 1024, 1024).astype(np.float32)
outputs = onnx_model.run(None, {"image": input_data})
print("✅ Test ONNX réussi")