from prot_eye.spatial_tensor_builder import (
    build_spatial_graph_tensors
)

from inference.predictor import (
    ProtEyePredictor
)

from prot_eye.pdb_writer import (
    write_predicted_structure
)

predictor = ProtEyePredictor()

tensors = build_spatial_graph_tensors(
    "data/sample/1AAR.pdb"
)

predicted_coords = predictor.predict(
    tensors
)

write_predicted_structure(
    "data/sample/1AAR.pdb",
    "data/sample/1AAR_predicted.pdb",
    predicted_coords.numpy()
)

print("Saved")

