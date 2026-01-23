import torch
from src.alcohol_classifier.model import BeverageModel


def export_to_onnx(checkpoint_path="models/best_model.pt", output_path="models/model.onnx"):

    model = BeverageModel()

    print(f"Loading weights from {checkpoint_path}...")
    state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    if "state_dict" in state_dict:
        model.load_state_dict(state_dict["state_dict"])
    else:
        model.load_state_dict(state_dict)

    model.eval()

    dummy_input = torch.randn(1, 3, 224, 224)

    print("Starting standard ONNX export...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],

        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f"✅ Success! Model saved to: {output_path}")

if __name__ == "__main__":
    export_to_onnx()
