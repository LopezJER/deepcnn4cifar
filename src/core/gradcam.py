import torch
import logging

# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                module.register_forward_hook(self._save_activations)
                module.register_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_cam(self, input_image, target_class):
        """
        Generates the Class Activation Map (CAM).

        Parameters:
            input_image: Input tensor for the model.
            target_class: Class index for which CAM is generated.

        Returns:
            cam: Class Activation Map as a 2D array.
        """
        self.model.zero_grad()
        output = self.model(input_image)
        class_score = output[:, target_class]
        class_score.backward()

        weights = torch.mean(self.gradients, dim=(2, 3))  # Global Average Pooling
        cam = torch.zeros(self.activations.shape[2:], dtype=torch.float32)
        for i, w in enumerate(weights[0]):
            cam += w * self.activations[0, i, :, :]
        cam = torch.clamp(cam, min=0).detach().cpu().numpy()  # Fix: Use .detach()
        cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam
