import torch
import logging

# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

class GradCAM:
    """
    GradCAM class to generate Gradient-weighted Class Activation Mapping (Grad-CAM)
    for a given model and target layer.
    
    Attributes:
        model (torch.nn.Module): The model to use for generating Grad-CAM.
        target_layer (str): The name of the target layer for the CAM.
        gradients (torch.Tensor): The gradients of the target layer.
        activations (torch.Tensor): The activations of the target layer.
    """

    def __init__(self, model, target_layer):
        """
        Initializes the GradCAM object with the model and target layer.
        
        Args:
            model (torch.nn.Module): The model from which Grad-CAM will be generated.
            target_layer (str): The name of the target layer where activations and gradients will be captured.
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        """
        Registers hooks to the target layer to capture activations and gradients.
        """
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                module.register_forward_hook(self._save_activations)
                module.register_backward_hook(self._save_gradients)
                logger.info(f"Registered hooks for layer: {name}")

    def _save_activations(self, module, input, output):
        """
        Saves the activations during the forward pass.

        Args:
            module (torch.nn.Module): The module (layer) from which activations are obtained.
            input (Tensor): The input tensor to the module.
            output (Tensor): The output tensor from the module (activations).
        """
        self.activations = output
        logger.debug(f"Activations captured for layer {self.target_layer}")

    def _save_gradients(self, module, grad_input, grad_output):
        """
        Saves the gradients during the backward pass.

        Args:
            module (torch.nn.Module): The module (layer) for which gradients are calculated.
            grad_input (Tuple[Tensor]): Gradients with respect to the input.
            grad_output (Tuple[Tensor]): Gradients with respect to the output.
        """
        self.gradients = grad_output[0]
        logger.debug(f"Gradients captured for layer {self.target_layer}")

    def generate_cam(self, input_image, target_class):
        """
        Generates the Grad-CAM for a given input image and target class.
        
        Args:
            input_image (Tensor): The input image to generate Grad-CAM for.
            target_class (int): The index of the target class for which the Grad-CAM is generated.
        
        Returns:
            numpy.ndarray: The normalized Grad-CAM heatmap.
        """
        self.model.zero_grad()
        output = self.model(input_image)
        output[:, target_class].backward()

        weights = torch.mean(self.gradients, dim=(2, 3))
        cam = torch.zeros(self.activations.shape[2:], dtype=torch.float32)
        for i, w in enumerate(weights[0]):
            cam += w * self.activations[0, i, :, :]
        cam = torch.clamp(cam, min=0).cpu().numpy()
        logger.info("Generated Grad-CAM heatmap.")
        return (cam - cam.min()) / (cam.max() - cam.min())
