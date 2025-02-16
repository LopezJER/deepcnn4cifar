import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.info("Setting up model training libraries. Torch might take a minute.")
from src.core.config import paths
from src.scripts.train import run_train_pipeline, visualize_losses
from src.scripts.evaluate import (
    run_evaluate_pipeline,
    plot_confusion_matrix,
    plot_roc_curves
)
from src.core.config import class_names

# Log the start of training
logger.info("Starting model training...")

# Initialize training pipeline and train the model
# If debug mode is on, only 2 epochs :)
vgg_model, train_losses, val_losses = run_train_pipeline()
logger.info("Model training complete.")

# Log the start of evaluation
logger.info("Starting model evaluation...")

# Evaluate the trained model
evaluation_results = run_evaluate_pipeline(vgg_model)

if evaluation_results:
    outputs_dir = paths['outputs_dir']
    accuracy, loss, precision, recall, f1, cm, labels_oh, probs = evaluation_results

    # Log the performance metrics in one line
    logger.info(f"Metrics - Accuracy: {accuracy:.2f}%, Loss: {loss:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

    # Save confusion matrix and ROC curves to outputs folder
    logger.info("Saving confusion matrix...")
    plot_confusion_matrix(cm, class_names, save_path=f"{outputs_dir}/confusion_matrix.png", show=False)
    
    logger.info("Saving ROC curves...")
    plot_roc_curves(labels_oh, probs, class_names, save_path=f"{outputs_dir}/roc_curves.png", show=False)

    logger.info("Saving train/val loss plots...")
    visualize_losses(train_losses, val_losses, save_path=f"{outputs_dir}/loss_plot.png", show=False)

# Log completion of the process
logger.info(f"Model evaluation complete. Basic evaluattion plots saved to {outputs_dir}.")

