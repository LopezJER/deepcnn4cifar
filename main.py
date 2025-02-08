import logging
from src.scripts.train import run_train_pipeline, visualize_losses
from src.scripts.evaluate import (
    run_evaluate_pipeline,
    plot_confusion_matrix,
    plot_roc_curves
)
from src.core.config import class_names

# Set up logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

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

# If evaluation results are returned, process and log the metrics
if evaluation_results:
    accuracy, loss, precision, recall, f1, cm, labels_oh, probs = evaluation_results

    # Log the performance metrics
    logger.info(f"Accuracy: {accuracy:.2f}%")
    logger.info(f"Loss: {loss:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    
    # Log confusion matrix and ROC curve plotting
    logger.info("Plotting confusion matrix...")
    plot_confusion_matrix(cm, class_names)
    
    logger.info("Plotting ROC curves...")
    plot_roc_curves(labels_oh, probs, class_names)

    logger.info("Plotting train/val losses...")
    visualize_losses(train_losses, val_losses)

# Log completion of the process
logger.info("Model evaluation and plotting complete.")
