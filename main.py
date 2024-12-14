import os
import torch
import logging
from torch.utils.data import DataLoader
from src.dataset import MFCCDataset, get_data_transforms
from src.train import TrainingMode, train_model  # Remove run_cross_validation
from src.model import initialize_model, initialize_criterion, initialize_optimizer
from src.evaluation import evaluate_model
from src.utils import setup_logging

def initialize_model_fn(classes, device, config, class_weights_tensor):
    """Factory function for model initialization."""
    def init_fn():
        model = initialize_model(len(classes), device)
        criterion = initialize_criterion(class_weights_tensor)
        optimizer = initialize_optimizer(
            model,
            learning_rate=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        return model, criterion, optimizer, None
    return init_fn

def main():
    # Configuration
    config = {
        'dataset_dir': "./preprocessed",
        'save_dir': "./model",
        'batch_size': 32,
        'num_epochs': 1, 
        'n_splits': 2,
        'test_split_ratio': 0.2,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'training_mode': TrainingMode.SIMPLE,  # or TrainingMode.KFOLD
        'early_stopping_patience': 5,
        'scheduler_fn': None # For k-fold
    }

    # Setup
    setup_logging()
    os.makedirs(config['save_dir'], exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load and prepare data
    train_paths, train_labels, test_paths, test_labels, classes, _ = MFCCDataset.split_train_test(
        config['dataset_dir'], 
        test_size=config['test_split_ratio'],
        n_splits=config['n_splits'] if config['training_mode'] == TrainingMode.KFOLD else 1
    )

    # Add classes to config
    config['classes'] = classes

    train_dataset = MFCCDataset(train_paths, train_labels, classes, transform=get_data_transforms())
    test_dataset = MFCCDataset(test_paths, test_labels, classes, transform=get_data_transforms())

    # Determine number of workers
    if torch.backends.mps.is_available():
        num_workers = 0  # Multiprocessing not supported with MPS
        pin_memory = False
    else:
        num_workers = os.cpu_count()
        pin_memory = True

    # Data loaders with multiprocessing
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    # Prepare class weights using the updated method
    class_weights = MFCCDataset.compute_class_weights(train_labels, classes)
    class_weights_tensor = torch.tensor([class_weights[label] for label in classes], 
                                      device=device, dtype=torch.float32)
    
    # Log class weights for verification
    # logging.info("Class weights:")
    for cls, weight in zip(classes, class_weights_tensor.cpu().numpy()):
        logging.info(f"{cls}: {weight:.4f}")

    # Initialize model function
    model_fn = initialize_model_fn(classes, device, config, class_weights_tensor)

    # Training
    try:
        results, best_state = train_model(
            mode=config['training_mode'],
            model_fn=model_fn,
            dataset=train_dataset,
            device=device,
            config=config
        )

        # Log training results
        if isinstance(results, list):
            logging.info("K-fold cross-validation results:")
            for i, fold_result in enumerate(results):
                logging.info(f"Fold {i+1}: Loss = {fold_result['loss']:.4f}, "
                           f"Accuracy = {fold_result['accuracy']:.4f}")
        else:
            logging.info(f"Training completed: Loss = {results['loss']:.4f}, "
                        f"Accuracy = {results['accuracy']:.4f}")

        # Final evaluation
        logging.info("Evaluating final model on test set...")
        try:
            model = initialize_model(len(classes), device)
            if best_state:
                model.load_state_dict(best_state['model_state_dict'])
                evaluation_results = evaluate_model(model, test_loader, device, classes)
                logging.info("Evaluation results:")
                for metric, value in evaluation_results.items():
                    if isinstance(value, dict):
                        logging.info(f"{metric}:")
                        for k, v in value.items():
                            logging.info(f"  {k}: {v:.4f}")
                    else:
                        logging.info(f"{metric}: {value:.4f}")
        except Exception as e:
            logging.error(f"Error in final evaluation: {e}")
            raise

    except Exception as e:
        logging.error(f"Error during training: {e}")
        raise

if __name__ == "__main__":
    main()
