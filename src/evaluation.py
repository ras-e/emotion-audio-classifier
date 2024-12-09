import torch
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


def evaluate_model(model, dataloader, device, classes):
    """
    Evaluate the trained model on a test dataset.

    Args:
        model (torch.nn.Module): Trained CNN model.
        dataloader (torch.utils.data.DataLoader): DataLoader for test data.
        device (torch.device): Device to run evaluation on (CPU or GPU).
        classes (list): List of class names (emotion labels).

    Returns:
        dict: Evaluation metrics (accuracy, precision, recall, F1-score).
    """
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    # Convert tuple-based classes to string-based labels
    class_names = ["_".join(cls) for cls in classes]


    # Disable gradient calculations for evaluation
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            # Store predictions and labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    accuracy = report["accuracy"]

    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))


    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
   
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()


    return {
        "accuracy": accuracy,
        "precision": {cls: report[cls]["precision"] for cls in class_names},
        "recall": {cls: report[cls]["recall"] for cls in class_names},
        "f1_score": {cls: report[cls]["f1-score"] for cls in class_names},
    }
