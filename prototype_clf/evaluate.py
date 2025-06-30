import torch
import random
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def evaluate_model(model, classifier, dataloader, device, class_feature_map=None):
    model.eval()
    classifier.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for _, labels, _, embeddings in dataloader:
            if embeddings is None:
                continue
            embeddings = embeddings.to(device)
            labels = labels.to(device)
            outputs = model(embeddings)
            logits = classifier(outputs)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = correct / total if total > 0 else 0
    print(f"\nValidation Accuracy: {accuracy:.4f} ({correct}/{total})")

    # --- Calculate precision, recall, F1-score ---
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

    # --- Classification report ---
    if class_feature_map is not None:
        target_names = [k for k, v in sorted(class_feature_map.items(), key=lambda x: x[1])]
    else:
        target_names = None
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=target_names, zero_division=0))

    # --- Confusion matrix ---
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()

def random_inference(prototype_classifier, val_dataset, ):
    sample_idx = random.randint(0, len(val_dataset) - 1)
    sample_image, sample_label, sample_file_path, sample_embedding = val_dataset[sample_idx]

    print(f"Sample image path: {sample_file_path}")
    print(f"True label: {sample_label}")

    if sample_embedding is not None:
        sample_embedding_tensor = torch.tensor(sample_embedding, dtype=torch.float32).unsqueeze(0)
        
        # Get prediction probabilities
        with torch.no_grad():
            probas = prototype_classifier.make_prediction(sample_embedding_tensor)
            
        # Get top 5 predictions
        top5_probs, top5_indices = torch.topk(probas, k=5, dim=1)
        
        print(f"\nTop 5 predictions:")
        for i in range(5):
            pred_class_id = top5_indices[0][i].item()
            pred_prob = top5_probs[0][i].item()
            class_name = [k for k, v in val_dataset.class_feature_map.items() if v == pred_class_id][0]
            print(f"  {i+1}. {class_name} (Class {pred_class_id}): {pred_prob:.4f}")
            
        # Check if true class is in top 5
        true_class_in_top5 = sample_label in top5_indices[0].tolist()
        print(f"\nTrue class in top 5: {true_class_in_top5}")
        
        if true_class_in_top5:
            true_class_rank = (top5_indices[0] == sample_label).nonzero(as_tuple=True)[0].item() + 1
            print(f"True class rank: {true_class_rank}")
    else:
        print("No embedding available for sample image")