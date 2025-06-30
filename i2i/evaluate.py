import torch
import random
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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
    print(f"\nValidation Accuracy: {accuracy:.4f}")

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

    # --- Confusion matrix (seaborn) ---
    cm = confusion_matrix(all_labels, all_preds)
    if class_feature_map is not None:
        class_names = [k for k, v in sorted(class_feature_map.items(), key=lambda x: x[1])]
    else:
        class_names = [str(i) for i in range(cm.shape[0])]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix (Seaborn)')
    plt.show()

def random_inference(prototype_classifier, test_dataset):
    sample_idx = random.randint(0, len(test_dataset) - 1)
    _, sample_label, sample_file_path, sample_embedding = test_dataset[sample_idx]

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
            class_name = [k for k, v in test_dataset.class_feature_map.items() if v == pred_class_id][0]
            print(f"  {i+1}. {class_name} (Class {pred_class_id}): {pred_prob:.4f}")
            
        # Check if true class is in top 5
        true_class_in_top5 = sample_label in top5_indices[0].tolist()
        print(f"\nTrue class in top 5: {true_class_in_top5}")
        
        if true_class_in_top5:
            true_class_rank = (top5_indices[0] == sample_label).nonzero(as_tuple=True)[0].item() + 1
            print(f"True class rank: {true_class_rank}")
    else:
        print("No embedding available for sample image")


def evaluate_retrieval(model, dataloader, device, top_k=5):
    model.eval()
    all_embeddings = []
    all_paths = []
    with torch.no_grad():
        for _, file_paths, embeddings in dataloader:
            if embeddings is None:
                continue
            embeddings = embeddings.to(device)
            outputs = model(embeddings)
            all_embeddings.append(outputs.cpu())
            all_paths.extend(file_paths)
    all_embeddings = torch.cat(all_embeddings, dim=0)
    # Compute cosine similarity matrix
    similarity_matrix = torch.nn.functional.cosine_similarity(
        all_embeddings.unsqueeze(1), all_embeddings.unsqueeze(0), dim=-1
    )
    # For each image, retrieve top-K most similar images (excluding itself)
    for idx, path in enumerate(all_paths):
        sim_row = similarity_matrix[idx]
        sim_row[idx] = -float('inf')  # Exclude self
        topk_indices = torch.topk(sim_row, k=top_k).indices.tolist()
        print(f"Query: {path}")
        print("Top-{} retrieved: ".format(top_k))
        for rank, i in enumerate(topk_indices):
            print(f"  {rank+1}. {all_paths[i]}")
        print()