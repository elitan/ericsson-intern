import numpy as np
import torch


def calibrate(model, cal_loader, alpha=0.1, device="cpu"):
    model.eval()
    scores = []

    with torch.no_grad():
        for features, labels in cal_loader:
            features, labels = features.to(device), labels.to(device)
            logits = model(features)
            probs = torch.softmax(logits, dim=1)
            true_probs = probs[torch.arange(len(labels)), labels]
            scores.append(1.0 - true_probs.cpu().numpy())

    scores = np.concatenate(scores)
    n = len(scores)
    q = np.ceil((n + 1) * (1 - alpha)) / n
    threshold = np.quantile(scores, min(q, 1.0))
    return threshold, scores


def predict_sets(model, data_loader, threshold, device="cpu"):
    model.eval()
    prediction_sets = []

    with torch.no_grad():
        for features, labels in data_loader:
            features = features.to(device)
            logits = model(features)
            probs = torch.softmax(logits, dim=1).cpu().numpy()

            for i in range(len(probs)):
                beam_set = np.where(probs[i] >= 1.0 - threshold)[0]
                if len(beam_set) == 0:
                    beam_set = np.array([np.argmax(probs[i])])
                prediction_sets.append(beam_set)

    return prediction_sets


def calibrate_beam_aware(model, cal_loader, alpha=0.1, device="cpu"):
    model.eval()
    scores = []

    with torch.no_grad():
        for features, labels in cal_loader:
            features, labels = features.to(device), labels.to(device)
            logits = model(features)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            labels_np = labels.cpu().numpy()

            for i in range(len(labels_np)):
                true_label = labels_np[i]
                pred = np.argmax(probs[i])
                base_score = 1.0 - probs[i, true_label]
                gap_penalty = np.abs(int(true_label) - int(pred)) / probs.shape[1]
                scores.append(base_score + 0.5 * gap_penalty)

    scores = np.array(scores)
    n = len(scores)
    q = np.ceil((n + 1) * (1 - alpha)) / n
    threshold = np.quantile(scores, min(q, 1.0))
    return threshold, scores


def predict_sets_beam_aware(model, data_loader, threshold, device="cpu"):
    model.eval()
    prediction_sets = []

    with torch.no_grad():
        for features, labels in data_loader:
            features = features.to(device)
            logits = model(features)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            n_beams = probs.shape[1]

            for i in range(len(probs)):
                pred = np.argmax(probs[i])
                beam_set = []
                for j in range(n_beams):
                    base_score = 1.0 - probs[i, j]
                    gap_penalty = np.abs(j - pred) / n_beams
                    if base_score + 0.5 * gap_penalty <= threshold:
                        beam_set.append(j)
                if len(beam_set) == 0:
                    beam_set = [pred]
                prediction_sets.append(np.array(beam_set))

    return prediction_sets


def set_sizes(prediction_sets):
    return np.array([len(s) for s in prediction_sets])


def coverage(prediction_sets, true_labels):
    true_labels = np.asarray(true_labels)
    covered = sum(1 for s, y in zip(prediction_sets, true_labels) if y in s)
    return covered / len(true_labels)
