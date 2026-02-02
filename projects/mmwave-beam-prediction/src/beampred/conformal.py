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
            above = probs >= 1.0 - threshold

            for i in range(len(probs)):
                beam_set = np.where(above[i])[0]
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

            preds = np.argmax(probs, axis=1)
            base_scores = 1.0 - probs[np.arange(len(labels_np)), labels_np]
            gap_penalties = np.abs(labels_np.astype(int) - preds.astype(int)) / probs.shape[1]
            scores.append(base_scores + 0.5 * gap_penalties)

    scores = np.concatenate(scores)
    n = len(scores)
    q = np.ceil((n + 1) * (1 - alpha)) / n
    threshold = np.quantile(scores, min(q, 1.0))
    return threshold, scores


def predict_sets_beam_aware(model, data_loader, threshold, device="cpu"):
    model.eval()
    prediction_sets = []
    beam_indices = None

    with torch.no_grad():
        for features, labels in data_loader:
            features = features.to(device)
            logits = model(features)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            n_beams = probs.shape[1]

            if beam_indices is None:
                beam_indices = np.arange(n_beams)

            preds = np.argmax(probs, axis=1)
            base_scores = 1.0 - probs
            gap_penalties = np.abs(beam_indices[None, :] - preds[:, None]) / n_beams
            combined = base_scores + 0.5 * gap_penalties
            included = combined <= threshold

            for i in range(len(probs)):
                beam_set = np.where(included[i])[0]
                if len(beam_set) == 0:
                    beam_set = np.array([preds[i]])
                prediction_sets.append(beam_set)

    return prediction_sets


def set_sizes(prediction_sets):
    return np.array([len(s) for s in prediction_sets])


def coverage(prediction_sets, true_labels):
    true_labels = np.asarray(true_labels)
    covered = sum(1 for s, y in zip(prediction_sets, true_labels) if y in s)
    return covered / len(true_labels)
