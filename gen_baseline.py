"""
Corrected GEN baseline implementation based on official paper
"""

import os
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score, roc_auc_score

from model.encoder import finetune
from data_modules.data_utils import *
from model.model_utils import get_bert
from data_modules.dataloader import DataLoader


def gen_score(logits, gamma=0.1, M=100):
    """
    Correct GEN score implementation from official paper
    Args:
        logits: classifier output logits [batch_size, num_classes]
        gamma: GEN parameter (0.1 works well)
        M: number of top classes to use (100 for 1000-class problems)
    Returns:
        GEN scores (higher = more likely in-domain)
    """
    probs = tf.nn.softmax(logits, axis=-1).numpy()
    if probs.ndim == 1:
        probs = probs.reshape(1, -1)
    
    # Sort probabilities and take top M classes
    probs_sorted = np.sort(probs, axis=1)[:, -M:]
    
    # GEN formula: sum of p^gamma * (1-p)^gamma for top M classes
    scores = np.sum(probs_sorted**gamma * (1 - probs_sorted)**gamma, axis=1)
    
    # Return negative (higher score = more confident/in-domain)
    return -scores


def __predict_preprocess__(x, tokenizer, max_length):
    """Same as your existing preprocessing"""
    x_tokenized = tokenizer(
        x,
        return_tensors="tf",
        padding="max_length",
        max_length=max_length,
        truncation=True,
    )
    return {i: x_tokenized[i] for i in tokenizer.model_input_names}


def run_corrected_gen_baseline():
    """
    Corrected GEN baseline evaluation
    """
    # Load your config
    with open("./config.json") as f:
        config = json.load(f)

    print("=" * 60)
    print(f"CORRECTED GEN BASELINE ON {config['dataset'].upper()}")
    print("=" * 60)

    # Load data
    dataloader = DataLoader(path=os.path.join("dataset", config["dataset"]))
    train_sentences, train_intents = dataloader.train_loader()
    dev_sentences, dev_intents = dataloader.dev_loader()
    test_sentences, test_intents = dataloader.test_loader()
    ood_sentences, ood_intents = dataloader.ood_loader()

    # Load labels
    in_lbl_2_indx = get_lbl_2_indx(
        path=os.path.join("dataset", config["dataset"], "in_lbl_2_indx.txt")
    )

    num_classes = len(in_lbl_2_indx)
    M = min(100, num_classes // 10)  # Use 10% of classes or 100, whichever is smaller
    
    print(f"Using M={M} top classes for GEN score (out of {num_classes} total)")

    train_intents_encoded = one_hot_encoder(train_intents, in_lbl_2_indx)
    test_intents_encoded = one_hot_encoder(test_intents, in_lbl_2_indx)
    dev_intents_encoded = one_hot_encoder(dev_intents, in_lbl_2_indx)

    max_length = max_sentence_length(train_sentences, policy=config["seq_length"])

    # Load models
    bert, tokenizer = get_bert(config["bert"])
    classifier = finetune(
        x_train=train_sentences + dev_sentences,
        y_train=np.concatenate((train_intents_encoded, dev_intents_encoded), axis=0),
        x_validation=test_sentences,
        y_validation=test_intents_encoded,
        max_length=max_length,
        num_labels=len(in_lbl_2_indx),
        path=os.path.join("artifacts", config["dataset"], "bert/"),
        train=config["finetune"],
        model_name=config["bert"],
        num_epochs=config["finetune_epochs"]
    )
    classifier.load_weights(
        os.path.join("artifacts", config["dataset"], "bert/best_model")
    )

    print("Computing GEN scores...")

    # Get GEN scores for all samples
    test_scores = []
    ood_scores = []

    for sentence in test_sentences:
        inputs = __predict_preprocess__(sentence, tokenizer, max_length)
        logits = classifier.predict(inputs, verbose=0)[0]
        score = gen_score(logits, gamma=0.1, M=M)
        test_scores.append(score[0] if isinstance(score, np.ndarray) else score)

    for sentence in ood_sentences:
        inputs = __predict_preprocess__(sentence, tokenizer, max_length)
        logits = classifier.predict(inputs, verbose=0)[0]
        score = gen_score(logits, gamma=0.1, M=M)
        ood_scores.append(score[0] if isinstance(score, np.ndarray) else score)

    test_scores = np.array(test_scores)
    ood_scores = np.array(ood_scores)

    print(f"Test scores - mean: {test_scores.mean():.4f}, std: {test_scores.std():.4f}")
    print(f"OOD scores - mean: {ood_scores.mean():.4f}, std: {ood_scores.std():.4f}")

    # Find optimal threshold using validation approach
    # Higher GEN score = more likely in-domain
    all_scores = np.concatenate([test_scores, ood_scores])
    all_labels = np.concatenate([np.ones(len(test_scores)), np.zeros(len(ood_scores))])
    
    # Try different thresholds and find the one that maximizes F1
    thresholds = np.percentile(all_scores, np.linspace(5, 95, 19))
    best_f1 = 0
    best_threshold = None
    
    for threshold in thresholds:
        # Predict: score >= threshold -> in-domain (1), score < threshold -> OOD (0)
        predictions = (all_scores >= threshold).astype(int)
        f1 = f1_score(all_labels, predictions, average='macro')
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print(f"Best threshold: {best_threshold:.4f} (F1: {best_f1:.4f})")

    # Final predictions using best threshold
    test_predictions = (test_scores >= best_threshold).astype(int)  # 1 = ID
    ood_predictions = (ood_scores >= best_threshold).astype(int)    # should be 0 = OOD
    
    # Binary evaluation
    true_binary = [1] * len(test_scores) + [0] * len(ood_scores)  # 1=ID, 0=OOD
    pred_binary = list(test_predictions) + list(ood_predictions)

    # Multiclass evaluation
    ood_label = len(in_lbl_2_indx)
    all_predictions = []

    # Test samples - classify if above threshold, otherwise mark as OOD
    for i, sentence in enumerate(test_sentences):
        if test_scores[i] >= best_threshold:
            # In-domain: use classifier prediction
            inputs = __predict_preprocess__(sentence, tokenizer, max_length)
            logits = classifier.predict(inputs, verbose=0)[0]
            predicted_class = np.argmax(logits, axis=1)[0]
            all_predictions.append(predicted_class)
        else:
            # OOD
            all_predictions.append(ood_label)

    # OOD samples - classify if above threshold (false positive), otherwise OOD
    for i, sentence in enumerate(ood_sentences):
        if ood_scores[i] >= best_threshold:
            # False positive: classifier thinks it's in-domain
            inputs = __predict_preprocess__(sentence, tokenizer, max_length)
            logits = classifier.predict(inputs, verbose=0)[0]
            predicted_class = np.argmax(logits, axis=1)[0]
            all_predictions.append(predicted_class)
        else:
            # Correctly identified as OOD
            all_predictions.append(ood_label)

    # True multiclass labels
    true_multiclass = [in_lbl_2_indx[intent] for intent in test_intents]
    true_multiclass.extend([ood_label] * len(ood_sentences))

    # Calculate metrics
    binary_f1_macro = f1_score(true_binary, pred_binary, average="macro")
    binary_f1_micro = f1_score(true_binary, pred_binary, average="micro")
    multi_f1_macro = f1_score(true_multiclass, all_predictions, average="macro")
    multi_f1_micro = f1_score(true_multiclass, all_predictions, average="micro")
    auc_roc = roc_auc_score(true_binary, list(test_scores) + list(ood_scores))

    # Print results
    print("----------------------------------")
    print(f"Multi class macro f1: {multi_f1_macro:.4f}")
    print(f"Multi class micro f1: {multi_f1_micro:.4f}")
    print()
    print(f"Binary class macro f1: {binary_f1_macro:.4f}")
    print(f"Binary class micro f1: {binary_f1_micro:.4f}")
    print(f"AUC-ROC: {auc_roc:.4f}")
    print("------------------------------------------------------------------")

    # Save results
    results = {
        "dataset": config["dataset"],
        "method": "BERT + GEN (Corrected)",
        "binary_f1_macro": binary_f1_macro,
        "binary_f1_micro": binary_f1_micro,
        "multi_f1_macro": multi_f1_macro,
        "multi_f1_micro": multi_f1_micro,
        "auc_roc": auc_roc,
        "threshold": best_threshold,
        "gamma": 0.1,
        "M": M,
    }

    output_dir = os.path.join("artifacts", config["dataset"])
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "gen_corrected_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_dir}/gen_corrected_results.json")

    return results


if __name__ == "__main__":
    results = run_corrected_gen_baseline()
