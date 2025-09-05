"""
Minimal GEN baseline implementation
Just add this to your existing predict.py or run as standalone
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


def gen_score(logits, gamma=0.1):
    """
    Simple GEN score implementation
    Args:
        logits: classifier output logits
        gamma: GEN parameter (0.1 works well)
    Returns:
        OOD score (higher = more likely OOD)
    """
    probs = tf.nn.softmax(logits, axis=-1).numpy()
    if probs.ndim == 1:
        probs = probs.reshape(1, -1)

    # GEN formula: sum of p^gamma * (1-p)^gamma
    gen_entropy = np.sum(probs**gamma * (1 - probs) ** gamma, axis=1)
    return -gen_entropy[0] if len(gen_entropy) == 1 else -gen_entropy


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


def run_gen_baseline_simple():
    """
    Simple GEN baseline evaluation
    Uses your existing config.json
    """
    # Load your config
    with open("./config.json") as f:
        config = json.load(f)

    print("=" * 60)
    print(f"GEN BASELINE ON {config['dataset'].upper()}")
    print("=" * 60)

    # Load data (same as your predict.py)
    dataloader = DataLoader(path=os.path.join("dataset", config["dataset"]))
    train_sentences, train_intents = dataloader.train_loader()
    dev_sentences, dev_intents = dataloader.dev_loader()
    test_sentences, test_intents = dataloader.test_loader()
    ood_sentences, ood_intents = dataloader.ood_loader()

    # Load labels (same as your predict.py)
    in_lbl_2_indx = get_lbl_2_indx(
        path=os.path.join("dataset", config["dataset"], "in_lbl_2_indx.txt")
    )

    train_intents_encoded = one_hot_encoder(train_intents, in_lbl_2_indx)
    test_intents_encoded = one_hot_encoder(test_intents, in_lbl_2_indx)
    dev_intents_encoded = one_hot_encoder(dev_intents, in_lbl_2_indx)

    max_length = max_sentence_length(train_sentences, policy=config["seq_length"])

    # Load models (same as your predict.py)
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
        score = gen_score(logits, gamma=0.1)
        test_scores.append(score)

    for sentence in ood_sentences:
        inputs = __predict_preprocess__(sentence, tokenizer, max_length)
        logits = classifier.predict(inputs, verbose=0)[0]
        score = gen_score(logits, gamma=0.1)
        ood_scores.append(score)

    # Find threshold (95th percentile of test scores)
    threshold = np.percentile(test_scores, 95)
    print(f"GEN threshold: {threshold:.4f}")

    # Binary predictions (OOD detection)
    all_scores = np.array(test_scores + ood_scores)
    true_binary = [0] * len(test_scores) + [1] * len(ood_scores)
    pred_binary = (all_scores > threshold).astype(int)

    # Multiclass predictions (intent classification + OOD)
    ood_label = len(in_lbl_2_indx)
    all_predictions = []

    # Test samples
    for i, sentence in enumerate(test_sentences):
        if test_scores[i] <= threshold:
            # In-domain: use classifier
            inputs = __predict_preprocess__(sentence, tokenizer, max_length)
            logits = classifier.predict(inputs, verbose=0)[0]
            predicted_class = np.argmax(logits, axis=1)[0]
            all_predictions.append(predicted_class)
        else:
            # OOD
            all_predictions.append(ood_label)

    # OOD samples (all should be OOD)
    all_predictions.extend([ood_label] * len(ood_sentences))

    # True multiclass labels
    true_multiclass = [in_lbl_2_indx[intent] for intent in test_intents]
    true_multiclass.extend([ood_label] * len(ood_sentences))

    # Calculate metrics (same format as your paper)
    binary_f1_macro = f1_score(true_binary, pred_binary, average="macro")
    binary_f1_micro = f1_score(true_binary, pred_binary, average="micro")
    multi_f1_macro = f1_score(true_multiclass, all_predictions, average="macro")
    multi_f1_micro = f1_score(true_multiclass, all_predictions, average="micro")
    auc_roc = roc_auc_score(true_binary, all_scores)

    # Print results (same format as your predict.py)
    print("----------------------------------")
    print(f"Multi class macro f1: {multi_f1_macro:.4f}")
    print(f"Multi class micro f1: {multi_f1_micro:.4f}")
    print()
    print(f"Binary class macro f1: {binary_f1_macro:.4f}")
    print(f"Binary class micro f1: {binary_f1_micro:.4f}")
    print(f"AUC-ROC: {auc_roc:.4f}")
    print("------------------------------------------------------------------")

    # Save results for your paper
    results = {
        "dataset": config["dataset"],
        "method": "BERT + GEN",
        "binary_f1_macro": binary_f1_macro,
        "binary_f1_micro": binary_f1_micro,
        "multi_f1_macro": multi_f1_macro,
        "multi_f1_micro": multi_f1_micro,
        "auc_roc": auc_roc,
        "threshold": threshold,
    }

    # Save to artifacts directory
    output_dir = os.path.join("artifacts", config["dataset"])
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "gen_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {output_dir}/gen_results.json")

    return results


if __name__ == "__main__":
    # Just run this simple version
    results = run_gen_baseline_simple()
