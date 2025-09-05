"""
Comprehensive evaluation and analysis scripts for GEN baseline
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    auc,
    f1_score,
    roc_curve,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    precision_recall_fscore_support,
)
import tensorflow as tf
from sklearn.manifold import TSNE


class GENEvaluator:
    """Comprehensive evaluation for GEN baseline"""

    def __init__(self, config):
        self.config = config
        self.results_dir = os.path.join("artifacts", config["dataset"], "gen_baseline")
        os.makedirs(self.results_dir, exist_ok=True)

    def detailed_evaluation(
        self, gen_baseline, test_sentences, test_intents, ood_sentences, ood_intents
    ):
        """
        Perform detailed evaluation with multiple metrics and visualizations
        """
        print("Performing detailed evaluation...")

        # Prepare data
        all_sentences = test_sentences + ood_sentences
        y_true_multiclass = []

        # True multiclass labels
        for intent in test_intents:
            y_true_multiclass.append(gen_baseline.in_lbl_2_indx[intent])
        for _ in ood_intents:
            y_true_multiclass.append(gen_baseline.ood_class_id)

        y_true_binary = [0] * len(test_sentences) + [1] * len(ood_sentences)

        # Get predictions and probabilities
        y_pred_multiclass = []
        y_pred_probs = []
        prediction_confidences = []

        for sentence in all_sentences:
            inputs = gen_baseline._preprocess_sentence(sentence)
            logits = gen_baseline.classifier.predict(inputs, verbose=0)[0][0]
            probs = tf.nn.softmax(logits, axis=0).numpy()

            predicted_class = np.argmax(probs)
            max_confidence = np.max(probs)

            y_pred_multiclass.append(predicted_class)
            y_pred_probs.append(probs)
            prediction_confidences.append(max_confidence)

        # Binary predictions
        y_pred_binary = [
            1 if pred == gen_baseline.ood_class_id else 0 for pred in y_pred_multiclass
        ]

        # Calculate comprehensive metrics
        results = self._calculate_comprehensive_metrics(
            y_true_binary,
            y_pred_binary,
            y_true_multiclass,
            y_pred_multiclass,
            y_pred_probs,
            prediction_confidences,
            gen_baseline.ood_class_id,
        )

        # Create visualizations
        self._create_visualizations(
            y_true_binary,
            y_pred_binary,
            y_true_multiclass,
            y_pred_multiclass,
            y_pred_probs,
            prediction_confidences,
            gen_baseline,
        )

        # Generate detailed report
        self._generate_detailed_report(results, gen_baseline)

        return results

    def _calculate_comprehensive_metrics(
        self,
        y_true_binary,
        y_pred_binary,
        y_true_multiclass,
        y_pred_multiclass,
        y_pred_probs,
        prediction_confidences,
        ood_class_id,
    ):
        """Calculate comprehensive metrics"""

        # Binary classification metrics
        binary_precision, binary_recall, binary_f1, _ = precision_recall_fscore_support(
            y_true_binary, y_pred_binary, average=None
        )

        # Multiclass metrics
        multiclass_precision, multiclass_recall, multiclass_f1, _ = (
            precision_recall_fscore_support(
                y_true_multiclass, y_pred_multiclass, average=None, zero_division=0
            )
        )

        # ROC and PR curves
        ood_probs = [probs[ood_class_id] for probs in y_pred_probs]
        fpr, tpr, roc_thresholds = roc_curve(y_true_binary, ood_probs)
        roc_auc = auc(fpr, tpr)

        precision_curve, recall_curve, pr_thresholds = precision_recall_curve(
            y_true_binary, ood_probs
        )
        pr_auc = auc(recall_curve, precision_curve)

        # Confidence analysis
        in_domain_confidences = prediction_confidences[
            : len(y_true_binary) - sum(y_true_binary)
        ]
        ood_confidences = prediction_confidences[
            len(y_true_binary) - sum(y_true_binary) :
        ]

        results = {
            "binary": {
                "f1_macro": f1_score(y_true_binary, y_pred_binary, average="macro"),
                "f1_micro": f1_score(y_true_binary, y_pred_binary, average="micro"),
                "precision_per_class": binary_precision.tolist(),
                "recall_per_class": binary_recall.tolist(),
                "f1_per_class": binary_f1.tolist(),
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
            },
            "multiclass": {
                "f1_macro": f1_score(
                    y_true_multiclass, y_pred_multiclass, average="macro"
                ),
                "f1_micro": f1_score(
                    y_true_multiclass, y_pred_multiclass, average="micro"
                ),
                "precision_per_class": multiclass_precision.tolist(),
                "recall_per_class": multiclass_recall.tolist(),
                "f1_per_class": multiclass_f1.tolist(),
            },
            "confidence_analysis": {
                "in_domain_mean_confidence": np.mean(in_domain_confidences),
                "in_domain_std_confidence": np.std(in_domain_confidences),
                "ood_mean_confidence": np.mean(ood_confidences),
                "ood_std_confidence": np.std(ood_confidences),
                "confidence_separation": np.mean(in_domain_confidences)
                - np.mean(ood_confidences),
            },
            "curves": {
                "roc": {
                    "fpr": fpr.tolist(),
                    "tpr": tpr.tolist(),
                    "thresholds": roc_thresholds.tolist(),
                },
                "pr": {
                    "precision": precision_curve.tolist(),
                    "recall": recall_curve.tolist(),
                    "thresholds": pr_thresholds.tolist(),
                },
            },
        }

        return results

    def _create_visualizations(
        self,
        y_true_binary,
        y_pred_binary,
        y_true_multiclass,
        y_pred_multiclass,
        y_pred_probs,
        prediction_confidences,
        gen_baseline,
    ):
        """Create comprehensive visualizations"""

        # 1. Confusion matrices
        self._plot_confusion_matrices(
            y_true_binary, y_pred_binary, y_true_multiclass, y_pred_multiclass
        )

        # 2. ROC and PR curves
        self._plot_roc_pr_curves(y_true_binary, y_pred_probs, gen_baseline.ood_class_id)

        # 3. Confidence distributions
        self._plot_confidence_distributions(y_true_binary, prediction_confidences)

        # 4. Class-wise performance
        self._plot_class_performance(
            y_true_multiclass, y_pred_multiclass, gen_baseline.in_lbl_2_indx
        )

        # 5. Generation method comparison (if multiple methods were tested)
        self._plot_generation_comparison()

    def _plot_confusion_matrices(
        self, y_true_binary, y_pred_binary, y_true_multiclass, y_pred_multiclass
    ):
        """Plot confusion matrices for binary and multiclass classification"""

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Binary confusion matrix
        cm_binary = confusion_matrix(y_true_binary, y_pred_binary)
        sns.heatmap(
            cm_binary,
            annot=True,
            fmt="d",
            ax=ax1,
            xticklabels=["In-Domain", "OOD"],
            yticklabels=["In-Domain", "OOD"],
        )
        ax1.set_title("Binary Classification Confusion Matrix")
        ax1.set_ylabel("True Label")
        ax1.set_xlabel("Predicted Label")

        # Multiclass confusion matrix
        cm_multiclass = confusion_matrix(y_true_multiclass, y_pred_multiclass)
        sns.heatmap(cm_multiclass, annot=True, fmt="d", ax=ax2)
        ax2.set_title("Multiclass Classification Confusion Matrix")
        ax2.set_ylabel("True Label")
        ax2.set_xlabel("Predicted Label")

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.results_dir, "confusion_matrices.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _plot_roc_pr_curves(self, y_true_binary, y_pred_probs, ood_class_id):
        """Plot ROC and Precision-Recall curves"""

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Extract OOD probabilities
        ood_probs = [probs[ood_class_id] for probs in y_pred_probs]

        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true_binary, ood_probs)
        roc_auc = auc(fpr, tpr)

        ax1.plot(
            fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})"
        )
        ax1.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel("False Positive Rate")
        ax1.set_ylabel("True Positive Rate")
        ax1.set_title("ROC Curve")
        ax1.legend(loc="lower right")
        ax1.grid(True, alpha=0.3)

        # Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true_binary, ood_probs)
        pr_auc = auc(recall, precision)

        ax2.plot(
            recall,
            precision,
            color="darkorange",
            lw=2,
            label=f"PR curve (AUC = {pr_auc:.3f})",
        )
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel("Recall")
        ax2.set_ylabel("Precision")
        ax2.set_title("Precision-Recall Curve")
        ax2.legend(loc="lower left")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.results_dir, "roc_pr_curves.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _plot_confidence_distributions(self, y_true_binary, prediction_confidences):
        """Plot confidence score distributions"""

        # Separate confidences by true label
        in_domain_confidences = [
            conf
            for i, conf in enumerate(prediction_confidences)
            if y_true_binary[i] == 0
        ]
        ood_confidences = [
            conf
            for i, conf in enumerate(prediction_confidences)
            if y_true_binary[i] == 1
        ]

        plt.figure(figsize=(12, 6))

        # Plot histograms
        plt.hist(
            in_domain_confidences,
            bins=30,
            alpha=0.7,
            label="In-Domain",
            color="blue",
            density=True,
        )
        plt.hist(
            ood_confidences, bins=30, alpha=0.7, label="OOD", color="red", density=True
        )

        # Add mean lines
        plt.axvline(
            np.mean(in_domain_confidences),
            color="blue",
            linestyle="--",
            label=f"In-Domain Mean: {np.mean(in_domain_confidences):.3f}",
        )
        plt.axvline(
            np.mean(ood_confidences),
            color="red",
            linestyle="--",
            label=f"OOD Mean: {np.mean(ood_confidences):.3f}",
        )

        plt.xlabel("Prediction Confidence")
        plt.ylabel("Density")
        plt.title("Distribution of Prediction Confidences")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.savefig(
            os.path.join(self.results_dir, "confidence_distributions.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _plot_class_performance(
        self, y_true_multiclass, y_pred_multiclass, in_lbl_2_indx
    ):
        """Plot per-class performance metrics"""

        # Calculate per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true_multiclass, y_pred_multiclass, average=None, zero_division=0
        )

        # Create class names
        class_names = list(in_lbl_2_indx.keys()) + ["OOD"]

        # Create DataFrame for easier plotting
        df = pd.DataFrame(
            {
                "Class": class_names,
                "Precision": precision,
                "Recall": recall,
                "F1-Score": f1,
                "Support": support,
            }
        )

        # Plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Precision
        axes[0, 0].bar(range(len(class_names)), precision, color="skyblue")
        axes[0, 0].set_title("Precision by Class")
        axes[0, 0].set_ylabel("Precision")
        axes[0, 0].set_xticks(range(len(class_names)))
        axes[0, 0].set_xticklabels(class_names, rotation=45)
        axes[0, 0].grid(True, alpha=0.3)

        # Recall
        axes[0, 1].bar(range(len(class_names)), recall, color="lightcoral")
        axes[0, 1].set_title("Recall by Class")
        axes[0, 1].set_ylabel("Recall")
        axes[0, 1].set_xticks(range(len(class_names)))
        axes[0, 1].set_xticklabels(class_names, rotation=45)
        axes[0, 1].grid(True, alpha=0.3)

        # F1-Score
        axes[1, 0].bar(range(len(class_names)), f1, color="lightgreen")
        axes[1, 0].set_title("F1-Score by Class")
        axes[1, 0].set_ylabel("F1-Score")
        axes[1, 0].set_xticks(range(len(class_names)))
        axes[1, 0].set_xticklabels(class_names, rotation=45)
        axes[1, 0].grid(True, alpha=0.3)

        # Support
        axes[1, 1].bar(range(len(class_names)), support, color="gold")
        axes[1, 1].set_title("Support by Class")
        axes[1, 1].set_ylabel("Number of Samples")
        axes[1, 1].set_xticks(range(len(class_names)))
        axes[1, 1].set_xticklabels(class_names, rotation=45)
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.results_dir, "class_performance.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # Save numerical results
        df.to_csv(os.path.join(self.results_dir, "class_performance.csv"), index=False)

    def _plot_generation_comparison(self):
        """Plot comparison of different generation methods if available"""

        # Check if multiple results files exist
        results_files = [
            f
            for f in os.listdir(self.results_dir)
            if f.startswith("results_") and f.endswith(".json")
        ]

        if len(results_files) < 2:
            return

        # Load results from different methods
        methods_data = {}
        for file in results_files:
            method_name = file.replace("results_", "").replace(".json", "")
            with open(os.path.join(self.results_dir, file), "r") as f:
                methods_data[method_name] = json.load(f)

        # Create comparison plot
        metrics = ["f1_macro", "f1_micro", "roc_auc"]
        methods = list(methods_data.keys())

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        for i, metric in enumerate(metrics):
            values = [methods_data[method]["binary"][metric] for method in methods]
            axes[i].bar(methods, values, color="steelblue")
            axes[i].set_title(f'Binary {metric.replace("_", " ").title()}')
            axes[i].set_ylabel(metric.replace("_", " ").title())
            axes[i].set_ylim(0, 1)
            axes[i].grid(True, alpha=0.3)

            # Add value labels on bars
            for j, v in enumerate(values):
                axes[i].text(j, v + 0.01, f"{v:.3f}", ha="center", va="bottom")

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.results_dir, "generation_method_comparison.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _generate_detailed_report(self, results, gen_baseline):
        """Generate a detailed text report"""

        report_path = os.path.join(self.results_dir, "detailed_report.txt")

        with open(report_path, "w") as f:
            f.write("GEN BASELINE - DETAILED EVALUATION REPORT\n")
            f.write("=" * 60 + "\n\n")

            # Dataset information
            f.write(f"Dataset: {self.config['dataset']}\n")
            f.write(f"BERT Model: {self.config['bert']}\n")
            f.write(
                f"Generation Method: {self.config.get('gen_method', 'unknown')}\n\n"
            )

            # Binary Classification Results
            f.write("BINARY CLASSIFICATION (In-Domain vs OOD)\n")
            f.write("-" * 40 + "\n")
            f.write(f"F1 Macro Score: {results['binary']['f1_macro']:.4f}\n")
            f.write(f"F1 Micro Score: {results['binary']['f1_micro']:.4f}\n")
            f.write(f"ROC-AUC Score: {results['binary']['roc_auc']:.4f}\n")
            f.write(f"PR-AUC Score: {results['binary']['pr_auc']:.4f}\n\n")

            # Per-class binary results
            f.write("Per-Class Binary Results:\n")
            class_names = ["In-Domain", "OOD"]
            for i, name in enumerate(class_names):
                f.write(f"  {name}:\n")
                f.write(
                    f"    Precision: {results['binary']['precision_per_class'][i]:.4f}\n"
                )
                f.write(f"    Recall: {results['binary']['recall_per_class'][i]:.4f}\n")
                f.write(f"    F1-Score: {results['binary']['f1_per_class'][i]:.4f}\n")
            f.write("\n")

            # Multiclass Classification Results
            f.write("MULTICLASS CLASSIFICATION (Intent + OOD)\n")
            f.write("-" * 40 + "\n")
            f.write(f"F1 Macro Score: {results['multiclass']['f1_macro']:.4f}\n")
            f.write(f"F1 Micro Score: {results['multiclass']['f1_micro']:.4f}\n\n")

            # Confidence Analysis
            f.write("CONFIDENCE ANALYSIS\n")
            f.write("-" * 40 + "\n")
            conf_analysis = results["confidence_analysis"]
            f.write(
                f"In-Domain Mean Confidence: {conf_analysis['in_domain_mean_confidence']:.4f} ± {conf_analysis['in_domain_std_confidence']:.4f}\n"
            )
            f.write(
                f"OOD Mean Confidence: {conf_analysis['ood_mean_confidence']:.4f} ± {conf_analysis['ood_std_confidence']:.4f}\n"
            )
            f.write(
                f"Confidence Separation: {conf_analysis['confidence_separation']:.4f}\n\n"
            )

            # Comparison with paper results (if available)
            f.write("COMPARISON WITH EXISTING METHODS\n")
            f.write("-" * 40 + "\n")

            # Add comparison with paper's results for SNIPS dataset
            if self.config["dataset"].lower() == "snips":
                paper_results = {
                    "BERT + AET (Paper)": {"binary_f1": 95.61, "multiclass_f1": 92.03},
                    "BERT + VAE (Paper)": {"binary_f1": 92.32, "multiclass_f1": 89.58},
                }

                f.write("Comparison with paper results on SNIPS dataset:\n")
                for method, scores in paper_results.items():
                    f.write(f"  {method}:\n")
                    f.write(f"    Binary F1: {scores['binary_f1']:.2f}%\n")
                    f.write(f"    Multiclass F1: {scores['multiclass_f1']:.2f}%\n")

                f.write(f"\n  GEN Baseline (This work):\n")
                f.write(f"    Binary F1: {results['binary']['f1_macro']*100:.2f}%\n")
                f.write(
                    f"    Multiclass F1: {results['multiclass']['f1_macro']*100:.2f}%\n\n"
                )

                # Performance gap analysis
                best_paper_binary = 95.61
                best_paper_multiclass = 92.03
                gen_binary = results["binary"]["f1_macro"] * 100
                gen_multiclass = results["multiclass"]["f1_macro"] * 100

                f.write("Performance Gap Analysis:\n")
                f.write(
                    f"  Binary F1 Gap: {gen_binary - best_paper_binary:.2f} percentage points\n"
                )
                f.write(
                    f"  Multiclass F1 Gap: {gen_multiclass - best_paper_multiclass:.2f} percentage points\n\n"
                )

            # Recommendations
            f.write("RECOMMENDATIONS FOR IMPROVEMENT\n")
            f.write("-" * 40 + "\n")
            f.write("1. Experiment with different negative generation strategies\n")
            f.write("2. Adjust the ratio of generated negatives to original data\n")
            f.write(
                "3. Use more sophisticated text generation techniques (e.g., GPT-based)\n"
            )
            f.write("4. Implement curriculum learning with generated examples\n")
            f.write(
                "5. Combine GEN with other OOD detection methods (ensemble approach)\n"
            )

    def compare_with_baselines(self, gen_results, baseline_results=None):
        """Compare GEN results with other baseline methods"""

        if baseline_results is None:
            # Default baseline results from the paper
            baseline_results = {
                "BERT": {"binary_f1": 0.5198, "multiclass_f1": 0.2002},
                "BERT + LMCL": {"binary_f1": 0.5291, "multiclass_f1": 0.2071},
                "BERT + DOC": {"binary_f1": 0.6375, "multiclass_f1": 0.2854},
                "BERT + ADB": {"binary_f1": 0.6279, "multiclass_f1": 0.7193},
                "BERT + VAE": {"binary_f1": 0.9232, "multiclass_f1": 0.8958},
                "BERT + AET": {"binary_f1": 0.9561, "multiclass_f1": 0.9203},
            }

        # Add GEN results
        comparison_data = baseline_results.copy()
        comparison_data["GEN (This work)"] = {
            "binary_f1": gen_results["binary"]["f1_macro"],
            "multiclass_f1": gen_results["multiclass"]["f1_macro"],
        }

        # Create comparison visualization
        methods = list(comparison_data.keys())
        binary_f1s = [comparison_data[method]["binary_f1"] for method in methods]
        multiclass_f1s = [
            comparison_data[method]["multiclass_f1"] for method in methods
        ]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Binary F1 comparison
        colors = ["red" if "GEN" in method else "steelblue" for method in methods]
        bars1 = ax1.bar(range(len(methods)), binary_f1s, color=colors)
        ax1.set_title("Binary F1 Score Comparison")
        ax1.set_ylabel("F1 Score")
        ax1.set_xticks(range(len(methods)))
        ax1.set_xticklabels(methods, rotation=45, ha="right")
        ax1.grid(True, alpha=0.3)

        # Add value labels
        for bar, value in zip(bars1, binary_f1s):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        # Multiclass F1 comparison
        bars2 = ax2.bar(range(len(methods)), multiclass_f1s, color=colors)
        ax2.set_title("Multiclass F1 Score Comparison")
        ax2.set_ylabel("F1 Score")
        ax2.set_xticks(range(len(methods)))
        ax2.set_xticklabels(methods, rotation=45, ha="right")
        ax2.grid(True, alpha=0.3)

        # Add value labels
        for bar, value in zip(bars2, multiclass_f1s):
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.results_dir, "baseline_comparison.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # Save comparison data
        comparison_df = pd.DataFrame(comparison_data).T
        comparison_df.to_csv(os.path.join(self.results_dir, "baseline_comparison.csv"))

        return comparison_data


def analyze_generated_examples(gen_baseline, generated_sentences, original_sentences):
    """Analyze the quality and characteristics of generated examples"""

    analysis_results = {
        "statistics": {},
        "quality_metrics": {},
        "similarity_analysis": {},
    }

    # Basic statistics
    analysis_results["statistics"] = {
        "num_generated": len(generated_sentences),
        "avg_length_generated": np.mean([len(s.split()) for s in generated_sentences]),
        "avg_length_original": np.mean([len(s.split()) for s in original_sentences]),
        "length_ratio": np.mean([len(s.split()) for s in generated_sentences])
        / np.mean([len(s.split()) for s in original_sentences]),
    }

    # Vocabulary overlap analysis
    original_vocab = set()
    generated_vocab = set()

    for sentence in original_sentences:
        original_vocab.update(sentence.lower().split())

    for sentence in generated_sentences:
        generated_vocab.update(sentence.lower().split())

    vocab_overlap = len(original_vocab.intersection(generated_vocab))
    vocab_union = len(original_vocab.union(generated_vocab))

    analysis_results["quality_metrics"] = {
        "vocab_overlap_ratio": vocab_overlap / len(original_vocab),
        "vocab_diversity": len(generated_vocab) / len(original_vocab),
        "jaccard_similarity": vocab_overlap / vocab_union,
    }

    print("Generated Examples Analysis:")
    print(
        f"  Number of generated examples: {analysis_results['statistics']['num_generated']}"
    )
    print(
        f"  Average length ratio: {analysis_results['statistics']['length_ratio']:.2f}"
    )
    print(
        f"  Vocabulary overlap ratio: {analysis_results['quality_metrics']['vocab_overlap_ratio']:.3f}"
    )
    print(
        f"  Vocabulary diversity: {analysis_results['quality_metrics']['vocab_diversity']:.3f}"
    )

    return analysis_results


def run_comprehensive_gen_evaluation(config_path="./gen_config.json"):
    """Run comprehensive evaluation of GEN baseline"""

    # Load config
    with open(config_path) as f:
        config = json.load(f)

    # Import and run GEN baseline
    from gen_baseline import GENBaseline

    # Create and run GEN baseline
    gen_baseline = GENBaseline(config)
    results = gen_baseline.run_full_pipeline(
        generation_method=config.get("gen_method", "random_token_replacement")
    )

    # Create evaluator and run detailed evaluation
    evaluator = GENEvaluator(config)

    detailed_results = evaluator.detailed_evaluation(
        gen_baseline,
        gen_baseline.test_sentences,
        gen_baseline.test_intents,
        gen_baseline.ood_sentences,
        gen_baseline.ood_intents,
    )

    # Compare with baselines
    comparison_results = evaluator.compare_with_baselines(detailed_results)

    print("\nGEN Baseline evaluation completed!")
    print(f"Results saved to: {evaluator.results_dir}")

    return detailed_results, comparison_results


if __name__ == "__main__":
    # Run comprehensive evaluation
    detailed_results, comparison_results = run_comprehensive_gen_evaluation()
