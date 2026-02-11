import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix,
    roc_auc_score, balanced_accuracy_score
)
from sklearn.model_selection import cross_val_score, StratifiedKFold

from src.logger import get_logger
from src.visualizer import Visualizer

logger = get_logger(__name__)


class ModelEvaluator:
    def __init__(self, output_dir: str = "outputs"):
        self.visualizer = Visualizer(output_dir)
        self.output_dir = self.visualizer.output_dir
        logger.info("Model evaluator initialized")
    
    def evaluate_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: Optional[np.ndarray] = None,
        class_names: list = None
    ) -> Dict:
        
        if class_names is None:
            class_names = ['non-responder', 'partial', 'responder']
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=range(len(class_names))
        )
        
        # Weighted averages
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted'
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # ROC AUC (if probabilities available)
        roc_auc = None
        if y_pred_proba is not None:
            try:
                from sklearn.preprocessing import label_binarize
                y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
                roc_auc = roc_auc_score(y_true_bin, y_pred_proba, average='weighted', multi_class='ovr')
            except Exception as e:
                logger.warning(f"Could not calculate ROC AUC: {e}")
        
        # Build results dictionary
        results = {
            'accuracy': float(accuracy),
            'balanced_accuracy': float(balanced_acc),
            'precision_weighted': float(precision_weighted),
            'recall_weighted': float(recall_weighted),
            'f1_weighted': float(f1_weighted),
            'roc_auc_weighted': float(roc_auc) if roc_auc else None,
            'per_class_metrics': {
                class_names[i]: {
                    'precision': float(precision[i]),
                    'recall': float(recall[i]),
                    'f1_score': float(f1[i]),
                    'support': int(support[i])
                }
                for i in range(len(class_names))
            },
            'confusion_matrix': cm.tolist(),
            'classification_report': classification_report(
                y_true, y_pred, target_names=class_names, output_dict=True
            )
        }
        
        # Log key metrics
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Balanced Accuracy: {balanced_acc:.4f}")
        logger.info(f"F1 Score (weighted): {f1_weighted:.4f}")
        if roc_auc:
            logger.info(f"ROC AUC (weighted): {roc_auc:.4f}")
        
        # Generate visualizations
        self.visualizer.plot_confusion_matrix(
            y_true, y_pred, class_names=class_names,
            title="Confusion Matrix - Treatment Response Prediction"
        )
        
        if y_pred_proba is not None:
            self.visualizer.plot_roc_curves(
                y_true, y_pred_proba, class_names=class_names,
                title="ROC Curves - Treatment Response Prediction"
            )
        
        # Save results to file
        results_df = pd.DataFrame([results['per_class_metrics']]).T
        results_path = self.output_dir / "evaluation_results.csv"
        results_df.to_csv(results_path)
        logger.info(f"Evaluation results saved to {results_path}")
        
        return results
    
    def cross_validate_model(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5
    ) -> Dict:
        logger.info(f"Performing {cv}-fold cross-validation...")
        
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        
        # Calculate scores
        accuracy_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
        f1_scores = cross_val_score(model, X, y, cv=skf, scoring='f1_weighted')
        
        results = {
            'accuracy_mean': float(accuracy_scores.mean()),
            'accuracy_std': float(accuracy_scores.std()),
            'accuracy_scores': accuracy_scores.tolist(),
            'f1_mean': float(f1_scores.mean()),
            'f1_std': float(f1_scores.std()),
            'f1_scores': f1_scores.tolist()
        }
        
        logger.info(f"Cross-validation Accuracy: {results['accuracy_mean']:.4f} (+/- {results['accuracy_std']:.4f})")
        logger.info(f"Cross-validation F1: {results['f1_mean']:.4f} (+/- {results['f1_std']:.4f})")
        
        return results
    
    def analyze_feature_importance(
        self,
        model,
        feature_names: list,
        top_n: int = 20
    ):
        logger.info("Analyzing feature importance...")
        
        if not hasattr(model, 'feature_importances_'):
            logger.warning("Model does not have feature_importances_ attribute")
            return
        
        importance_scores = model.feature_importances_
        
        # Create DataFrame for easy analysis
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)
        
        # Save to file
        importance_path = self.output_dir / "feature_importance.csv"
        importance_df.to_csv(importance_path, index=False)
        logger.info(f"Feature importance saved to {importance_path}")
        
        # Visualize
        self.visualizer.plot_feature_importance(
            feature_names,
            importance_scores,
            top_n=top_n,
            title=f"Top {top_n} Most Important Features"
        )
        
        # Log top features
        logger.info(f"Top {min(10, len(importance_df))} most important features:")
        for idx, row in importance_df.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
    
    def evaluate_by_subgroups(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        patient_df: pd.DataFrame,
        groupby_col: str
    ) -> Dict:
        logger.info(f"Evaluating by subgroup: {groupby_col}")
        
        if groupby_col not in patient_df.columns:
            logger.error(f"Column {groupby_col} not found in patient data")
            return {}
        
        results = {}
        
        for group_value in patient_df[groupby_col].unique():
            mask = patient_df[groupby_col] == group_value
            group_y_true = y_true[mask]
            group_y_pred = y_pred[mask]
            
            if len(group_y_true) > 0:
                accuracy = accuracy_score(group_y_true, group_y_pred)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    group_y_true, group_y_pred, average='weighted', zero_division=0
                )
                
                results[str(group_value)] = {
                    'n_samples': int(len(group_y_true)),
                    'accuracy': float(accuracy),
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1_score': float(f1)
                }
                
                logger.info(f"  {group_value}: Accuracy={accuracy:.4f}, F1={f1:.4f}, N={len(group_y_true)}")
        
        # Save results
        results_df = pd.DataFrame(results).T
        results_path = self.output_dir / f"evaluation_by_{groupby_col}.csv"
        results_df.to_csv(results_path)
        logger.info(f"Subgroup evaluation saved to {results_path}")
        
        return results
    
    def generate_evaluation_report(
        self,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        patient_df: pd.DataFrame,
        feature_names: list,
        class_names: list = None
    ) -> Dict:
        logger.info("Generating evaluation report")
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        # Overall evaluation
        overall_results = self.evaluate_predictions(
            y_test, y_pred, y_pred_proba, class_names
        )
        
        # Feature importance
        self.analyze_feature_importance(model, feature_names)
        
        # Subgroup evaluations
        subgroup_results = {}
        for col in ['treatment_type', 'gender', 'baseline_severity']:
            if col in patient_df.columns:
                subgroup_results[col] = self.evaluate_by_subgroups(
                    y_test, y_pred, patient_df.iloc[len(patient_df)-len(y_test):], col
                )
        
        # Generate visualizations for patient data
        self.visualizer.plot_treatment_distribution(patient_df)
        self.visualizer.plot_clinical_scores(patient_df)
        self.visualizer.create_interactive_dashboard(patient_df)
        
        # Compile final report
        report = {
            'overall_metrics': overall_results,
            'subgroup_analysis': subgroup_results,
            'test_set_size': len(y_test),
            'class_distribution': pd.Series(y_test).value_counts().to_dict()
        }
        
        # Save complete report
        import json
        report_path = self.output_dir / "evaluation_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Complete evaluation report saved to {report_path}")
        
        return report
