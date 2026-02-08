"""
Visualization Tools
Generate plots and charts for model evaluation and insights
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

from src.logger import get_logger

logger = get_logger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class Visualizer:
    """Create visualizations for model analysis"""
    
    def __init__(self, output_dir: str = "outputs"):
        """
        Initialize visualizer
        
        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        logger.info(f"Visualizer initialized. Output directory: {self.output_dir}")
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: List[str] = None,
        title: str = "Confusion Matrix",
        save_name: str = "confusion_matrix.png"
    ):
        """
        Plot confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Names of classes
            title: Plot title
            save_name: Filename to save plot
        """
        logger.info(f"Plotting confusion matrix: {title}")
        
        if class_names is None:
            class_names = ['non-responder', 'partial', 'responder']
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={'label': 'Count'}
        )
        plt.title(title, fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {save_path}")
        plt.close()
    
    def plot_feature_importance(
        self,
        feature_names: List[str],
        importance_scores: np.ndarray,
        top_n: int = 20,
        title: str = "Feature Importance",
        save_name: str = "feature_importance.png"
    ):
        """
        Plot feature importance
        
        Args:
            feature_names: Names of features
            importance_scores: Importance scores
            top_n: Number of top features to display
            title: Plot title
            save_name: Filename to save plot
        """
        logger.info(f"Plotting feature importance: {title}")
        
        # Get top N features
        indices = np.argsort(importance_scores)[-top_n:]
        top_features = [feature_names[i] for i in indices]
        top_scores = importance_scores[indices]
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(top_features)), top_scores, color='steelblue')
        plt.yticks(range(len(top_features)), top_features)
        plt.xlabel('Importance Score', fontsize=12)
        plt.title(title, fontsize=16, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Feature importance plot saved to {save_path}")
        plt.close()
    
    def plot_roc_curves(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        class_names: List[str] = None,
        title: str = "ROC Curves",
        save_name: str = "roc_curves.png"
    ):
        """
        Plot ROC curves for multi-class classification
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            class_names: Names of classes
            title: Plot title
            save_name: Filename to save plot
        """
        logger.info(f"Plotting ROC curves: {title}")
        
        if class_names is None:
            class_names = ['non-responder', 'partial', 'responder']
        
        n_classes = len(class_names)
        
        # Binarize the labels
        y_true_bin = label_binarize(y_true, classes=range(n_classes))
        
        plt.figure(figsize=(10, 8))
        
        colors = ['blue', 'green', 'red']
        for i, (class_name, color) in enumerate(zip(class_names, colors)):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            
            plt.plot(
                fpr, tpr,
                color=color,
                lw=2,
                label=f'{class_name} (AUC = {roc_auc:.2f})'
            )
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(title, fontsize=16, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"ROC curves saved to {save_path}")
        plt.close()
    
    def plot_treatment_distribution(
        self,
        patient_df: pd.DataFrame,
        save_name: str = "treatment_distribution.png"
    ):
        """
        Plot distribution of treatment types and outcomes
        
        Args:
            patient_df: Patient profiles DataFrame
            save_name: Filename to save plot
        """
        logger.info("Plotting treatment distribution")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Treatment type distribution
        treatment_counts = patient_df['treatment_type'].value_counts()
        axes[0, 0].bar(range(len(treatment_counts)), treatment_counts.values, color='steelblue')
        axes[0, 0].set_xticks(range(len(treatment_counts)))
        axes[0, 0].set_xticklabels(treatment_counts.index, rotation=45, ha='right')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Treatment Type Distribution', fontweight='bold')
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # Response distribution
        response_counts = patient_df['treatment_response'].value_counts()
        axes[0, 1].bar(range(len(response_counts)), response_counts.values, color='forestgreen')
        axes[0, 1].set_xticks(range(len(response_counts)))
        axes[0, 1].set_xticklabels(response_counts.index, rotation=45, ha='right')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].set_title('Treatment Response Distribution', fontweight='bold')
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # Response by treatment type
        response_by_treatment = pd.crosstab(
            patient_df['treatment_type'],
            patient_df['treatment_response'],
            normalize='index'
        ) * 100
        
        response_by_treatment.plot(kind='bar', stacked=False, ax=axes[1, 0], 
                                    color=['red', 'orange', 'green'])
        axes[1, 0].set_ylabel('Percentage (%)')
        axes[1, 0].set_xlabel('Treatment Type')
        axes[1, 0].set_title('Response Rate by Treatment Type', fontweight='bold')
        axes[1, 0].legend(title='Response', bbox_to_anchor=(1.05, 1))
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Severity distribution
        severity_counts = patient_df['baseline_severity'].value_counts()
        axes[1, 1].bar(range(len(severity_counts)), severity_counts.values, color='coral')
        axes[1, 1].set_xticks(range(len(severity_counts)))
        axes[1, 1].set_xticklabels(severity_counts.index, rotation=45, ha='right')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Baseline Severity Distribution', fontweight='bold')
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Treatment distribution plot saved to {save_path}")
        plt.close()
    
    def plot_clinical_scores(
        self,
        patient_df: pd.DataFrame,
        save_name: str = "clinical_scores.png"
    ):
        """
        Plot baseline and outcome clinical scores
        
        Args:
            patient_df: Patient profiles DataFrame
            save_name: Filename to save plot
        """
        logger.info("Plotting clinical scores")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # PHQ-9 scores
        axes[0, 0].scatter(patient_df['baseline_phq9'], patient_df['outcome_phq9'], 
                          alpha=0.5, c='blue')
        axes[0, 0].plot([0, 27], [0, 27], 'r--', label='No change')
        axes[0, 0].set_xlabel('Baseline PHQ-9')
        axes[0, 0].set_ylabel('Outcome PHQ-9')
        axes[0, 0].set_title('Depression Scores (PHQ-9)', fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        # GAD-7 scores
        axes[0, 1].scatter(patient_df['baseline_gad7'], patient_df['outcome_gad7'], 
                          alpha=0.5, c='green')
        axes[0, 1].plot([0, 21], [0, 21], 'r--', label='No change')
        axes[0, 1].set_xlabel('Baseline GAD-7')
        axes[0, 1].set_ylabel('Outcome GAD-7')
        axes[0, 1].set_title('Anxiety Scores (GAD-7)', fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        # Improvement by response category
        response_order = ['non-responder', 'partial', 'responder']
        improvement_data = [
            patient_df[patient_df['treatment_response'] == resp]['improvement_percentage'].values
            for resp in response_order if resp in patient_df['treatment_response'].values
        ]
        
        axes[1, 0].boxplot(improvement_data, labels=response_order[:len(improvement_data)])
        axes[1, 0].set_ylabel('Improvement (%)')
        axes[1, 0].set_xlabel('Response Category')
        axes[1, 0].set_title('Improvement by Response Category', fontweight='bold')
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # Session attendance vs outcome
        axes[1, 1].scatter(
            patient_df['session_attendance_rate'],
            patient_df['improvement_percentage'],
            c=patient_df['treatment_response'].map({
                'non-responder': 0, 'partial': 1, 'responder': 2
            }),
            cmap='RdYlGn',
            alpha=0.6
        )
        axes[1, 1].set_xlabel('Session Attendance Rate')
        axes[1, 1].set_ylabel('Improvement (%)')
        axes[1, 1].set_title('Attendance vs Improvement', fontweight='bold')
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Clinical scores plot saved to {save_path}")
        plt.close()
    
    def create_interactive_dashboard(
        self,
        patient_df: pd.DataFrame,
        save_name: str = "dashboard.html"
    ):
        """
        Create interactive Plotly dashboard
        
        Args:
            patient_df: Patient profiles DataFrame
            save_name: Filename to save HTML dashboard
        """
        logger.info("Creating interactive dashboard")
        
        # Create subplots
        from plotly.subplots import make_subplots
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Response by Treatment Type',
                'Baseline Severity Distribution',
                'PHQ-9 Score Changes',
                'Improvement Distribution'
            ),
            specs=[
                [{'type': 'bar'}, {'type': 'pie'}],
                [{'type': 'scatter'}, {'type': 'box'}]
            ]
        )
        
        # Response by treatment type
        response_by_treatment = pd.crosstab(
            patient_df['treatment_type'],
            patient_df['treatment_response']
        )
        
        for response in response_by_treatment.columns:
            fig.add_trace(
                go.Bar(
                    name=response,
                    x=response_by_treatment.index,
                    y=response_by_treatment[response]
                ),
                row=1, col=1
            )
        
        # Severity pie chart
        severity_counts = patient_df['baseline_severity'].value_counts()
        fig.add_trace(
            go.Pie(
                labels=severity_counts.index,
                values=severity_counts.values,
                hole=0.3
            ),
            row=1, col=2
        )
        
        # PHQ-9 scatter
        fig.add_trace(
            go.Scatter(
                x=patient_df['baseline_phq9'],
                y=patient_df['outcome_phq9'],
                mode='markers',
                marker=dict(
                    color=patient_df['treatment_response'].map({
                        'non-responder': 'red',
                        'partial': 'orange',
                        'responder': 'green'
                    }),
                    size=8,
                    opacity=0.6
                ),
                text=patient_df['patient_id']
            ),
            row=2, col=1
        )
        
        # Improvement box plot
        for response in ['non-responder', 'partial', 'responder']:
            data = patient_df[patient_df['treatment_response'] == response]['improvement_percentage']
            fig.add_trace(
                go.Box(
                    y=data,
                    name=response,
                    boxmean='sd'
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title_text="Mental Health Treatment Response Dashboard",
            height=800,
            showlegend=True
        )
        
        save_path = self.output_dir / save_name
        fig.write_html(str(save_path))
        logger.info(f"Interactive dashboard saved to {save_path}")
