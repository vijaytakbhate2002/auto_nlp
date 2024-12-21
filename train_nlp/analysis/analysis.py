import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.metrics import confusion_matrix
from ..nlp_config import nlp_config

class ModelPerformanceAnalyzer:
    def __init__(self, results):
        """
        Initialize the analyzer with model results.
        
        :param results: Dictionary containing model performance metrics.
        """
        self.results = results

    def plot_accuracy(self):
        """
        Plot a bar chart for model accuracies using Plotly.
        """
        accuracies = {model: metrics['accuracy'] for model, metrics in self.results.items()}
        
        fig = go.Figure(go.Bar(
            x=list(accuracies.keys()),
            y=list(accuracies.values()),
            text=[f"{val:.3f}" for val in accuracies.values()],
            textposition="outside",
            marker=dict(color='skyblue'),
            hovertemplate="<b>Model</b>: %{x}<br><b>Accuracy</b>: %{y:.3f}<extra></extra>"
        ))
        
        fig.update_layout(
            title="Model Accuracy Comparison",
            xaxis_title="Models",
            yaxis_title="Accuracy",
            yaxis=dict(range=[0, 1.1]),
            template='plotly_white'
        )
        
        fig.show()

    def plot_all_metrics(self):
        """
        Plot a grouped bar chart comparing precision, recall, and F1-score across models.
        """
        print("model result = ", self.results)
        metrics = ['precision', 'recall', 'f1', 'train_score', 'test_score']
        data = {metric: [np.mean(self.results[model][metric]) for model in self.results] for metric in metrics}
        models = list(self.results.keys())

        fig = go.Figure()
        
        for metric in metrics:
            fig.add_trace(go.Bar(
                name=metric.capitalize(),
                x=models,
                y=data[metric],
                text=[f"{val:.3f}" for val in data[metric]],
                textposition='outside',
                hovertemplate=f"<b>{metric.capitalize()}</b>: {{y:.3f}}<br><b>Model</b>: {{x}}<extra></extra>"
            ))
        
        fig.update_layout(
            title="Model Performance Comparison",
            xaxis_title="Models",
            yaxis_title="Scores",
            yaxis=dict(range=[0, 1.1]),
            barmode='group',
            template='plotly_white',
            legend_title="Metrics"
        )
        
        fig.show()

    def compare_categories(self, categories:list):
        """
        Compare the precision, recall, and F1 scores across categories for each model.
        categories: list (Uniques labels from target column)
        """
        metrics = ['precision', 'recall', 'f1']
        
        for model, scores in self.results.items():
            fig = go.Figure()
            
            for metric in metrics:
                fig.add_trace(go.Bar(
                    name=metric.capitalize(),
                    x=categories,
                    y=scores[metric],
                    text=[f"{val:.3f}" for val in scores[metric]],
                    textposition='outside',
                    hovertemplate=f"<b>{metric.capitalize()}</b>: {{y:.3f}}<br><b>Category</b>: {{x}}<br><extra></extra>"
                ))
            
            fig.update_layout(
                title=f'Category-wise Comparison for {model}',
                xaxis_title='Categories',
                yaxis_title='Scores',
                yaxis=dict(range=[0, 1.1]),
                barmode='group',
                legend_title="Metrics",
                template='plotly_white'
            )
            
            fig.show()

    def plot_confusion_matrix(self, y_true, y_pred, labels=None):
        """
        Plot a confusion matrix using Plotly.

        :param y_true: List or array of true class labels.
        :param y_pred: List or array of predicted class labels.
        :param labels: List of class labels for the matrix.
        """

        if labels is None:
            labels = sorted(set(y_true).union(set(y_pred)))

        cm = confusion_matrix(y_true, y_pred)
        print("confusion matrix = ", cm)
        fig = ff.create_annotated_heatmap(
            z=cm, 
            x=labels, 
            y=labels,
            colorscale="Blues",
            showscale=True,
            annotation_text=cm.astype(str)
        )
        
        fig.update_layout(
            title="Confusion Matrix",
            xaxis_title="Predicted Label",
            yaxis_title="True Label",
            template="plotly_white"
        )
        
        fig.show()

def compare(results:dict, categories:list) -> None:
    """ results: dict[dict]
        categories: list (unique labels from target column)"""
    analyzer = ModelPerformanceAnalyzer(results)
    analyzer.plot_accuracy()          
    analyzer.plot_all_metrics()      
    analyzer.compare_categories(categories=categories)     
