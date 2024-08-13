import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast

class ModelFiguresSaver:
    def __init__(self, data):
        self.data = pd.read_csv(data) if isinstance(data, str) else pd.DataFrame(data)
        self.data['Confusion Matrix'] = self.data['Confusion Matrix'].apply(ast.literal_eval)
        sns.set_theme(style="darkgrid")

    def save_accuracy_barplot(self, filename='accuracy_barplot.png'):
        plt.figure(figsize=(12, 8))
        barplot = sns.barplot(x='Accuracy', y='Model', data=self.data, palette='viridis')
        plt.title('Model Accuracy', fontsize=16, fontweight='bold')
        plt.xlabel('Accuracy', fontsize=14, fontweight='bold')
        plt.ylabel('Model', fontsize=14, fontweight='bold')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        for p in barplot.patches:
            barplot.annotate(format(p.get_width(), '.2f'), 
                             (p.get_width(), p.get_y() + p.get_height() / 2.), 
                             ha = 'center', va = 'center', 
                             xytext = (20, 0), 
                             textcoords = 'offset points', fontsize=12)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def save_precision_barplot(self, filename='precision_barplot.png'):
        plt.figure(figsize=(12, 8))
        barplot = sns.barplot(x='Precision', y='Model', data=self.data, palette='magma')
        plt.title('Model Precision', fontsize=16, fontweight='bold')
        plt.xlabel('Precision', fontsize=14, fontweight='bold')
        plt.ylabel('Model', fontsize=14, fontweight='bold')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        for p in barplot.patches:
            barplot.annotate(format(p.get_width(), '.2f'), 
                             (p.get_width(), p.get_y() + p.get_height() / 2.), 
                             ha = 'center', va = 'center', 
                             xytext = (20, 0), 
                             textcoords = 'offset points', fontsize=12)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def save_recall_barplot(self, filename='recall_barplot.png'):
        plt.figure(figsize=(12, 8))
        barplot = sns.barplot(x='Recall', y='Model', data=self.data, palette='coolwarm')
        plt.title('Model Recall', fontsize=16, fontweight='bold')
        plt.xlabel('Recall', fontsize=14, fontweight='bold')
        plt.ylabel('Model', fontsize=14, fontweight='bold')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        for p in barplot.patches:
            barplot.annotate(format(p.get_width(), '.2f'), 
                             (p.get_width(), p.get_y() + p.get_height() / 2.), 
                             ha = 'center', va = 'center', 
                             xytext = (20, 0), 
                             textcoords = 'offset points', fontsize=12)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def save_f1_score_barplot(self, filename='f1_score_barplot.png'):
        plt.figure(figsize=(12, 8))
        barplot = sns.barplot(x='F1 Score', y='Model', data=self.data, palette='plasma')
        plt.title('Model F1 Score', fontsize=16, fontweight='bold')
        plt.xlabel('F1 Score', fontsize=14, fontweight='bold')
        plt.ylabel('Model', fontsize=14, fontweight='bold')
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        for p in barplot.patches:
            barplot.annotate(format(p.get_width(), '.2f'), 
                             (p.get_width(), p.get_y() + p.get_height() / 2.), 
                             ha = 'center', va = 'center', 
                             xytext = (20, 0), 
                             textcoords = 'offset points', fontsize=12)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def save_confusion_matrices(self, directory='confusion_matrices'):
        import os
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        for index, row in self.data.iterrows():
            cm = row['Confusion Matrix']
            model_name = row['Model']
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={"size": 14})
            plt.title(f'Confusion Matrix for {model_name}', fontsize=16, fontweight='bold')
            plt.xlabel('Predicted', fontsize=14, fontweight='bold')
            plt.ylabel('Actual', fontsize=14, fontweight='bold')
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.tight_layout()
            plt.savefig(f"{directory}/{model_name.replace(' ', '_')}_confusion_matrix.png")
            plt.close()

    def save_all_figures(self):
        self.save_accuracy_barplot()
        self.save_precision_barplot()
        self.save_recall_barplot()
        self.save_f1_score_barplot()
        self.save_confusion_matrices()