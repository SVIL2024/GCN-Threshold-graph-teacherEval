# gcn_03_traditional_mlp_comparison.py
# =================================================
# GCN (Threshold=0.3) vs Traditional ML vs MLP Comparison
# =================================================

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

# Import project modules
from dataprocess import DataLoader
from trainer import EnhancedExperimentManager
from models import EnhancedGCNModel
from config import Config

# =================================================
# MLP Model Definition
# =================================================
class MLPModel(nn.Module):
    """Multi-Layer Perceptron Model"""
    def __init__(self, input_dim, hidden_dims, num_classes, dropout=0.5):
        super(MLPModel, self).__init__()
        self.layers = nn.ModuleList()
        self.dropout = dropout
        
        # Input layer
        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
        self.layers.append(nn.BatchNorm1d(hidden_dims[0]))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(dropout))
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            self.layers.append(nn.BatchNorm1d(hidden_dims[i+1]))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout))
        
        # Output layer
        self.layers.append(nn.Linear(hidden_dims[-1], num_classes))
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return F.log_softmax(x, dim=1)

# =================================================
# Training Functions
# =================================================
def train_mlp_model(model, X_train, y_train, X_val, y_val, epochs=200, lr=0.001, weight_decay=1e-4):
    """Train MLP model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.LongTensor(y_val).to(device)
    
    best_loss = float('inf')
    patience_counter = 0
    patience_limit = 20
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        output = model(X_train_tensor)
        loss = F.nll_loss(output, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_output = model(X_val_tensor)
            val_loss = F.nll_loss(val_output, y_val_tensor)
        
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience_limit:
            print(f"Early stopping at epoch {epoch+1}")
            break
            
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:3d}, Training Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}")
    
    return model

def evaluate_mlp_model(model, X_test, y_test):
    """Evaluate MLP model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.LongTensor(y_test).to(device)
    
    with torch.no_grad():
        output = model(X_test_tensor)
        pred = output.argmax(dim=1)
        correct = (pred == y_test_tensor).sum().item()
        accuracy = correct / len(y_test)
        
        # Calculate probabilities for detailed metrics
        prob = torch.softmax(output, dim=1)
        
    return accuracy, pred.cpu().numpy(), prob.cpu().numpy()

# =================================================
# Comparison Functions
# =================================================
def compare_traditional_ml_methods(processed_data, handle_imbalance=False, imbalance_method='smote'):
    """Compare traditional machine learning methods using the same data split as GNN"""
    
    data_loader = DataLoader("")
    
    if handle_imbalance:
        print(f"Processing imbalanced data for traditional ML methods, using method: {imbalance_method}")
        X_train_balanced, y_train_balanced = data_loader.handle_imbalanced_data(
            processed_data['X_train'], processed_data['y_train'], 
            method=imbalance_method
        )
        print(f"Training data before and after balancing: {len(processed_data['y_train'])} -> {len(y_train_balanced)}")
        X_train = X_train_balanced
        y_train = y_train_balanced
    else:
        X_train = processed_data['X_train']
        y_train = processed_data['y_train']
    
    X_val = processed_data['X_val']
    X_test = processed_data['X_test']
    y_val = processed_data['y_val']
    y_test = processed_data['y_test']
    
    # Define models
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=Config.RANDOM_SEED),
        "SVM": SVC(random_state=Config.RANDOM_SEED, probability=True),  # Enable probability prediction
        "Logistic Regression": LogisticRegression(random_state=Config.RANDOM_SEED, max_iter=1000)
    }
    
    results = {}
    
    print("\n" + "="*60)
    print("Traditional Machine Learning Methods Comparison")
    print("="*60)
    
    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate on validation set to select best model
        y_val_pred = model.predict(X_val)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        
        # Evaluate on test set
        y_test_pred = model.predict(X_test)
        y_test_prob = model.predict_proba(X_test)  # Get prediction probabilities
        
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred, average='weighted')
        test_precision = precision_score(y_test, y_test_pred, average='weighted')
        test_recall = recall_score(y_test, y_test_pred, average='weighted')
        
        # Calculate AUC
        try:
            if len(np.unique(y_test)) == 2:  # Binary classification
                test_auc = roc_auc_score(y_test, y_test_prob[:, 1])
            else:  # Multi-class
                test_auc = roc_auc_score(y_test, y_test_prob, multi_class='ovr')
        except:
            test_auc = 0.0
        
        results[name] = {
            'val_accuracy': val_accuracy,
            'test_accuracy': test_accuracy,
            'test_f1': test_f1,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_auc': test_auc
        }
        
        print(f"{name}:")
        print(f"  Validation accuracy: {val_accuracy:.4f}")
        print(f"  Test accuracy: {test_accuracy:.4f}")
        print(f"  F1-Score (weighted): {test_f1:.4f}")
        print(f"  Precision: {test_precision:.4f}")
        print(f"  Recall: {test_recall:.4f}")
        print(f"  AUC: {test_auc:.4f}")
        
        # Print detailed classification report
        print(f"  Classification report:")
        print(classification_report(y_test, y_test_pred, zero_division=0))
        print()
    
    return results

def compare_with_mlp(processed_data, handle_imbalance=False, imbalance_method='smote'):
    """Compare with MLP model"""
    
    data_loader = DataLoader("")
    
    if handle_imbalance:
        print(f"Processing imbalanced data for MLP, using method: {imbalance_method}")
        X_train_balanced, y_train_balanced = data_loader.handle_imbalanced_data(
            processed_data['X_train'], processed_data['y_train'], 
            method=imbalance_method
        )
        print(f"Training data before and after balancing: {len(processed_data['y_train'])} -> {len(y_train_balanced)}")
        X_train = X_train_balanced
        y_train = y_train_balanced
    else:
        X_train = processed_data['X_train']
        y_train = processed_data['y_train']
    
    X_val = processed_data['X_val']
    X_test = processed_data['X_test']
    y_val = processed_data['y_val']
    y_test = processed_data['y_test']
    
    # Get input dimensions
    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))
    
    print(f"\nMLP Model Configuration:")
    print(f"  Input dimensions: {input_dim}")
    print(f"  Number of classes: {num_classes}")
    print(f"  Hidden layers: [64, 32]")
    
    # Create MLP model
    mlp_model = MLPModel(input_dim, [64, 32], num_classes, dropout=0.5)
    
    # Train MLP model
    print("\nTraining MLP model...")
    trained_mlp = train_mlp_model(mlp_model, X_train, y_train, X_val, y_val, 
                                 epochs=200, lr=0.001, weight_decay=1e-4)
    
    # Evaluate MLP model
    print("\nEvaluating MLP model...")
    accuracy, y_pred, y_prob = evaluate_mlp_model(trained_mlp, X_test, y_test)
    
    # Calculate detailed metrics
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    
    try:
        if len(np.unique(y_test)) == 2:  # Binary classification
            auc = roc_auc_score(y_test, y_prob[:, 1])
        else:  # Multi-class
            auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
    except:
        auc = 0.0
    
    print(f"\nMLP Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1-Score (weighted): {f1:.4f}")
    print(f"  Precision (weighted): {precision:.4f}")
    print(f"  Recall (weighted): {recall:.4f}")
    print(f"  AUC: {auc:.4f}")
    
    # Print detailed classification report
    print(f"\nClassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    mlp_results = {
        'test_accuracy': accuracy,
        'test_f1': f1,
        'test_precision': precision,
        'test_recall': recall,
        'test_auc': auc
    }
    
    return mlp_results

def compare_gcn_threshold_03(X_processed, y, df, handle_imbalance=False, imbalance_method='smote'):
    """Compare GCN model with 0.3 threshold"""
    
    print("\n" + "="*60)
    print("GCN Model with 0.3 Threshold")
    print("="*60)
    
    # Define graph method with 0.3 threshold
    graph_method = ("Threshold Graph(0.3)", "build_threshold_graph", {"threshold": 0.3})
    
    # Define model configuration (using EnhancedGCN)
    model_config = Config.MODELS["EnhancedGCN"]
    model_params = model_config["params"].copy()
    training_params = model_config["training"]
    
    # Create experiment manager
    experiment_manager = EnhancedExperimentManager(
        X_processed, y, df,
        handle_imbalance=handle_imbalance,
        imbalance_method=imbalance_method
    )
    
    try:
        # Run GCN experiment
        gcn_acc, method_name, gcn_model, gcn_data, gcn_detailed_metrics = experiment_manager.run_experiment(
            graph_method,
            EnhancedGCNModel,
            model_params,
            training_params
        )
        
        gcn_results = {
            'accuracy': gcn_acc,
            'f1_weighted': gcn_detailed_metrics['f1_weighted'],
            'f1_macro': gcn_detailed_metrics['f1_macro'],
            'precision_weighted': gcn_detailed_metrics['precision_weighted'],
            'recall_weighted': gcn_detailed_metrics['recall_weighted'],
            'auc': gcn_detailed_metrics.get('auc', 0.0)
        }
        
        print(f"\nGCN Results with 0.3 Threshold:")
        print(f"  Accuracy: {gcn_acc:.4f}")
        print(f"  F1-Score (weighted): {gcn_detailed_metrics['f1_weighted']:.4f}")
        print(f"  F1-Score (macro): {gcn_detailed_metrics['f1_macro']:.4f}")
        print(f"  Precision (weighted): {gcn_detailed_metrics['precision_weighted']:.4f}")
        print(f"  Recall (weighted): {gcn_detailed_metrics['recall_weighted']:.4f}")
        print(f"  AUC: {gcn_detailed_metrics.get('auc', 0.0):.4f}")
        
    except Exception as e:
        print(f"Error training GCN with 0.3 threshold: {e}")
        import traceback
        traceback.print_exc()
        gcn_results = {
            'accuracy': 0.0,
            'f1_weighted': 0.0,
            'f1_macro': 0.0,
            'precision_weighted': 0.0,
            'recall_weighted': 0.0,
            'auc': 0.0
        }
    
    return gcn_results

# =================================================
# Visualization Functions
# =================================================
def plot_comprehensive_comparison(gcn_results, ml_results, mlp_results):
    """Plot comprehensive comparison of all methods"""
    try:
        # Set font
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # Prepare data
        methods = ['GCN (Threshold=0.3)'] + list(ml_results.keys()) + ['MLP']
        
        accuracies = [gcn_results['accuracy']] + \
                    [result['test_accuracy'] for result in ml_results.values()] + \
                    [mlp_results['test_accuracy']]
        
        f1_scores = [gcn_results['f1_weighted']] + \
                   [result['test_f1'] for result in ml_results.values()] + \
                   [mlp_results['test_f1']]
        
        auc_scores = [gcn_results['auc']] + \
                   [result['test_auc'] for result in ml_results.values()] + \
                   [mlp_results['test_auc']]
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot Accuracy
        bars1 = axes[0].bar(methods, accuracies, color=plt.cm.Set3(np.linspace(0, 1, len(methods))))
        axes[0].set_title('Accuracy Comparison')
        axes[0].set_ylabel('Accuracy')
        axes[0].set_ylim(0, 1)
        axes[0].tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, value in zip(bars1, accuracies):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
        
        # Plot F1-Score
        bars2 = axes[1].bar(methods, f1_scores, color=plt.cm.Set3(np.linspace(0, 1, len(methods))))
        axes[1].set_title('F1-Score (Weighted) Comparison')
        axes[1].set_ylabel('F1-Score')
        axes[1].set_ylim(0, 1)
        axes[1].tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, value in zip(bars2, f1_scores):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
        
        # Plot AUC
        bars3 = axes[2].bar(methods, auc_scores, color=plt.cm.Set3(np.linspace(0, 1, len(methods))))
        axes[2].set_title('AUC Comparison')
        axes[2].set_ylabel('AUC')
        axes[2].set_ylim(0, 1)
        axes[2].tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, value in zip(bars3, auc_scores):
            axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('gcn_03_traditional_mlp_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Comparison chart saved as gcn_03_traditional_mlp_comparison.png")
        
    except Exception as e:
        print(f"Error plotting comparison: {e}")

def plot_radar_chart(gcn_results, ml_results, mlp_results):
    """Plot radar chart for multi-dimensional comparison"""
    try:
        import numpy as np
        
        # Set font
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # Methods
        methods = ['GCN (Threshold=0.3)', 'Best Traditional ML', 'MLP']
        
        # Select best traditional ML method based on F1-score
        best_ml_name = max(ml_results.keys(), key=lambda k: ml_results[k]['test_f1'])
        best_ml_result = ml_results[best_ml_name]
        
        # Metrics for each method
        metrics_data = {
            'GCN (Threshold=0.3)': [
                gcn_results['accuracy'],
                gcn_results['f1_weighted'],
                gcn_results['auc']
            ],
            f'Best ML ({best_ml_name})': [
                best_ml_result['test_accuracy'],
                best_ml_result['test_f1'],
                best_ml_result['test_auc']
            ],
            'MLP': [
                mlp_results['test_accuracy'],
                mlp_results['test_f1'],
                mlp_results['test_auc']
            ]
        }
        
        # Metrics labels
        metrics_labels = ['Accuracy', 'F1-Score\n(Weighted)', 'AUC']
        num_vars = len(metrics_labels)
        
        # Compute angle for each axis
        angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
        angles += angles[:1]  # Complete the circle
        
        # Initialise the spider plot
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Draw one axe per variable + add labels
        plt.xticks(angles[:-1], metrics_labels, color='grey', size=12)
        
        # Draw ylabels
        ax.set_rlabel_position(0)
        plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=10)
        plt.ylim(0, 1)
        
        # Plot data for each method
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        for i, (method, color) in enumerate(zip(methods, colors)):
            values = list(metrics_data.values())[i]
            values += values[:1]  # Complete the circle
            ax.plot(angles, values, 'o-', linewidth=2, label=method, color=color)
            ax.fill(angles, values, alpha=0.25, color=color)
        
        # Add a title
        plt.title('Performance Comparison: GCN vs Traditional ML vs MLP', size=14, pad=20)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        plt.savefig('gcn_03_radar_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Radar chart saved as gcn_03_radar_comparison.png")
        
    except Exception as e:
        print(f"Error plotting radar chart: {e}")

def print_summary_table(gcn_results, ml_results, mlp_results):
    """Print summary comparison table"""
    print("\n" + "="*80)
    print("SUMMARY COMPARISON TABLE")
    print("="*80)
    
    # Create summary data
    summary_data = []
    
    # GCN results
    summary_data.append({
        'Method': 'GCN (Threshold=0.3)',
        'Accuracy': gcn_results['accuracy'],
        'F1-Score (Weighted)': gcn_results['f1_weighted'],
        'Precision (Weighted)': gcn_results['precision_weighted'],
        'Recall (Weighted)': gcn_results['recall_weighted'],
        'AUC': gcn_results['auc']
    })
    
    # Traditional ML results
    for method_name, results in ml_results.items():
        summary_data.append({
            'Method': method_name,
            'Accuracy': results['test_accuracy'],
            'F1-Score (Weighted)': results['test_f1'],
            'Precision (Weighted)': results['test_precision'],
            'Recall (Weighted)': results['test_recall'],
            'AUC': results['test_auc']
        })
    
    # MLP results
    summary_data.append({
        'Method': 'MLP',
        'Accuracy': mlp_results['test_accuracy'],
        'F1-Score (Weighted)': mlp_results['test_f1'],
        'Precision (Weighted)': mlp_results['test_precision'],
        'Recall (Weighted)': mlp_results['test_recall'],
        'AUC': mlp_results['test_auc']
    })
    
    # Convert to DataFrame for better display
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False, float_format='%.4f'))
    
    # Find best method for each metric
    print("\n" + "-"*80)
    print("BEST METHODS BY METRIC")
    print("-"*80)
    
    best_accuracy = summary_df.loc[summary_df['Accuracy'].idxmax()]
    best_f1 = summary_df.loc[summary_df['F1-Score (Weighted)'].idxmax()]
    best_auc = summary_df.loc[summary_df['AUC'].idxmax()]
    
    print(f"Highest Accuracy: {best_accuracy['Method']} ({best_accuracy['Accuracy']:.4f})")
    print(f"Highest F1-Score: {best_f1['Method']} ({best_f1['F1-Score (Weighted)']:.4f})")
    print(f"Highest AUC: {best_auc['Method']} ({best_auc['AUC']:.4f})")

# =================================================
# Main Function
# =================================================
def main():
    """Main function to run GCN vs traditional ML vs MLP comparison experiment"""
    try:
        print("="*80)
        print("GCN (Threshold=0.3) vs Traditional ML vs MLP Methods Comparison")
        print("="*80)
        
        # Configuration
        file_path = "teacher_info_2025_name-7-3-english.xlsx"
        
        # 1. Data loading
        print("\n" + "="*60)
        print("Data Loading Phase")
        print("="*60)
        data_loader = DataLoader(file_path)
        df = data_loader.load_data()
        
        # 2. Data preprocessing for traditional ML methods (maintaining same split)
        processed_data = data_loader.create_simple_classification_dataset_with_split(df)
        
        # 3. Traditional ML methods comparison
        ml_results = compare_traditional_ml_methods(
            processed_data,
            handle_imbalance=Config.HANDLE_IMBALANCE["enabled"],
            imbalance_method=Config.HANDLE_IMBALANCE["method"]
        )
        
        # 4. MLP comparison
        print("\n" + "="*60)
        print("MLP Model Comparison")
        print("="*60)
        mlp_results = compare_with_mlp(
            processed_data,
            handle_imbalance=Config.HANDLE_IMBALANCE["enabled"],
            imbalance_method=Config.HANDLE_IMBALANCE["method"]
        )
        
        # 5. GCN model with 0.3 threshold
        X_processed, y = data_loader.preprocess_data(df)  # Returns two values
        gcn_results = compare_gcn_threshold_03(
            X_processed, y, df,
            handle_imbalance=Config.HANDLE_IMBALANCE["enabled"],
            imbalance_method=Config.HANDLE_IMBALANCE["method"]
        )
        
        # 6. Summary and visualization
        print_summary_table(gcn_results, ml_results, mlp_results)
        
        # 7. Generate visualization charts
        print("\n" + "="*60)
        print("Generating Visualization Charts")
        print("="*60)
        
        plot_comprehensive_comparison(gcn_results, ml_results, mlp_results)
        plot_radar_chart(gcn_results, ml_results, mlp_results)
        
        print("\n" + "="*60)
        print("Comparison Experiment Completed")
        print("="*60)
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()