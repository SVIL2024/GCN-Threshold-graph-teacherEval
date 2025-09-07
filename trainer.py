# trainer.py
# =================================================
# Training Module
# =================================================

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from sklearn.metrics import classification_report, confusion_matrix
from models import SGCModel, APPNPModel, EnhancedGCNModel, EnhancedGATModel
from graphbulider import EnhancedGraphBuilder
from config import Config
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, classification_report

# from dataprocess import DataLoader
import pandas as pd
import numpy as np
import os
import random

# Enable deterministic algorithms, but disable for GAT models
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
# Check if we have GAT models in our configuration
has_gat_model = any("GAT" in model_config["class"] for model_config in Config.MODELS.values())

# If we have GAT models, disable deterministic algorithms from the start
if has_gat_model:
    print("GAT models detected, disabling deterministic algorithms")
    torch.use_deterministic_algorithms(False)
else:
    torch.use_deterministic_algorithms(True)

# =================================================
# Training Module
# =================================================

class EnhancedTrainer:
    def __init__(self, model, data, device):
        self.model = model
        self.data = data
        self.device = device
        self._set_random_seed()
        self.model.to(device)
        self.data = data.to(device)

    def _set_random_seed(self):
        """Set random seed for reproducibility"""
        torch.manual_seed(Config.RANDOM_SEED)
        torch.cuda.manual_seed(Config.RANDOM_SEED)
        np.random.seed(Config.RANDOM_SEED)
        random.seed(Config.RANDOM_SEED)
        # Only set deterministic if it's enabled
        if torch.are_deterministic_algorithms_enabled():
            torch.backends.cudnn.deterministic = True
    
    def train(self, training_params):
        """Enhanced training - including learning rate scheduling and early stopping"""
        optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=training_params["lr"], 
            weight_decay=training_params["weight_decay"]
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode=Config.SCHEDULER_PARAMS["mode"], 
            factor=Config.SCHEDULER_PARAMS["factor"], 
            patience=Config.SCHEDULER_PARAMS["patience"]
        )
        
        best_loss = float('inf')
        patience_counter = 0
        
        self.model.train()
        for epoch in range(training_params["epochs"]):
            optimizer.zero_grad()
            
            out = self.model(self.data)
            loss = F.nll_loss(out[self.data.train_mask], self.data.y[self.data.train_mask])
            loss.backward()
            optimizer.step()
            
            scheduler.step(loss)
            
            # Early stopping mechanism
            if loss < best_loss:
                best_loss = loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= Config.TRAINING_PARAMS["patience_limit"]:
                print(f"Early stopping at epoch {epoch+1}")
                break
            
            if (epoch + 1) % Config.TRAINING_PARAMS["print_every"] == 0:
                print(f"Epoch {epoch+1:3d}, Loss: {loss.item():.4f}")

    def evaluate(self):
        """Evaluate model"""
        self.model.eval()
        with torch.no_grad():
            pred = self.model(self.data).argmax(dim=1)
            correct = (pred[self.data.test_mask] == self.data.y[self.data.test_mask]).sum()
            acc = int(correct) / self.data.test_mask.sum().item()
            return acc, pred

# =================================================
# Evaluation Module
# =================================================
class Evaluator:
    def __init__(self):
        pass
    
    def detailed_evaluation(self, data, pred, class_names=['Class 0', 'Class 1']):
        """Detailed evaluation"""
        y_true = data.y[data.test_mask].cpu().numpy()
        y_pred = pred[data.test_mask].cpu().numpy()
        
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_true, y_pred))
        
        return y_true, y_pred

# =================================================
# Experiment Management Module
# =================================================
class EnhancedExperimentManager:
    def __init__(self, X_processed, y, df, handle_imbalance=False, imbalance_method='smote'):
        self.X_processed = X_processed
        self.y = y
        self.df = df
        self.handle_imbalance = handle_imbalance
        self.imbalance_method = imbalance_method
        self.graph_builder = EnhancedGraphBuilder()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        Config.DEVICE = self.device
        # Set number of classes
        Config.NUM_CLASSES = len(np.unique(y))
        self._set_random_seed()
        print(f"Using device: {self.device}")
        print(f"Number of classes: {Config.NUM_CLASSES}")

    def _set_random_seed(self):
        """Set random seed for reproducibility"""
        torch.manual_seed(Config.RANDOM_SEED)
        torch.cuda.manual_seed(Config.RANDOM_SEED)
        np.random.seed(Config.RANDOM_SEED)
        random.seed(Config.RANDOM_SEED)
        torch.backends.cudnn.deterministic = True   

    def build_graph_data(self, adj_matrix, similarity_matrix):
        """Build PyG graph data object"""
        # Set seed to ensure consistent data splitting
        np.random.seed(Config.RANDOM_SEED)
        
        if self.handle_imbalance:
            from dataprocess import DataLoader
            data_loader = DataLoader("")  # Create instance to use methods
            print(f"Handling imbalanced data with method: {self.imbalance_method}")
            X_balanced, y_balanced = data_loader.handle_imbalanced_data(
                self.X_processed, self.y, method=self.imbalance_method
            )
            print(f"Data imbalance before and after: {len(self.y)} -> {len(y_balanced)}")
            self.X_processed = X_balanced
            self.y = y_balanced
            
            # Update number of classes in Config
            Config.NUM_CLASSES = len(np.unique(self.y))
            print(f"Updated number of classes: {Config.NUM_CLASSES}")

        edges = np.array(np.nonzero(adj_matrix))
        edge_index = torch.tensor(edges, dtype=torch.long)
        edge_weight = torch.tensor(similarity_matrix[edges[0], edges[1]], dtype=torch.float)
        
        data = Data(
            x=torch.tensor(self.X_processed.astype(np.float32)),
            edge_index=edge_index,
            edge_attr=edge_weight,
            y=torch.tensor(self.y, dtype=torch.long)
        )
        
        # Debug information
        print(f"Data shape: {data.x.shape}")
        print(f"Number of nodes: {data.num_nodes}")
        print(f"Number of edges: {data.edge_index.shape[1]}")
        print(f"Number of classes: {len(torch.unique(data.y))}")
        print(f"Label distribution: {torch.bincount(data.y)}")
        
        # Improved dataset split - ensure class balance
        num_nodes = data.num_nodes
        
        # Get indices for each class
        unique_classes = np.unique(self.y)
        train_indices, val_indices, test_indices = [], [], []
        
        for cls in unique_classes:
            class_indices = np.where(self.y == cls)[0]
            np.random.shuffle(class_indices)
            
            n_train = int(len(class_indices) * Config.TRAIN_RATIO)
            n_val = int(len(class_indices) * Config.VAL_RATIO)
            
            train_indices.extend(class_indices[:n_train])
            val_indices.extend(class_indices[n_train:n_train+n_val])
            test_indices.extend(class_indices[n_train+n_val:])
        
        # Create masks
        data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        data.train_mask[train_indices] = True
        data.val_mask[val_indices] = True
        data.test_mask[test_indices] = True
        
        print(f"Train samples: {data.train_mask.sum().item()}")
        print(f"Val samples: {data.val_mask.sum().item()}")
        print(f"Test samples: {data.test_mask.sum().item()}")
        
        return data 
 
    def run_experiment(self, graph_method, model_class, model_params, training_params):
        """Run a single experiment with comprehensive evaluation"""
        # Set seed for reproducible experiments
        self._set_random_seed()
        
        method_name, method_func_name, method_params = graph_method
        method_func = getattr(self.graph_builder, method_func_name)
        
        # Build graph
        if method_params:
            adj_matrix, similarity_matrix = method_func(self.X_processed, **method_params)
        else:
            adj_matrix, similarity_matrix = method_func(self.X_processed)
        
        print(f"  Building graph: {np.count_nonzero(adj_matrix)} edges")
        
        # Build data
        data = self.build_graph_data(adj_matrix, similarity_matrix)
        
        # Create model with num_classes parameter
        model_params_with_classes = model_params.copy()
        model_params_with_classes["num_classes"] = Config.NUM_CLASSES
        
        model = model_class(self.X_processed.shape[1], **model_params_with_classes)
        
        # Debug model
        print(f"  Model created: {model.__class__.__name__}")
        
        # Train and evaluate
        trainer = EnhancedTrainer(model, data, self.device)
        
        trainer.train(training_params)
        acc, pred = trainer.evaluate()
        
        # Calculate detailed metrics in EnhancedExperimentManager
        with torch.no_grad():
            model.eval()
            out = model(data)
            pred_prob = torch.softmax(out, dim=1)
            
            y_true = data.y[data.test_mask].cpu().numpy()
            y_pred = pred[data.test_mask].cpu().numpy()
            y_prob = pred_prob[data.test_mask].cpu().numpy()
            
            detailed_metrics = self.calculate_detailed_metrics(y_true, y_pred, y_prob)
        
        print(f"  Accuracy: {acc:.4f}")
        print(f"  F1-Score (weighted): {detailed_metrics['f1_weighted']:.4f}")
        print(f"  F1-Score (macro): {detailed_metrics['f1_macro']:.4f}")
        print(f"  Precision (weighted): {detailed_metrics['precision_weighted']:.4f}")
        print(f"  Recall (weighted): {detailed_metrics['recall_weighted']:.4f}")
        if 'auc' in detailed_metrics:
            print(f"  AUC: {detailed_metrics['auc']:.4f}")

        # Return accuracy, method name, trained model, and detailed metrics
        return acc, method_name, model, data, detailed_metrics

    def ensemble_predict(self, models, data, model_weights=None):
        """Ensemble prediction with weighted voting"""
        import torch
        
        if model_weights is None:
            model_weights = [1.0] * len(models)  # Default equal weights
        
        predictions = []
        for model in models:
            model.eval()
            with torch.no_grad():
                # Get probability distribution instead of direct prediction
                logits = model(data)
                probs = torch.softmax(logits, dim=1)
                predictions.append(probs)
        
        # Weighted average
        weighted_probs = torch.zeros_like(predictions[0])
        total_weight = sum(model_weights)
        
        for i, (probs, weight) in enumerate(zip(predictions, model_weights)):
            weighted_probs += probs * (weight / total_weight)
        
        # Final prediction
        final_pred = weighted_probs.argmax(dim=1)
        return final_pred

    # Modify compare_all_combinations method in EnhancedExperimentManager class
    def compare_all_combinations(self):
        """Compare all graph construction methods and models with comprehensive metrics"""
        print("="*60)
        print("Starting comprehensive comparison experiment")
        print("="*60)
        
        # Store results
        results = []
        
        # Run experiments for each graph construction method and model
        for graph_method in Config.GRAPH_METHODS:
            method_name = graph_method[0]
            print(f"\n--- Using {method_name} ---")
            
            for model_name, model_config in Config.MODELS.items():
                print(f"  Training {model_name} model...")
                
                # Get model class
                model_classes = {
                    "EnhancedGCNModel": EnhancedGCNModel,
                    "EnhancedGATModel": EnhancedGATModel,
                    "SGCModel": SGCModel,
                    "APPNPModel": APPNPModel
                }
                
                try:
                    # Receive all return values
                    acc, _, _, _, detailed_metrics = self.run_experiment(
                        graph_method,
                        model_classes[model_config["class"]],
                        model_config["params"],
                        model_config["training"]
                    )
                    
                    results.append({
                        "Graph Method": method_name,
                        "Model": model_name,
                        "Accuracy": acc,
                        "F1-Score (Weighted)": detailed_metrics['f1_weighted'],
                        "F1-Score (Macro)": detailed_metrics['f1_macro'],
                        "Precision (Weighted)": detailed_metrics['precision_weighted'],
                        "Recall (Weighted)": detailed_metrics['recall_weighted'],
                        "AUC": detailed_metrics.get('auc', 0.0)
                    })
                    print(f"  {model_name} accuracy: {acc:.4f}")
                    
                except Exception as e:
                    print(f"  {model_name} training failed: {e}")
                    import traceback
                    traceback.print_exc()
                    results.append({
                        "Graph Method": method_name,
                        "Model": model_name,
                        "Accuracy": 0.0,
                        "F1-Score (Weighted)": 0.0,
                        "F1-Score (Macro)": 0.0,
                        "Precision (Weighted)": 0.0,
                        "Recall (Weighted)": 0.0,
                        "AUC": 0.0
                    })
        
        # Summary of results
        print("\n" + "="*60)
        print("Experimental Results Summary")
        print("="*60)
        
        # Convert to DataFrame for analysis
        results_df = pd.DataFrame(results)
        print(results_df.to_string(index=False))
        
        # Find best combination based on F1-score (better for imbalanced data)
        best_result_idx = results_df['F1-Score (Weighted)'].idxmax()
        best_result = results_df.loc[best_result_idx]
        print(f"\nBest combination (based on F1-weighted score):")
        print(f"  Graph method: {best_result['Graph Method']}")
        print(f"  Model: {best_result['Model']}")
        print(f"  F1-Score (weighted): {best_result['F1-Score (Weighted)']:.4f}")
        print(f"  F1-Score (macro): {best_result['F1-Score (Macro)']:.4f}")
        print(f"  Accuracy: {best_result['Accuracy']:.4f}")
        print(f"  Precision (weighted): {best_result['Precision (Weighted)']:.4f}")
        print(f"  Recall (weighted): {best_result['Recall (Weighted)']:.4f}")
        if 'AUC' in best_result:
            print(f"  AUC: {best_result['AUC']:.4f}")
        
        # Average performance by graph method
        print("\nAverage performance by graph method (based on F1-weighted score):")
        graph_performance = results_df.groupby('Graph Method')['F1-Score (Weighted)'].mean().sort_values(ascending=False)
        for method, avg_f1 in graph_performance.items():
            print(f"  {method}: {avg_f1:.4f}")
        
        # Average performance by model
        print("\nAverage performance by model (based on F1-weighted score):")
        model_performance = results_df.groupby('Model')['F1-Score (Weighted)'].mean().sort_values(ascending=False)
        for model, avg_f1 in model_performance.items():
            print(f"  {model}: {avg_f1:.4f}")
        
        return results_df


    def ensemble_predict(self, models, data, model_weights=None):
        """模型集成预测 - 支持加权投票"""
        import torch
        
        # 检查是否有GAT模型
        has_gat_model = any("GAT" in model.__class__.__name__ for model in models)
        
        # 保存当前确定性算法状态
        deterministic_was_enabled = torch.are_deterministic_algorithms_enabled() if hasattr(torch, 'are_deterministic_algorithms_enabled') else True
        
        # 如果有GAT模型，临时禁用确定性算法
        # if has_gat_model:
        #     torch.use_deterministic_algorithms(False)
        
        try:
            if model_weights is None:
                model_weights = [1.0] * len(models)  # 默认等权重
            
            predictions = []
            for model in models:
                model.eval()
                with torch.no_grad():
                    # 获取概率分布而不是直接预测
                    logits = model(data)
                    probs = torch.softmax(logits, dim=1)
                    predictions.append(probs)
            
            # 加权平均
            weighted_probs = torch.zeros_like(predictions[0])
            total_weight = sum(model_weights)
            
            for i, (probs, weight) in enumerate(zip(predictions, model_weights)):
                weighted_probs += probs * (weight / total_weight)
            
            # 最终预测
            final_pred = weighted_probs.argmax(dim=1)
            return final_pred
            
        finally:
            # 恢复原来的确定性算法设置
            if has_gat_model and deterministic_was_enabled:
                torch.use_deterministic_algorithms(True)

    # Add new method in EnhancedExperimentManager class
    def calculate_detailed_metrics(self, y_true, y_pred, y_prob=None):
        """
        Calculate detailed evaluation metrics, better for imbalanced data
        """
        metrics = {}
        
        # Basic metrics (using weighted average to handle imbalanced data)
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Macro average metrics (focus more on minority classes)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        
        # If probabilities are provided, calculate AUC
        if y_prob is not None:
            try:
                if len(np.unique(y_true)) == 2:  # Binary classification
                    metrics['auc'] = roc_auc_score(y_true, y_prob[:, 1])
                else:  # Multi-class
                    metrics['auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr')
            except:
                metrics['auc'] = 0.0
        
        return metrics
        


        