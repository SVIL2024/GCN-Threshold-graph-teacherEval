# main_gcn_threshold.py
# =================================================
# Main Program for Threshold Comparison Experiment
# =================================================
import pandas as pd
import numpy as np
from config import Config
from dataprocess import DataLoader
from trainer import EnhancedExperimentManager
from models import EnhancedGCNModel
import matplotlib.pyplot as plt

def compare_gcn_thresholds():
    """Main function to compare GCN model performance with different threshold values"""
    
    # Configuration
    file_path = "teacher_info_2025_name-7-3-english.xlsx"
    
    # 1. Data loading
    print("="*60)
    print("Data Loading Phase")
    print("="*60)
    data_loader = DataLoader(file_path)
    df = data_loader.load_data()
    
    # 2. Data preprocessing
    X_processed, y = data_loader.preprocess_data(df)
    
    # 3. Define threshold values to test
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    # 4. Store results
    results = []
    
    print("\n" + "="*60)
    print("GCN Model Threshold Comparison Experiment")
    print("="*60)
    
    # 5. Run experiments for each threshold
    for threshold in thresholds:
        print(f"\n--- Testing Threshold: {threshold} ---")
        
        # Define graph method with current threshold
        graph_method = (f"Threshold Graph({threshold})", "build_threshold_graph", {"threshold": threshold})
        
        # Define model configuration (using EnhancedGCN)
        model_config = Config.MODELS["EnhancedGCN"]
        model_params = model_config["params"].copy()
        training_params = model_config["training"]
        
        # Create experiment manager
        experiment_manager = EnhancedExperimentManager(
            X_processed, y, df,
            handle_imbalance=Config.HANDLE_IMBALANCE["enabled"],
            imbalance_method=Config.HANDLE_IMBALANCE["method"]
        )
        
        try:
            # Run experiment
            acc, method_name, model, data, detailed_metrics = experiment_manager.run_experiment(
                graph_method,
                EnhancedGCNModel,
                model_params,
                training_params
            )
            
            # Store results
            results.append({
                "Threshold": threshold,
                "Graph_Method": method_name,
                "Accuracy": acc,
                "F1_Score_Weighted": detailed_metrics['f1_weighted'],
                "F1_Score_Macro": detailed_metrics['f1_macro'],
                "Precision_Weighted": detailed_metrics['precision_weighted'],
                "Recall_Weighted": detailed_metrics['recall_weighted'],
                "AUC": detailed_metrics.get('auc', 0.0)
            })
            
            print(f"Results for threshold {threshold}:")
            print(f"  Accuracy: {acc:.4f}")
            print(f"  F1-Score (Weighted): {detailed_metrics['f1_weighted']:.4f}")
            print(f"  F1-Score (Macro): {detailed_metrics['f1_macro']:.4f}")
            print(f"  AUC: {detailed_metrics.get('auc', 0.0):.4f}")
            
        except Exception as e:
            print(f"Error with threshold {threshold}: {e}")
            results.append({
                "Threshold": threshold,
                "Graph_Method": f"Threshold Graph({threshold})",
                "Accuracy": 0.0,
                "F1_Score_Weighted": 0.0,
                "F1_Score_Macro": 0.0,
                "Precision_Weighted": 0.0,
                "Recall_Weighted": 0.0,
                "AUC": 0.0
            })
    
    # 6. Convert results to DataFrame
    results_df = pd.DataFrame(results)
    print("\n" + "="*60)
    print("Final Results Summary")
    print("="*60)
    print(results_df.to_string(index=False))
    
    # 7. Find best threshold
    best_idx = results_df['F1_Score_Weighted'].idxmax()
    best_result = results_df.loc[best_idx]
    
    print(f"\nBest threshold (based on F1-Score Weighted):")
    print(f"  Threshold: {best_result['Threshold']}")
    print(f"  F1-Score (Weighted): {best_result['F1_Score_Weighted']:.4f}")
    print(f"  F1-Score (Macro): {best_result['F1_Score_Macro']:.4f}")
    print(f"  Accuracy: {best_result['Accuracy']:.4f}")
    print(f"  AUC: {best_result['AUC']:.4f}")
    
    # 8. Save results to CSV
    results_df.to_csv("threshold_comparison_results.csv", index=False)
    print(f"\nResults saved to threshold_comparison_results.csv")
    
    # 9. Plot results
    plot_threshold_comparison(results_df)
    
    return results_df

def plot_threshold_comparison(results_df):
    """Plot threshold comparison results with AUC"""
    try:
        # Set style
        plt.style.use('seaborn-v0_8')
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Extract data
        thresholds = results_df['Threshold']
        f1_scores = results_df['F1_Score_Weighted']
        accuracies = results_df['Accuracy']
        auc_scores = results_df['AUC']
        
        # Plot 1: Accuracy and F1-Score
        line1 = ax1.plot(thresholds, f1_scores, marker='o', linewidth=2, label='F1-Score (Weighted)', markersize=8)
        line2 = ax1.plot(thresholds, accuracies, marker='s', linewidth=2, label='Accuracy', markersize=8)
        
        ax1.set_xlabel('Threshold Value', fontsize=12)
        ax1.set_ylabel('Performance Score', fontsize=12)
        ax1.set_title('GCN Model Performance: Accuracy and F1-Score vs. Threshold', fontsize=14)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels for F1-Score
        for i, (threshold, f1) in enumerate(zip(thresholds, f1_scores)):
            ax1.annotate(f'{f1:.3f}', (threshold, f1), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=8)
        
        # Add value labels for Accuracy
        for i, (threshold, acc) in enumerate(zip(thresholds, accuracies)):
            ax1.annotate(f'{acc:.3f}', (threshold, acc), textcoords="offset points", 
                        xytext=(0,-15), ha='center', fontsize=8)
        
        # Plot 2: AUC and F1-Score
        line3 = ax2.plot(thresholds, f1_scores, marker='o', linewidth=2, label='F1-Score (Weighted)', markersize=8, color='tab:blue')
        line4 = ax2.plot(thresholds, auc_scores, marker='^', linewidth=2, label='AUC', markersize=8, color='tab:orange')
        
        ax2.set_xlabel('Threshold Value', fontsize=12)
        ax2.set_ylabel('Performance Score', fontsize=12)
        ax2.set_title('GCN Model Performance: F1-Score and AUC vs. Threshold', fontsize=14)
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels for F1-Score
        for i, (threshold, f1) in enumerate(zip(thresholds, f1_scores)):
            ax2.annotate(f'{f1:.3f}', (threshold, f1), textcoords="offset points", 
                        xytext=(0,10), ha='center', fontsize=8)
        
        # Add value labels for AUC
        for i, (threshold, auc) in enumerate(zip(thresholds, auc_scores)):
            ax2.annotate(f'{auc:.3f}', (threshold, auc), textcoords="offset points", 
                        xytext=(0,-15), ha='center', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('threshold_comparison_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Threshold comparison chart saved as threshold_comparison_analysis.png")
        
        # Create a comprehensive plot with all metrics
        plot_comprehensive_threshold_comparison(results_df)
        
    except Exception as e:
        print(f"Error plotting threshold comparison: {e}")

def plot_comprehensive_threshold_comparison(results_df):
    """Plot comprehensive threshold comparison with key metrics"""
    try:
        # Set style
        plt.style.use('seaborn-v0_8')
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 4))
        
        # Extract data
        thresholds = results_df['Threshold']
        f1_scores = results_df['F1_Score_Weighted']
        accuracies = results_df['Accuracy']
        auc_scores = results_df['AUC']
        
        # Plot key metrics
        line1 = ax.plot(thresholds, f1_scores, marker='o', linewidth=2, label='F1-Score (Weighted)', markersize=8)
        line2 = ax.plot(thresholds, accuracies, marker='s', linewidth=2, label='Accuracy', markersize=8)
        line3 = ax.plot(thresholds, auc_scores, marker='^', linewidth=2, label='AUC', markersize=8)
        
        ax.set_xlabel('Threshold Value', fontsize=14)
        ax.set_ylabel('Performance Score', fontsize=14)
        ax.set_title('GCN Model Performance: Key Metrics vs. Threshold Values', fontsize=16, pad=20)
        # 将图例放置在左下角
        ax.legend(fontsize=12, loc='lower left')
        ax.grid(True, alpha=0.3)
        
        # 找到所有指标的最小值和最大值
        all_values = np.concatenate([f1_scores, accuracies, auc_scores])
        all_values = all_values[~np.isnan(all_values)]  # 移除NaN值
        
        if len(all_values) > 0:
            min_val = np.min(all_values)
            max_val = np.max(all_values)
            # 设置纵坐标范围，留出一些边距
            margin = (max_val - min_val) * 0.15  # 15%的边距
            ax.set_ylim(max(0, min_val - margin), min(1, max_val + margin))
        else:
            ax.set_ylim(0, 1)
        
        # 获取线条颜色
        color1 = line1[0].get_color()  # F1-Score颜色
        color2 = line2[0].get_color()  # Accuracy颜色
        color3 = line3[0].get_color()  # AUC颜色
        
        # 添加数值标签，颜色与线条保持一致
        for i, (threshold, f1, acc, auc) in enumerate(zip(thresholds, f1_scores, accuracies, auc_scores)):
            # 为每个指标添加标签，使用与线条一致的颜色
            if not np.isnan(f1):
                ax.annotate(f'{f1:.3f}', (threshold, f1), textcoords="offset points", 
                           xytext=(0, 15), ha='center', va='bottom', fontsize=8, color=color1)
            if not np.isnan(acc):
                ax.annotate(f'{acc:.3f}', (threshold, acc), textcoords="offset points", 
                           xytext=(0, 15), ha='center', va='bottom', fontsize=8, color=color2)
            if not np.isnan(auc):
                ax.annotate(f'{auc:.3f}', (threshold, auc), textcoords="offset points", 
                           xytext=(0, 15), ha='center', va='bottom', fontsize=8, color=color3)
        
        plt.tight_layout()
        plt.savefig('comprehensive_threshold_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Comprehensive threshold comparison chart saved as comprehensive_threshold_comparison.png")
        
    except Exception as e:
        print(f"Error plotting comprehensive threshold comparison: {e}")

def main():
    """Main function to run threshold comparison experiment"""
    try:
        results = compare_gcn_thresholds()
        print("\n" + "="*60)
        print("Threshold Comparison Experiment Completed")
        print("="*60)
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()