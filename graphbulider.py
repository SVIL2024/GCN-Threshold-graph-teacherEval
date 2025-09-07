# =================================================
# Graph Building Module
# =================================================
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import kneighbors_graph

class EnhancedGraphBuilder:
    def __init__(self):
        pass
    
    def build_threshold_graph(self, X, threshold=0.8):
        """Build graph based on threshold - improved version"""
        similarity_matrix = cosine_similarity(X)
        # Use soft threshold instead of hard threshold
        adj_matrix = np.where(similarity_matrix > threshold, similarity_matrix, 0)
        np.fill_diagonal(adj_matrix, 0)
        return adj_matrix, similarity_matrix
    

    def visualize_graph(self, adj_matrix, similarity_matrix, title="Graph Visualization", max_nodes=50):
        """Visualize graph structure"""
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
            
            # Limit number of nodes to avoid overly complex graphs
            n_nodes = adj_matrix.shape[0]
            if n_nodes > max_nodes:
                # Only show first max_nodes nodes
                adj_matrix = adj_matrix[:max_nodes, :max_nodes]
                n_nodes = max_nodes
                print(f"Number of nodes exceeds {max_nodes}, only showing first {max_nodes} nodes")
            
            # Create graph object
            G = nx.from_numpy_array(adj_matrix)
            
            # Set figure size
            plt.figure(figsize=(12, 10))
            
            # Calculate node layout
            pos = nx.spring_layout(G, seed=42)
            
            # Draw nodes
            nx.draw_networkx_nodes(G, pos, node_size=300, node_color='lightblue', alpha=0.7)
            
            # Draw edges with transparency based on weights
            edges = G.edges()
            weights = [similarity_matrix[i, j] if i < similarity_matrix.shape[0] and j < similarity_matrix.shape[1] 
                    else 0 for i, j in edges]
            
            # Normalize weights for transparency
            if weights:
                min_weight = min(weights)
                max_weight = max(weights)
                if max_weight > min_weight:
                    alphas = [(w - min_weight) / (max_weight - min_weight) * 0.8 + 0.2 for w in weights]
                else:
                    alphas = [0.5] * len(weights)
            else:
                alphas = [0.5] * len(edges)
            
            # Draw edges
            for (u, v), alpha in zip(edges, alphas):
                nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], alpha=alpha, edge_color='gray')
            
            # Draw node labels
            labels = {i: str(i) for i in range(n_nodes)}
            nx.draw_networkx_labels(G, pos, labels, font_size=8)
            
            plt.title(title)
            plt.axis('off')
            plt.tight_layout()
            
            # Save figure
            filename = f"{title.replace(' ', '_').replace('(', '').replace(')', '')}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"Graph saved as {filename}")
            
            # Output graph statistics
            print(f"Graph statistics:")
            print(f"  Number of nodes: {G.number_of_nodes()}")
            print(f"  Number of edges: {G.number_of_edges()}")
            print(f"  Average degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")
            if G.number_of_edges() > 0:
                print(f"  Average weight: {np.mean([similarity_matrix[i, j] for i, j in G.edges()]):.4f}")
            
        except ImportError:
            print("Missing visualization dependencies, please install matplotlib and networkx")
        except Exception as e:
            print(f"Error during visualization: {e}")

    def build_fully_connected_graph(self, X):
        """Build fully connected graph"""
        n_samples = X.shape[0]
        
        # Create fully connected adjacency matrix
        adj_matrix = np.ones((n_samples, n_samples))
        np.fill_diagonal(adj_matrix, 0)
        
        # Create similarity matrix (using cosine similarity)
        from sklearn.metrics.pairwise import cosine_similarity
        similarity_matrix = cosine_similarity(X)
        np.fill_diagonal(similarity_matrix, 0)  # Set diagonal to 0
        
        return adj_matrix, similarity_matrix