# =================================================
# Graph Building Module
# =================================================
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import kneighbors_graph

class GraphBuilder:
    def __init__(self):
        pass
    
    def build_threshold_graph(self, X, threshold=0.8):
        """Build graph based on threshold"""
        similarity_matrix = cosine_similarity(X)
        adj_matrix = (similarity_matrix > threshold).astype(float)
        np.fill_diagonal(adj_matrix, 0)
        return adj_matrix, similarity_matrix
    
    def build_knn_graph(self, X, k=5):
        """K-nearest neighbor graph"""
        adj_matrix = kneighbors_graph(X, k, mode='connectivity', include_self=False)
        return adj_matrix.toarray(), cosine_similarity(X)
    
    def build_topk_graph(self, X, k=5):
        """Top-K similarity graph"""
        similarity_matrix = cosine_similarity(X)
        adj_matrix = np.zeros_like(similarity_matrix)
        
        for i in range(similarity_matrix.shape[0]):
            indices = np.argsort(similarity_matrix[i])[::-1][1:k+1]
            adj_matrix[i, indices] = 1
            adj_matrix[indices, i] = 1
        
        np.fill_diagonal(adj_matrix, 0)
        return adj_matrix, similarity_matrix
    
    def build_adaptive_threshold_graph(self, X, percentile=80):
        """Adaptive threshold graph"""
        similarity_matrix = cosine_similarity(X)
        threshold = np.percentile(similarity_matrix, percentile)
        print(f"Adaptive threshold ({percentile} percentile): {threshold:.4f}")
        
        adj_matrix = (similarity_matrix > threshold).astype(float)
        np.fill_diagonal(adj_matrix, 0)
        return adj_matrix, similarity_matrix

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
    
    def build_knn_graph_with_weights(self, X, k=5):
        """K-nearest neighbor graph with weights"""
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=k+1, metric='cosine').fit(X)
        distances, indices = nbrs.kneighbors(X)
        
        # Build adjacency matrix
        n_samples = X.shape[0]
        adj_matrix = np.zeros((n_samples, n_samples))
        
        # Calculate weights based on distance (smaller distance means higher weight)
        for i in range(n_samples):
            for j in range(1, k+1):  # Skip self
                neighbor_idx = indices[i, j]
                distance = distances[i, j]
                # Use Gaussian kernel to calculate weight
                weight = np.exp(-distance**2)
                adj_matrix[i, neighbor_idx] = weight
                adj_matrix[neighbor_idx, i] = weight  # Undirected graph
        
        similarity_matrix = cosine_similarity(X)
        return adj_matrix, similarity_matrix
    
    def build_adaptive_threshold_graph(self, X, percentile=80):
        """Adaptive threshold graph"""
        similarity_matrix = cosine_similarity(X)
        # Calculate percentile of non-zero similarities
        non_zero_similarities = similarity_matrix[similarity_matrix > 0]
        if len(non_zero_similarities) > 0:
            threshold = np.percentile(non_zero_similarities, percentile)
        else:
            threshold = 0.5
            
        print(f"Adaptive threshold ({percentile} percentile): {threshold:.4f}")
        
        adj_matrix = np.where(similarity_matrix > threshold, similarity_matrix, 0)
        np.fill_diagonal(adj_matrix, 0)
        return adj_matrix, similarity_matrix
    
    def build_mixed_graph(self, X, k=5, threshold_percentile=80):
        """Mixed graph construction method"""
        # Get KNN graph
        knn_adj, _ = self.build_knn_graph_with_weights(X, k)
        
        # Get adaptive threshold graph
        threshold_adj, similarity_matrix = self.build_adaptive_threshold_graph(X, threshold_percentile)
        
        # Combine both methods
        combined_adj = np.maximum(knn_adj, threshold_adj)
        return combined_adj, similarity_matrix
    
    def build_random_graph(self, X, edge_probability=0.1):
        """Random graph without using similarity matrix"""
        n_samples = X.shape[0]
        adj_matrix = np.random.choice([0, 1], size=(n_samples, n_samples), 
                                    p=[1-edge_probability, edge_probability])
        # Make it undirected
        adj_matrix = np.triu(adj_matrix, k=1)  # Take upper triangle
        adj_matrix = adj_matrix + adj_matrix.T  # Make symmetric
        np.fill_diagonal(adj_matrix, 0)
        
        # Create a simple similarity-like matrix (just for compatibility)
        similarity_matrix = np.random.rand(n_samples, n_samples)
        similarity_matrix = (similarity_matrix + similarity_matrix.T) / 2
        np.fill_diagonal(similarity_matrix, 1)
        
        return adj_matrix, similarity_matrix
    
    def build_isolated_nodes_graph(self, X, **kwargs):
        """
        Build a graph where each node is isolated (no edges between nodes)
        This creates a graph structure where each sample is an isolated node
        """
        n_samples = X.shape[0]
        
        # Create adjacency matrix with no connections (identity matrix with zeros)
        adj_matrix = np.zeros((n_samples, n_samples))
        
        # Create similarity matrix (can be zeros or identity)
        similarity_matrix = np.zeros((n_samples, n_samples))
        
        return adj_matrix, similarity_matrix

    def build_random_graph_with_min_degree(self, X, edge_probability=0.1, min_degree=1):
        """Build random graph and ensure minimum degree"""
        n_samples = X.shape[0]
        adj_matrix = np.random.choice([0, 1], size=(n_samples, n_samples), 
                                    p=[1-edge_probability, edge_probability])
        # Make it undirected
        adj_matrix = np.triu(adj_matrix, k=1)
        adj_matrix = adj_matrix + adj_matrix.T
        np.fill_diagonal(adj_matrix, 0)
        
        # Ensure no isolated nodes
        node_degrees = np.sum(adj_matrix, axis=1)
        isolated_nodes = np.where(node_degrees == 0)[0]
        
        # Randomly connect isolated nodes to other nodes
        for node in isolated_nodes:
            # Randomly select a different node to connect to
            other_node = np.random.choice([i for i in range(n_samples) if i != node])
            adj_matrix[node, other_node] = 1
            adj_matrix[other_node, node] = 1
        
        # Create similarity matrix
        similarity_matrix = np.random.rand(n_samples, n_samples)
        similarity_matrix = (similarity_matrix + similarity_matrix.T) / 2
        np.fill_diagonal(similarity_matrix, 1)
        
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