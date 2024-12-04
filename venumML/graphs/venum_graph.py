import numpy as np
from venumML.venumpy import small_glwe as vp
import hashlib
import networkx as nx
from venumML.venum_tools import *

import random
import pandas as pd

class Graph:
    """
    Represents a Venum encrypted graph that supports encryption and decryption 
    of the adjacency matrix for privacy-preserving computations.

    Attributes
    ----------
    use_hashing : bool
        Indicates whether to use hashing for node identifiers.
    nodes : dict
        Maps hashed node identifiers to integer indices.
    directed : bool
        Specifies if the graph is directed.
    adjacency_matrix : np.ndarray
        Encrypted adjacency matrix representing edge weights.
    boolean_matrix : np.ndarray
        Encrypted adjacency matrix representing the presence of edges.
    inverse_outbound : np.ndarray
        Encrypted array of inverse outbound degree counts for each node.
    boolean_outbound : np.ndarray
        Encrypted boolean array indicating nodes with outbound edges.
    """

    def __init__(self, ctx, directed=True, use_hashing=True):
        """
        Initialises an encrypted graph with optional hashing for node identifiers.

        Parameters
        ----------
        ctx : EncryptionContext
            The encryption context used to initialise matrices.
        directed : bool, optional, default=True
            Whether the graph is directed.
        use_hashing : bool, optional, default=True
            Whether to use SHA-256 hashing for node identifiers.
        """

        self._use_hashing = use_hashing
        self._nodes = {}  # Map node hashes to integer indices
        self._directed = directed
        self._adjacency_matrix = encrypt_array(np.zeros((0, 0)),ctx) # Initialise an empty adjacency matrix
        self._boolean_matrix = encrypt_array(np.zeros((0, 0),dtype=int),ctx) # Initialise an empty adjacency matrix
        self._inverse_outbound = None 
        self._boolean_outbound = None 

    @staticmethod
    def hash_node(node):
        """
        Hashes a node using SHA-256.

        Parameters
        ----------
        node : any
            The node identifier.

        Returns
        -------
        str
            The SHA-256 hexadecimal digest of the node.
        """

        node_str = str(node)
        hash_object = hashlib.sha256(node_str.encode())
        return hash_object.hexdigest()

    def is_sha256_hash(self, value):
        """
        Checks if the given value is a SHA-256 hash.

        Parameters
        ----------
        value : str
            The value to check.

        Returns
        -------
        bool
            True if the value is a valid SHA-256 hash, False otherwise.
        """

        value = str(value)
        if len(value) != 64:
            return False
        try:
            int(value, 16)  # Attempt to convert the string to an integer with base 16
            return True
        except ValueError:
            return False

    def add_node(self, ctx, node):
        """
        Adds a node to the graph after hashing and ensures the adjacency matrix is resized.

        Parameters
        ----------
        ctx : EncryptionContext
            The encryption context used to encrypt values.
        node : any
            The node identifier.
        """

        if self._use_hashing:
            node = self._hash_node(node) if not self._is_sha256_hash(node) else node

        if node not in self._nodes:
            # Assign an index to the new node
            new_index = len(self._nodes)
            self._nodes[node] = new_index

            # Resize the adjacency matrix to accommodate the new node
            current_size = self._adjacency_matrix.shape[0]
            new_matrix = encrypt_array(np.zeros((current_size + 1, current_size + 1)),ctx)
            new_matrix_bool= new_matrix.copy()
            new_matrix[:current_size, :current_size] = self._adjacency_matrix
            new_matrix_bool[:current_size, :current_size] = self._boolean_matrix
            self._adjacency_matrix = new_matrix
            self._boolean_matrix = new_matrix_bool

    def add_edge(self,ctx, from_node, to_node, weight=1):
        """
        Adds a weighted edge between two nodes.

        Parameters
        ----------
        ctx : EncryptionContext
            The encryption context used to encrypt values.
        from_node : any
            The starting node of the edge.
        to_node : any
            The ending node of the edge.
        weight : float, optional, default=1
            The weight of the edge.
        """

        if self._use_hashing:
            from_node = self._hash_node(from_node) if not self._is_sha256_hash(from_node) else from_node
            to_node = self._hash_node(to_node) if not self._is_sha256_hash(to_node) else to_node

        # Add nodes to ensure they exist
        self.add_node(ctx,from_node)
        self.add_node(ctx,to_node)

        # Get indices for the nodes
        from_index = self._nodes[from_node]
        to_index = self._nodes[to_node]
        if ctx:
            weight = ctx.encrypt(weight)
        # Update adjacency matrix with the edge weight
        self._adjacency_matrix[from_index, to_index] = weight
        self._boolean_matrix[from_index, to_index] = ctx.encrypt(1)

        if not self._directed:
            # For undirected graphs, mirror the edge
            self._adjacency_matrix[to_index, from_index] = weight
            self._boolean_matrix[to_index, from_index] = ctx.encrypt(1)

    def remove_edge(self, from_node, to_node):
        """
        Removes an edge between two nodes by setting the matrix value to 0.

        Parameters
        ----------
        from_node : any
            The starting node of the edge.
        to_node : any
            The ending node of the edge.
        """
        
        if self._use_hashing:
            from_node = self._hash_node(from_node) if not self._is_sha256_hash(from_node) else from_node
            to_node = self._hash_node(to_node) if not self._is_sha256_hash(to_node) else to_node

        from_index = self._nodes.get(from_node, None)
        to_index = self._nodes.get(to_node, None)

        if from_index is not None and to_index is not None:
            self._adjacency_matrix[from_index, to_index] = 0
            if not self._directed:
                self._adjacency_matrix[to_index, from_index] = 0

    def get_edge_weight(self, from_node, to_node):
        """
        Returns the weight of an edge between two nodes.

        Parameters
        ----------
        from_node : any
            The starting node of the edge.
        to_node : any
            The ending node of the edge.

        Returns
        -------
        float or int
            The weight of the edge, or 0 if no edge exists.
        """
        
        if self._use_hashing:
            from_node = self._hash_node(from_node) if not self._is_sha256_hash(from_node) else from_node
            to_node = self._hash_node(to_node) if not self._is_sha256_hash(to_node) else to_node

        from_index = self._nodes.get(from_node, None)
        to_index = self._nodes.get(to_node, None)

        if from_index is not None and to_index is not None:
            return self._adjacency_matrix[from_index, to_index]
        return 0

    def get_adjacency_matrix(self):
        """
        Retrieves the encrypted adjacency matrix.

        Returns
        -------
        np.ndarray
            The current encrypted adjacency matrix.
        """

        return self._adjacency_matrix
    def get_boolean_matrix(self):
        """
        Retrieves the encrypted boolean adjacency matrix.

        Returns
        -------
        np.ndarray
            The current encrypted boolean adjacency matrix.
        """

        return self._boolean_matrix

    def get_node_degree(self, node):
        """
        Calculates the degree of a node.

        Parameters
        ----------
        node : any
            The node identifier.

        Returns
        -------
        int
            The degree of the node.
        """

        if self._use_hashing:

            node = self._hash_node(node) if not self._is_sha256_hash(node) else node
            node_index = self._nodes.get(node, None)

        if node_index is None:
            return 0

        if self._directed:
            # For directed graphs, degree is the sum of incoming and outgoing edges
            incoming_degree = np.sum(self._boolean_matrix[:, node_index])
            outgoing_degree = np.sum(self._boolean_matrix[node_index, :])
            return incoming_degree + outgoing_degree
        else:
            # For undirected graphs, degree is the sum of edges connected to the node
            return np.sum(self._boolean_matrix[node_index, :])

def encrypt_networkx(ctx, nx_graph,use_hashing=True):
    """
    Converts a NetworkX graph to an encrypted custom Graph.

    Parameters
    ----------
    ctx : EncryptionContext
        The encryption context used to encrypt node and edge data.
    nx_graph : networkx.Graph or networkx.DiGraph
        The NetworkX graph to convert.
    use_hashing : bool, optional, default=True
        Whether to hash node identifiers.

    Returns
    -------
    Graph
        An encrypted custom Graph with nodes, edges, and outbound counts encrypted.
    """
    
    # Determine if the input graph is directed and use that to initialize the custom Graph
    directed = nx_graph.is_directed()
    custom_graph = Graph(ctx, directed=directed,use_hashing=use_hashing)
    
    # Convert NetworkX adjacency matrix to numpy array
    adj_matrix = nx.to_numpy_array(nx_graph)
    
    # Calculate outbound counts and their inverses
    outbound_degree = adj_matrix.sum(axis=1)
    inverse_outbound = np.zeros_like(outbound_degree, dtype=float)
    non_zero_outbound = outbound_degree != 0
    inverse_outbound[non_zero_outbound] = 1 / outbound_degree[non_zero_outbound]
    
    # Encrypt outbound counts and their boolean representation
    inverse_outbound = encrypt_array(inverse_outbound, ctx)
    boolean_outbound = encrypt_array(outbound_degree.astype(bool).astype(int), ctx)
    
    # Assign encrypted outbound counts to the custom graph
    custom_graph.boolean_outbound = boolean_outbound
    custom_graph.inverse_outbound = inverse_outbound
    
    # Add nodes to the custom graph
    nodes = list(nx_graph.nodes())
    for node in nodes:
        custom_graph.add_node(ctx, node)
    
    # Add edges to the custom graph, using the weight attribute if available, otherwise defaulting to 1
    edges = list(nx_graph.edges(data=True))
    for from_node, to_node, attrs in edges:
        weight = attrs.get('weight', 1)  # Use the 'weight' attribute from attrs if present
        custom_graph.add_edge(ctx, from_node, to_node, weight)
    
    return custom_graph

def df_to_encrypted_graph(ctx, df, from_col, to_col, weight_col = None, use_hashing = True, directed = True):
    """
    Creates an encrypted custom Graph from a pandas DataFrame.

    Parameters
    ----------
    ctx : EncryptionContext
        The encryption context used to encrypt node and edge data.
    df : pd.DataFrame
        DataFrame containing the edge data.
    from_col : str
        Column name representing the source node.
    to_col : str
        Column name representing the target node.
    weight_col : str
        Column name representing the edge weight.
    use_hashing : bool, optional, default=True
        Whether to hash node identifiers.

    Returns
    -------
    Graph
        An encrypted custom Graph with nodes, edges, and outbound counts encrypted.
    """
    
    # Initialize the custom Graph
    custom_graph = Graph(ctx, directed=directed,use_hashing=use_hashing)
    
    # Calculate outbound counts and their inverses from the DataFrame
    if weight_col is not None:
        outbound_degree = df.groupby(from_col)[weight_col].sum().reindex(df[from_col].unique(), fill_value=0).values
    else:
        # For unweighted graphs, count number of edges per node
        outbound_degree = df.groupby(from_col).size().reindex(df[from_col].unique(), fill_value=0).values
    inverse_outbound = np.zeros_like(outbound_degree, dtype=float)
    non_zero_outbound = outbound_degree != 0
    inverse_outbound[non_zero_outbound] = 1 / outbound_degree[non_zero_outbound]     
    # Encrypt outbound counts and their boolean representation
    inverse_outbound = encrypt_array(inverse_outbound, ctx)
    boolean_outbound = encrypt_array(outbound_degree.astype(bool).astype(int), ctx)
    
    # Assign encrypted outbound counts to the custom graph
    custom_graph.boolean_outbound = boolean_outbound
    custom_graph.inverse_outbound = inverse_outbound
    
    # Add nodes to the custom graph
    nodes = pd.concat([df[from_col], df[to_col]]).unique()
    for node in nodes:
        custom_graph.add_node(ctx, node)
    
    # Add edges to the custom graph
    for _, row in df.iterrows():
        from_node = row[from_col]
        to_node = row[to_col]
        if weight_col is None:
            weight = 1
        else:
            weight = row[weight_col]
        custom_graph.add_edge(ctx, from_node, to_node, weight)
    
    return custom_graph

def decrypt_graph(ctx, graph):
    """
    Decrypts an encrypted adjacency matrix of an encrypted Graph.

    Parameters
    ----------
    ctx : EncryptionContext
        The encryption context used to decrypt values.
    custom_graph : Graph
        The encrypted Graph to decrypt.

    Returns
    -------
    networkx.Graph or networkx.DiGraph
        A decrypted NetworkX graph with nodes and edges restored.
    """

    # Determine if the custom graph is directed and use that to initialize the NetworkX graph
    directed = graph.directed
    nx_graph = nx.DiGraph() if directed else nx.Graph()
    
    # Add nodes
    for hashed_node in graph.nodes.keys():
        nx_graph.add_node(hashed_node)
    
    # Add edges, decrypting the weights
    for from_node, from_index in graph.nodes.items():
        for to_node, to_index in graph.nodes.items():
            weight = graph.adjacency_matrix[from_index, to_index]
            if weight != 0:
                decrypted_weight = ctx.decrypt(weight) if ctx else weight
                nx_graph.add_edge(from_node, to_node, weight=decrypted_weight)
    
    return nx_graph

def pagerank(ctx,encrypted_graph,damping_factor=0.85,iters=20):
    """
    Computes PageRank scores for an encrypted Graph using the Power Method.

    Parameters
    ----------
    ctx : EncryptionContext
        The encryption context used to handle encrypted values.
    encrypted_graph : Graph
        The encrypted custom Graph on which to compute PageRank.
    damping_factor : float, optional, default=0.85
        The probability of following an edge; 1 - damping_factor is the teleport probability.
    iters : int, optional, default=20
        The number of iterations for the Power Method.

    Returns
    -------
    dict
        A dictionary mapping node identifiers to encrypted PageRank scores.
    """

    adj_matrix = encrypted_graph._adjacency_matrix
    bool_matrix = encrypted_graph._boolean_matrix
    N = adj_matrix.shape[0]
    inverse_outbound = encrypted_graph.inverse_outbound
    boolean_outbound = encrypted_graph.boolean_outbound
     # Initialize the PageRank scores to a uniform distribution
    pagerank_scores = encrypt_array(np.full(N, 1/ N),ctx)

    # Convert adjacency matrix to transition probability matrix
    dangling_norm = encrypt_array(np.full(N, 1/ N),ctx)
    reverse_bool = encrypt_array(np.ones_like(boolean_outbound),ctx)-boolean_outbound

    transition_matrix = (adj_matrix * inverse_outbound[:, np.newaxis]) * boolean_outbound[:, np.newaxis] + dangling_norm * reverse_bool[:, np.newaxis]

    teleportation = encrypt_array(np.full(N, 1/ N),ctx)
    
    df = ctx.encrypt(damping_factor)
    reverse_df = ctx.encrypt(1) - df
    for iteration in range(iters):

        # PageRank formula with damping factor
        pagerank_scores = (
            np.matmul(transition_matrix.T, pagerank_scores)*df
            +  teleportation*reverse_df
        )
    # Reverse the nodes dictionary to map indices back to node hashes
    reversed_nodes = {index: node for node, index in encrypted_graph._nodes.items()}
    pagerank_dict = {reversed_nodes[i]: pagerank_scores[i] for i in range(N)}
    return pagerank_dict


def decrypt_pagerank(ctx, encrypted_pagerank):
    """
    Decrypts encrypted PageRank scores.

    Parameters
    ----------
    ctx : EncryptionContext
        The encryption context used to decrypt values.
    encrypted_pagerank : dict
        A dictionary mapping node identifiers to encrypted PageRank scores.

    Returns
    -------
    dict
        A dictionary mapping node identifiers to decrypted PageRank scores.
    """

    decrypted_pagerank = {node: ctx.decrypt(score) for node, score in encrypted_pagerank.items()}
    return decrypted_pagerank