import os
import json
import numpy as np
import networkx as nx
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime

from infrastructure.utils.logger import logger
from infrastructure.config import load_paths, get_project_root


class KnowledgeGraph:
    """Graph-based representation of fraud patterns and relationships"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.paths = load_paths()
        self.graph_file = os.path.join(
            get_project_root(), 
            "experts/coordination/shared_context/knowledge_graph.json"
        )
        
        # Load existing graph if available
        self._load_graph()
        logger.info("Knowledge graph initialized")
    
    def _load_graph(self) -> None:
        """Load knowledge graph from file"""
        if os.path.exists(self.graph_file):
            try:
                with open(self.graph_file, 'r') as f:
                    graph_data = json.load(f)
                
                # Recreate graph from JSON
                self.graph = nx.node_link_graph(graph_data)
                logger.info(f"Loaded knowledge graph with {len(self.graph.nodes)} nodes")
            except Exception as e:
                logger.error(f"Error loading knowledge graph: {str(e)}")
                self._initialize_empty_graph()
        else:
            self._initialize_empty_graph()
    
    def _initialize_empty_graph(self) -> None:
        """Initialize an empty knowledge graph with base structure"""
        self.graph.clear()
        
        # Add base node types
        self.graph.add_node("ROOT", type="root", description="Root node of knowledge graph")
        
        # Add category nodes
        categories = ["USER", "MERCHANT", "COUNTRY", "PATTERN", "RULE"]
        for category in categories:
            self.graph.add_node(category, type="category", description=f"{category} category")
            self.graph.add_edge("ROOT", category, relation="has_category")
        
        logger.info("Initialized empty knowledge graph")
    
    def _save_graph(self) -> None:
        """Save knowledge graph to file"""
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.graph_file), exist_ok=True)
        
        # Convert to serializable format
        graph_data = nx.node_link_data(self.graph)
        
        with open(self.graph_file, 'w') as f:
            json.dump(graph_data, f, indent=2)
        
        logger.info(f"Saved knowledge graph with {len(self.graph.nodes)} nodes")
    
    def add_transaction(self, transaction: Dict[str, Any], 
                        is_fraud: bool = False) -> None:
        """Add transaction information to knowledge graph"""
        transaction_id = transaction.get('id', f"TX_{datetime.now().isoformat()}")
        
        # Add transaction node
        tx_node = f"TX_{transaction_id}"
        self.graph.add_node(
            tx_node, 
            type="transaction",
            timestamp=datetime.now().isoformat(),
            amount=transaction.get('amount', 0),
            is_fraud=is_fraud
        )
        
        # Link to user if exists
        user_id = transaction.get('user_id')
        if user_id:
            user_node = f"USER_{user_id}"
            if not self.graph.has_node(user_node):
                self.graph.add_node(user_node, type="user", first_seen=datetime.now().isoformat())
                self.graph.add_edge("USER", user_node, relation="has_instance")
            
            self.graph.add_edge(user_node, tx_node, relation="made_transaction")
        
        # Link to merchant if exists
        merchant = transaction.get('merchant')
        if merchant:
            merchant_node = f"MERCHANT_{merchant}"
            if not self.graph.has_node(merchant_node):
                self.graph.add_node(merchant_node, type="merchant", first_seen=datetime.now().isoformat())
                self.graph.add_edge("MERCHANT", merchant_node, relation="has_instance")
            
            self.graph.add_edge(tx_node, merchant_node, relation="paid_to")
        
        # Link to country if exists
        country = transaction.get('country')
        if country:
            country_node = f"COUNTRY_{country}"
            if not self.graph.has_node(country_node):
                self.graph.add_node(country_node, type="country", first_seen=datetime.now().isoformat())
                self.graph.add_edge("COUNTRY", country_node, relation="has_instance")
            
            self.graph.add_edge(tx_node, country_node, relation="occurred_in")
        
        # Add other transaction attributes as needed
        
        # Save updated graph
        self._save_graph()
    
    def add_fraud_pattern(self, pattern_name: str, 
                         attributes: Dict[str, Any],
                         related_entities: List[str] = None) -> None:
        """Add a fraud pattern to the knowledge graph"""
        pattern_node = f"PATTERN_{pattern_name}"
        
        # Add pattern node
        self.graph.add_node(
            pattern_node,
            type="pattern",
            discovered=datetime.now().isoformat(),
            **attributes
        )
        
        # Link to pattern category
        self.graph.add_edge("PATTERN", pattern_node, relation="has_instance")
        
        # Link to related entities
        if related_entities:
            for entity in related_entities:
                if self.graph.has_node(entity):
                    self.graph.add_edge(pattern_node, entity, relation="involves")
        
        # Save updated graph
        self._save_graph()
    
    def add_rule(self, rule_name: str, rule_definition: Dict[str, Any]) -> None:
        """Add fraud detection rule to knowledge graph"""
        rule_node = f"RULE_{rule_name}"
        
        # Add rule node
        self.graph.add_node(
            rule_node,
            type="rule",
            created=datetime.now().isoformat(),
            **rule_definition
        )
        
        # Link to rule category
        self.graph.add_edge("RULE", rule_node, relation="has_instance")
        
        # Link to related patterns if specified
        if "related_patterns" in rule_definition:
            for pattern in rule_definition["related_patterns"]:
                pattern_node = f"PATTERN_{pattern}"
                if self.graph.has_node(pattern_node):
                    self.graph.add_edge(rule_node, pattern_node, relation="detects")
        
        # Save updated graph
        self._save_graph()
    
    def get_entity_relationships(self, entity_id: str, 
                                max_depth: int = 2) -> Dict[str, Any]:
        """Get relationships for a specific entity"""
        if not self.graph.has_node(entity_id):
            return {"error": f"Entity {entity_id} not found in knowledge graph"}
        
        # Get subgraph up to max_depth
        neighbors = set()
        current = {entity_id}
        
        for _ in range(max_depth):
            next_nodes = set()
            for node in current:
                # Add outgoing edges
                next_nodes.update(self.graph.neighbors(node))
                # Add incoming edges
                next_nodes.update(self.graph.predecessors(node))
            
            neighbors.update(next_nodes)
            current = next_nodes - neighbors
        
        # Create subgraph
        subgraph = self.graph.subgraph(neighbors.union({entity_id}))
        
        # Convert to serializable format
        return {
            "entity_id": entity_id,
            "entity_type": self.graph.nodes[entity_id].get("type", "unknown"),
            "relationships": nx.node_link_data(subgraph)
        }
    
    def find_connections(self, entity1: str, entity2: str, 
                        max_paths: int = 3) -> List[List[str]]:
        """Find connections between two entities"""
        if not (self.graph.has_node(entity1) and self.graph.has_node(entity2)):
            return []
        
        try:
            # Find all simple paths between entities
            paths = list(nx.all_simple_paths(
                self.graph, 
                source=entity1, 
                target=entity2, 
                cutoff=5  # Limit path length
            ))
            
            # Return limited number of paths
            return [list(path) for path in paths[:max_paths]]
        except:
            # Fallback to shortest path if all_simple_paths fails
            try:
                path = nx.shortest_path(self.graph, source=entity1, target=entity2)
                return [path]
            except:
                return []
    
    def get_high_risk_entities(self, entity_type: str = None, 
                              min_connections: int = 3) -> List[Dict[str, Any]]:
        """Get high-risk entities based on connections to fraud"""
        high_risk = []
        
        # Get all fraud transactions
        fraud_txs = [node for node, data in self.graph.nodes(data=True) 
                    if data.get("type") == "transaction" and data.get("is_fraud", False)]
        
        # Get all entities of the specified type
        if entity_type:
            entities = [node for node, data in self.graph.nodes(data=True) 
                       if data.get("type") == entity_type]
        else:
            # Filter for user, merchant, country nodes
            entities = [node for node, data in self.graph.nodes(data=True) 
                       if data.get("type") in ["user", "merchant", "country"]]
        
        # Count connections to fraud transactions
        for entity in entities:
            fraud_connections = 0
            
            for fraud_tx in fraud_txs:
                if nx.has_path(self.graph, entity, fraud_tx) or nx.has_path(self.graph, fraud_tx, entity):
                    fraud_connections += 1
            
            if fraud_connections >= min_connections:
                high_risk.append({
                    "entity_id": entity,
                    "entity_type": self.graph.nodes[entity].get("type"),
                    "fraud_connections": fraud_connections
                })
        
        # Sort by number of fraud connections
        return sorted(high_risk, key=lambda x: x["fraud_connections"], reverse=True)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph"""
        node_types = {}
        edge_types = {}
        
        # Count node types
        for _, data in self.graph.nodes(data=True):
            node_type = data.get("type", "unknown")
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
        # Count edge types
        for _, _, data in self.graph.edges(data=True):
            relation = data.get("relation", "unknown")
            edge_types[relation] = edge_types.get(relation, 0) + 1
        
        return {
            "total_nodes": len(self.graph.nodes),
            "total_edges": len(self.graph.edges),
            "node_types": node_types,
            "edge_types": edge_types,
            "density": nx.density(self.graph)
        }