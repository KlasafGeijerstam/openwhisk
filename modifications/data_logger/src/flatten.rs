use crate::{epoch_cache::EpochCache, CallGraph, NodeIndicies};
use std::fmt;

use petgraph::algo::all_simple_paths;
use petgraph::graph::Frozen;
use petgraph::prelude::NodeIndex;
use petgraph::stable_graph::EdgeReference;
use petgraph::{algo::*, visit::IntoNodeReferences};
use petgraph::{stable_graph::StableGraph, visit::EdgeRef, Directed};

pub type Graph = StableGraph<Node, f64>;

#[derive(Default, Debug)]
pub struct ActionInfo {
    pub action_name: String,
    pub invoke_count: usize,
    pub buffer: EpochCache,
}

impl fmt::Display for ActionInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

#[derive(Default, Debug)]
pub struct EdgeInfo {
    pub call_count: usize,
}

impl fmt::Display for EdgeInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

#[derive(Clone, PartialEq, Debug)]
pub enum Node {
    N(u64),
    B(Vec<(Node, f64)>),
    P(u64),
}

use Node::*;

pub fn has_branch(
    from_node: NodeIndex,
    to_node: NodeIndex,
    graph: &Graph,
    max_length: usize,
) -> bool {
    let paths = all_simple_paths::<Vec<_>, _>(&graph, from_node, to_node, 1, Some(max_length));

    let mut sum = 0.0;
    for p in paths {
        let mut tmp_sum = 1.0;
        for (&from, &to) in p.iter().zip(p.iter().skip(1)) {
            let edge_index = graph.find_edge(from, to).unwrap();
            tmp_sum *= graph[edge_index];
        }

        sum += tmp_sum;
    }
    sum == 1.0
}

pub fn remove_branch(
    from_node: NodeIndex,
    to_node: NodeIndex,
    graph: &mut Graph,
    max_length: usize,
) {
    let tmp = graph.clone();

    // get all simple paths
    let paths: Vec<_> =
        all_simple_paths::<Vec<NodeIndex>, _>(&tmp, from_node, to_node, 1, Some(max_length))
            .collect();

    // TODO: should be set?
    let mut v: Vec<(Node, f64)> = Vec::new();

    // merge all nodes that are neither from_node and to_node into a single node
    for p in &paths {
        let mut added: bool = false;
        for (&from, &to) in p.iter().zip(p.iter().skip(1)) {
            let edge_index = graph.find_edge(from, to).unwrap();

            match &graph[to] {
                B(node_v) => {
                    v.extend(node_v.clone());
                }
                w => {
                    v.push((w.clone(), graph[edge_index]));
                }
            }
            added = true;
        }
        if added {
            v.pop();
        }
    }

    // remove edges leading to nodes that will be merged
    for p in &paths {
        for (&from, &to) in p.iter().zip(p.iter().skip(1)) {
            let edge_index = graph.find_edge(from, to).unwrap();
            graph.remove_edge(edge_index);
        }
    }

    // remove branching nodes
    for p in paths {
        for &node_index in &p[1..p.len() - 1] {
            graph.remove_node(node_index);
        }
    }

    // create B1 node and add edges
    let b = graph.add_node(Node::B(v));
    graph.add_edge(from_node, b, 1.0);
    graph.add_edge(b, to_node, 1.0);
}

pub fn remove_branches(graph: &mut Graph) {
    let tmp = graph.clone();

    for path_length in 1..graph.node_count() {
        for from_node in tmp.node_indices() {
            for to_node in tmp.node_indices() {
                if has_branch(from_node, to_node, graph, path_length) {
                    remove_branch(from_node, to_node, graph, path_length);
                }
            }
        }
    }
}

pub fn remove_self_loops(graph: &mut Graph) {
    let tmp = graph.clone();

    // remove self-loops
    for node in tmp.node_indices() {
        // - find node with self-loop
        if graph.contains_edge(node, node) {
            // - remove the edge that causes self-loop
            let e_index = graph.find_edge(node, node).unwrap();
            let e_weight = graph[e_index];
            graph.remove_edge(e_index);

            // Update other edges with new probability
            for edge in tmp.edges(node) {
                graph.update_edge(
                    edge.source(),
                    edge.target(),
                    edge.weight() / (1.0 - e_weight),
                );
            }

            // - add the additional execution time to the node according to paper
            if let Node::N(node_weight) = &mut graph[node] {
                *node_weight += ((e_weight / (1.0 - e_weight)) * (*node_weight as f64)) as u64;
            }
        }
    }
}

pub fn has_parallel(
    from_node: NodeIndex,
    to_node: NodeIndex,
    graph: &mut Graph,
    max_length: usize,
) -> bool{
    let tmp = graph.clone();
    let paths = all_simple_paths::<Vec<_>, _>(&tmp, from_node, to_node, 1, Some(max_length));

    let mut sum = 0.0;
    for p in paths {
        let mut tmp_sum = 1.0;
        for (&from, &to) in p.iter().zip(p.iter().skip(1)) {
            let edge_index = graph.find_edge(from, to).unwrap();
            tmp_sum *= graph[edge_index];
        }

        sum += tmp_sum;
    }
    sum > 1.0
}
pub fn remove_parallel(
    from_node: NodeIndex,
    to_node: NodeIndex,
    graph: &mut Graph,
    max_length: usize,
) {

    // find the simple path with the longest execution time
    
}

pub fn remove_parallels(graph: &mut Graph) {
    let tmp = graph.clone();

    for path_length in 1..graph.node_count() {
        for from_node in tmp.node_indices() {
            for to_node in tmp.node_indices() {
                if has_parallel(from_node, to_node, graph, path_length) {
                    remove_parallel(from_node, to_node, graph, path_length);
                }
            }
        }
    }

}

#[test]
fn test_removing_branches() {
    let mut graph: Graph = StableGraph::new();

    let f2 = graph.add_node(N(320));
    let f3 = graph.add_node(N(260));
    let f4 = graph.add_node(N(840));
    let f6 = graph.add_node(N(150));

    graph.add_edge(f2, f3, 0.7);
    graph.add_edge(f2, f4, 0.3);
    graph.add_edge(f3, f6, 1.0);
    graph.add_edge(f4, f6, 1.0);

    remove_branches(&mut graph);

    assert_eq!(graph.node_count(), 3);
    assert_eq!(graph.edge_count(), 2);

    for node in graph.node_indices() {
        if node != f2 && node != f6 {
            if let B(v) = &graph[node] {
                assert_eq!(v.len(), 2);
                assert_eq!(v, &[(N(840), 0.3), (N(260), 0.7)]);
            }
        }
    }
}

#[test]
fn test_removing_self_loops() {
    let mut graph: Graph = StableGraph::new();

    let a = graph.add_node(N(520));
    let b = graph.add_node(N(150));

    graph.add_edge(a, a, 0.2);
    graph.add_edge(a, b, 0.8);

    remove_self_loops(&mut graph);

    assert_eq!(graph[a], N(650));
}

pub fn flatten(indices: NodeIndicies, graph: &mut Graph) {
    let new_graph: Graph = StableGraph::new();

    // While graph has cycles or branches
    while is_cyclic_directed(&new_graph)
        || all_simple_paths::<Vec<NodeIndex>, _>(
            &new_graph,
            indices["entry_point"],
            indices["end_point"],
            1,
            None,
        )
        .count()
            > 1
    {

        // remove cycles

        // remove parallels
        // remove branches
        // if has_branch(graph,
    }
}
