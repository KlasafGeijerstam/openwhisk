use crate::{epoch_cache::EpochCache};
use std::fmt;

use petgraph::algo::{all_simple_paths, astar, has_path_connecting};
use petgraph::prelude::NodeIndex;
use petgraph::{stable_graph::StableGraph, visit::EdgeRef};

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

impl Node {
    fn delay(&self) -> u64 {
        match self {
            B(data) => data
                .iter()
                .map(|(n, p)| (n.delay() as f64 * p) as u64)
                .sum(),
            other => other.delay(),
        }
    }
}

use Node::*;

pub fn has_n_branch_paths(
    from_node: NodeIndex,
    to_node: NodeIndex,
    graph: &Graph,
    path_count: usize,
) -> bool {
    let paths: Vec<_> =
        all_simple_paths::<Vec<_>, _>(&graph, from_node, to_node, 1, None).collect();

    if paths.len() != path_count {
        return false;
    }

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

pub fn remove_branch(from_node: NodeIndex, to_node: NodeIndex, graph: &mut Graph) {
    let tmp = graph.clone();

    // get all simple paths
    let paths: Vec<_> =
        all_simple_paths::<Vec<NodeIndex>, _>(&tmp, from_node, to_node, 1, None).collect();

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

pub fn has_n_parallel_paths(
    from_node: NodeIndex,
    to_node: NodeIndex,
    graph: &Graph,
    path_count: usize,
) -> bool {
    let tmp = graph.clone();
    let paths: Vec<_> = all_simple_paths::<Vec<_>, _>(&tmp, from_node, to_node, 1, None).collect();

    if graph.edges(from_node).map(|e| e.weight()).sum::<f64>() <= 1.0 || paths.len() != path_count {
        return false;
    }

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

pub fn simple_path_count(graph: &Graph, from: NodeIndex, to: NodeIndex) -> usize {
    all_simple_paths::<Vec<_>, _>(graph, from, to, 1, None)
        .collect::<Vec<_>>()
        .len()
}

pub fn remove_parallel(from_node: NodeIndex, to_node: NodeIndex, graph: &mut Graph) {
    let tmp = graph.clone();
    let paths: Vec<_> = all_simple_paths::<Vec<_>, _>(&tmp, from_node, to_node, 1, None).collect();

    let (_, len) = paths
        .iter()
        .map(|path| {
            (
                path,
                path.iter()
                    .zip(path.iter().skip(1))
                    .fold(0.0, |acc, (&from, &to)| {
                        if acc != 0.0 {
                            // Remove intermideary nodes
                            graph.remove_node(from);
                        }

                        let edge = graph.find_edge(from, to).unwrap();
                        let edge_probability = graph[edge];
                        acc + (graph[to].delay() as f64 * edge_probability)
                    }) as u64,
            )
        })
        .max_by_key(|(_, length)| *length)
        .unwrap();

    for edge in tmp.edges(from_node) {
        graph.remove_edge(edge.id());
    }

    let new_node = graph.add_node(P(len));
    graph.add_edge(from_node, new_node, 1.0);
    graph.add_edge(new_node, to_node, 1.0);
}

pub fn remove_parallels(graph: &mut Graph) {
    let tmp = graph.clone();
}


fn remove_cycle(node: NodeIndex, graph: &mut Graph) {
    let g = graph.clone();
    let (_, mut path) = astar(&g, node, |n| n == node, |_| 0, |_| 0)
        .expect("Could not find a path (cycle) starting with given node");

    let &last_node = path.last().unwrap();

    path.insert(0, node);
    path.push(node);

    let cycle_probability: f64 = path
        .iter()
        .zip(path.iter().skip(1))
        .map(|(from, to)| {
            let edge = graph.find_edge(*from, *to).unwrap();
            graph[edge]
        })
        .product();

    let cycle_cost: u64 = path.iter().map(|n| graph[*n].delay()).sum();

    // - remove the edge that causes self-loop
    let e_index = graph.find_edge(last_node, node).unwrap();
    let e_weight = graph[e_index];
    graph.remove_edge(e_index);

    // Update other edges with new probability
    for edge in g.edges(last_node) {
        graph.update_edge(
            edge.source(),
            edge.target(),
            edge.weight() / (1.0 - e_weight),
        );
        
    }

    // TODO: Backtrack and remove first non-1.0 edge (and subsequent nodes)

    // - add the additional execution time to the node according to paper
    graph[last_node] = P(graph[last_node].delay()
        + ((cycle_probability / (1.0 - cycle_probability)) * (cycle_cost as f64)) as u64);
}

pub fn flatten(graph: &mut Graph) {
    remove_self_loops(graph);

    let mut processed = true;

    'outer: while processed {
        processed = false;

        let tmp = graph.clone();
        for path_count in 2..tmp.node_count() {
            for from_node in tmp.node_indices() {
                for to_node in tmp.node_indices() {
                    if has_n_parallel_paths(from_node, to_node, graph, path_count) {
                        remove_parallel(from_node, to_node, graph);
                        processed = true;
                        continue 'outer;
                    }

                    if has_n_branch_paths(from_node, to_node, graph, path_count) {
                        remove_branch(from_node, to_node, graph);
                        processed = true;
                        continue 'outer;
                    }
                }
            }
        }

        for node in tmp.node_indices() {
            if has_path_connecting(&tmp, node, node, None) {
                remove_cycle(node, graph);

                processed = true;
                continue 'outer;
            }
        }
    }
}

#[test]
fn test_removing_cycles() {
    let mut graph = StableGraph::new();

    let f1 = graph.add_node(N(100));
    let f2 = graph.add_node(N(50));
    let f3 = graph.add_node(N(200));

    graph.add_edge(f1, f2, 1.0);
    graph.add_edge(f2, f3, 0.2);
    graph.add_edge(f2, f1, 0.8);
    
    for i in 0..10 {
        assert_eq!(has_n_branch_paths(f1, f2, &graph, i), false);
        assert_eq!(has_n_parallel_paths(f1, f2, &graph, i), false);
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

    remove_branch(f2, f6, &mut graph);

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
