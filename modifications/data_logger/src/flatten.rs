use crate::epoch_cache::EpochCache;
use std::fmt;

use petgraph::algo::all_simple_paths;
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

impl fmt::Display for Node {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(
            &match self {
                N(n) => format!("N({})", n),
                B(_) => format!("B({})", self.delay()),
                P(n) => format!("P({})", n),
            },
            f,
        )
    }
}

impl Node {
    fn delay(&self) -> u64 {
        match self {
            B(data) => data
                .iter()
                .map(|(n, p)| (n.delay() as f64 * p) as u64)
                .sum(),
            N(a) => *a,
            P(a) => *a,
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

    if paths.len() != path_count || graph.edges(from_node).count() < 2 || !graph.edges(from_node).any(
        |e| *e.weight() < 1.0) {
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
    (sum - 1.0).abs() <= f64::EPSILON
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
            if let Some(edge_index) = graph.find_edge(from, to) {
                graph.remove_edge(edge_index);
            }
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

            let tmp = graph.clone();
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
                        println!("from: {:?},  to: {:?}", from, to);
                        let edge_probability = if let Some(edge) = graph.find_edge(from, to) {
                            graph[edge]
                        } else {
                            panic!("aaah")
                        };

                        acc + (graph[to].delay() as f64 * edge_probability)
                    }) as u64,
            )
        })
        .max_by_key(|(_, length)| *length)
        .unwrap();


    for edge in tmp.edges(from_node) {
        graph.remove_edge(edge.id());
    }

    let len = len - graph[to_node].delay();


    let new_node = graph.add_node(P(len));
    graph.add_edge(from_node, new_node, 1.0);
    graph.add_edge(new_node, to_node, 1.0);
}

pub fn find_cyclic_path(path: &mut Vec<NodeIndex>, graph: &Graph) -> bool {
    for edge in graph.edges(*path.last().unwrap()) {
        if edge.target() == path[0] {
            return true;
        }

        if !path.contains(&edge.target()) {
            path.push(edge.target());

            if find_cyclic_path(path, graph) {
                return true;
            } else {
                path.pop();
            }
        }
    }
    false
}

fn remove_cycle(node: NodeIndex, graph: &mut Graph) -> bool {
    let mut path = vec![node];
    if !find_cyclic_path(&mut path, &graph) {
        return false;
    }

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

    for (from, to) in path.iter().zip(path.iter().skip(1)).rev() {
        let e_index = graph.find_edge(*from, *to).unwrap();

        if graph[e_index] < 1.0 {
            let e_weight = graph[e_index];
            graph.remove_edge(e_index);

            // Update other edges with new probability
            let g = graph.clone();
            for edge in g.edges(*from) {
                graph.update_edge(
                    edge.source(),
                    edge.target(),
                    edge.weight() / (1.0 - e_weight),
                );
            }
            graph[*from] = P(graph[*from].delay()
                + ((cycle_probability / (1.0 - cycle_probability)) * (cycle_cost as f64)) as u64);

            return true;
        }
    }

    panic!("remove_cycle");
}

pub fn flatten(graph: &mut Graph, start: NodeIndex, end: NodeIndex) -> u64 {
    remove_self_loops(graph);
    //println!("remove_self_loops");
    //println!("{}", get_graph(&graph));

    let mut processed = true;

    'outer: while processed {
        processed = false;

        let tmp = graph.clone();
        for node in tmp.node_indices() {
            if remove_cycle(node, graph) {
                //println!("remove_cycle");
                //println!("{}", get_graph(&graph));

                processed = true;
                continue 'outer;
            }
        }
        for path_count in 2..tmp.node_count() {
            for from_node in tmp.node_indices() {
                for to_node in tmp.node_indices() {
                    if has_n_parallel_paths(from_node, to_node, graph, path_count) {
                        remove_parallel(from_node, to_node, graph);
                        processed = true;
                        //println!("remove_parallel {} {} {}", from_node.index(), to_node.index(), path_count);
                        //println!("{}", get_graph(&graph));
                        continue 'outer;
                    }

                    if has_n_branch_paths(from_node, to_node, graph, path_count) {
                        remove_branch(from_node, to_node, graph);
                        processed = true;
                        //println!("remove_branch {} {} {}", from_node.index(), to_node.index(), path_count);
                        //println!("{}", get_graph(&graph));
                        continue 'outer;
                    }
                }
            }
        }

    }

    prune_graph(graph, start, end);
    //println!("{}", get_graph(&graph));
    let paths: Vec<_> = all_simple_paths::<Vec<_>, _>(&graph.clone(), start, end, 1, None).collect();
    if paths.len() != 1 {
        println!("ERROR: Simple path count from start to end is: {}", paths.len());
        0
    } else {
        paths[0].iter().map(|&n| graph[n].delay()).sum()
    }
}

pub fn prune_graph(graph: &mut Graph, start: NodeIndex, end: NodeIndex) {
    let paths: Vec<_> = all_simple_paths::<Vec<_>, _>(&graph.clone(), start, end, 1, None).collect();
    graph.retain_nodes(|_, node| paths[0].contains(&node));
}

pub fn get_graph(graph: &Graph) -> String {
    use petgraph::dot::*;

    let gv = Dot::with_attr_getters(
        graph,
        &[Config::EdgeNoLabel, Config::NodeNoLabel],
        &|_, edge| format!("label = \"{:.2}\"", edge.weight()),
        &|_, (i, node)| format!("label = \"{} {}\"", node, i.index()),
    );

    format!("{}", gv)
}
/*
#[test]
fn test_slapp_example() {
    let mut graph = StableGraph::new();

    let f1 = graph.add_node(N(560));
    let f2 = graph.add_node(N(320));
    let f3 = graph.add_node(N(260));
    let f4 = graph.add_node(N(840));
    let f5 = graph.add_node(N(520));
    let f6 = graph.add_node(N(150));
    let f7 = graph.add_node(N(430));

    graph.add_edge(f1, f2, 1.0);
    graph.add_edge(f1, f5, 1.0);

    graph.add_edge(f2, f3, 0.7);
    graph.add_edge(f2, f4, 0.3);

    graph.add_edge(f3, f6, 1.0);
    graph.add_edge(f4, f6, 1.0);

    graph.add_edge(f5, f5, 0.2);
    graph.add_edge(f5, f6, 0.8);

    graph.add_edge(f6, f1, 0.1);
    graph.add_edge(f6, f7, 0.9);

    println!("{}", get_graph(&graph));
    flatten(&mut graph);
    //println!("{}", get_graph(&graph));
}
*/

#[test]
fn test_graph_app16() {
    let mut graph = StableGraph::new();

    let start = graph.add_node(N(0));
    let f1 = graph.add_node(N(1));
    let f2 = graph.add_node(N(2));
    let f3 = graph.add_node(N(3));
    let f4 = graph.add_node(N(4));
    let f5 = graph.add_node(N(5));
    let f6 = graph.add_node(N(6));
    let f7 = graph.add_node(N(7));
    let f8 = graph.add_node(N(8));
    let f9 = graph.add_node(N(9));
    let f10 = graph.add_node(N(10));
    let f11 = graph.add_node(N(11));
    let f12 = graph.add_node(N(12));
    let f13 = graph.add_node(N(13));
    let f14 = graph.add_node(N(14));
    let f15 = graph.add_node(N(15));
    let f16 = graph.add_node(N(16));

    let end = graph.add_node(N(17));

    graph.add_edge(start, f1, 1.0);
    graph.add_edge(f8, end, 1.0);


    graph.add_edge(f1, f2, 1.0);
    graph.add_edge(f1, f3, 1.0);

    graph.add_edge(f2, f4, 0.6);
    graph.add_edge(f2, f5, 0.4);


    graph.add_edge(f4, f7, 1.0);

    graph.add_edge(f5, f11, 1.0);
    graph.add_edge(f5, f12, 1.0);

    graph.add_edge(f11, f7, 1.0);
    graph.add_edge(f12, f13, 1.0);


    graph.add_edge(f13, f14, 1.0);

    graph.add_edge(f14, f13, 0.3);
    graph.add_edge(f14, f7, 0.7);

    graph.add_edge(f3, f10, 0.2);
    graph.add_edge(f3, f9, 0.8);

    graph.add_edge(f10, f6, 1.0);

    graph.add_edge(f9, f15, 1.0);

    graph.add_edge(f15, f15, 0.05);
    graph.add_edge(f15, f16, 0.95);

    graph.add_edge(f16, f6, 1.0);

    graph.add_edge(f6, f3, 0.1);
    graph.add_edge(f6, f7, 0.9);

    graph.add_edge(f7, f7, 0.2);
    graph.add_edge(f7, f8, 0.8);

    println!("{}", get_graph(&graph));
    flatten(&mut graph, start, end);
    //println!("{}", get_graph(&graph));
}
/*
#[test]
fn test_graph() {
    let mut graph = StableGraph::new();

    let f1 = graph.add_node(N(1));
    let f2 = graph.add_node(N(2));
    let f3 = graph.add_node(N(3));
    let f4 = graph.add_node(N(4));
    let f5 = graph.add_node(N(5));
    let f6 = graph.add_node(N(6));

    graph.add_edge(f1, f3, 0.8);
    graph.add_edge(f1, f5, 0.2);
    graph.add_edge(f1, f2, 1.0);

    graph.add_edge(f3, f4, 1.0);
    graph.add_edge(f5, f4, 1.0);
    graph.add_edge(f2, f4, 1.0);

    graph.add_edge(f4, f1, 0.1);
    graph.add_edge(f4, f4, 0.2);
    graph.add_edge(f4, f6, 0.7);

    println!("{}", get_graph(&graph));
    flatten(&mut graph);
    //println!("{}", get_graph(&graph));
}
*/

/*
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

    assert_eq!(graph.edge_count(), 3);
    remove_cycle(f1, &mut graph);

    assert_eq!(graph.node_count(), 3);
    assert_eq!(graph.edge_count(), 2);
    let edge_index = graph.find_edge(f2, f3).unwrap();
    assert_eq!((graph[edge_index] - 1.0).abs() <= f64::EPSILON, true);
    assert_eq!(graph.edges(f2).count(), 1);
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
*/
