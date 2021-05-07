use actix_web::{get, post, web, App, HttpServer, Responder};
use log::{error, info};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, time::Duration};
use std::sync::Mutex;
use std::time::Instant;
use structopt::StructOpt;
use web::Path;

use petgraph::stable_graph::StableGraph;

mod epoch_cache;
use epoch_cache::EpochCache;

mod flatten;
use flatten::*;

#[derive(StructOpt, Clone)]
struct Arguments {
    /// Application config file (JSON)
    appplication_config: String,
}

#[derive(Deserialize, Serialize, Debug)]
struct LogEntry {
    action: String,
    actual: u64,
}

#[derive(Serialize)]
struct Memory {
    memory: u64,
}

#[derive(Deserialize)]
struct Application {
    credentials: String,
    application_id: String,
    actions: HashMap<String, Action>,
}

#[derive(Deserialize)]
struct Action {
    memory: u64,
    actions: Vec<Transition>,
}

#[derive(Deserialize)]
struct Transition {
    action_name: String,
    probability: f32,
}

#[get("/{action}/memory")]
async fn get_memory(
    Path(action): Path<String>,
    application: web::Data<Application>,
) -> impl Responder {
    info!("Got memory request for action: {}", action);
    if let Some(app_action) = application.actions.get(&action) {
        info!(
            "Found memory for action, returning memory: {}",
            app_action.memory
        );
        web::Json(Memory {
            memory: app_action.memory,
        })
    } else {
        error!(
            "Did not find memory for action {}, returning default",
            action
        );
        web::Json(Memory { memory: 256 })
    }
}

#[post("/calls/{application_id}/{caller}/{callee}")]
async fn calls(
    Path((application_id, caller, callee)): Path<(String, String, String)>,
    graph: CallGraph,
) -> impl Responder {
    info!("Received: {} {} {}", application_id, caller, callee);

    let mut call_graph = graph.lock().unwrap();
    let (graph, indices) = &mut *call_graph;

    let from = *indices.entry(caller).or_insert_with_key(|caller| {
        let action = ActionInfo {
            action_name: caller.into(),
            invoke_count: 0,
            buffer: EpochCache::new(),
        };

        graph.add_node(action)
    });

    let to = *indices.entry(callee).or_insert_with_key(|callee| {
        let action = ActionInfo {
            action_name: callee.into(),
            invoke_count: 0,
            buffer: EpochCache::new(),
        };

        graph.add_node(action)
    });

    if graph[from].action_name == "entry_point" {
        graph[from].invoke_count += 1;
    }

    let edge = match graph.find_edge(from, to) {
        Some(index) => index,
        _ => graph.add_edge(from, to, EdgeInfo::default()),
    };

    graph[edge].call_count += 1;


    "Ok"
}

#[get("/experiment_1")]
async fn experiment_1(graph: CallGraph, times: AppTimes, exp_1: Experiment1) -> impl Responder {
    use flatten::{Node, flatten};
    use petgraph::visit::{IntoNodeIdentifiers};

    let call_graph = graph.lock().unwrap();
    let (graph, _) = &*call_graph;
    
    let mut ng = graph.map(|_, nw| {
        Node::N(nw.buffer.current())
    }, |edge_index, ew| {
        let (source, _) = graph.edge_endpoints(edge_index).unwrap();
        ew.call_count as f64 / graph[source].invoke_count as f64
    });

    let start = graph.node_identifiers().find(|&x| graph[x].action_name == "entry_point")
        .expect("Failed to find entry_point");
    let end = graph.node_identifiers().find(|&x| graph[x].action_name == "end_point")
        .expect("Failed to find end_point");

    let flatten_time = flatten(&mut ng, start, end);

    //println!("Graph time estimation: {}", flatten_time);
    //println!("EMA estimation: {}", times.lock().unwrap().durations.current());

    // estimation, actual
    format!("{},{}\n", flatten_time , exp_1.lock().unwrap().0)
}

#[get("/graph")]
async fn get_graph(graph: CallGraph, times: AppTimes) -> impl Responder {
    use petgraph::dot::*;
    use flatten::{Node, flatten};
    use petgraph::visit::{EdgeRef, IntoNodeIdentifiers};

    let call_graph = graph.lock().unwrap();
    let (graph, _) = &*call_graph;
    
    let mut ng = graph.map(|_, nw| {
        Node::N(nw.buffer.current())
    }, |edge_index, ew| {
        let (source, _) = graph.edge_endpoints(edge_index).unwrap();
        ew.call_count as f64 / graph[source].invoke_count as f64
    });

    let start = graph.node_identifiers().find(|&x| graph[x].action_name == "entry_point")
        .expect("Failed to find entry_point");
    let end = graph.node_identifiers().find(|&x| graph[x].action_name == "end_point")
        .expect("Failed to find end_point");

    let flatten_time = flatten(&mut ng, start, end);

    println!("Graph time estimation: {}", flatten_time);
    println!("EMA estimation: {}", times.lock().unwrap().durations.current());
    

    let gv = Dot::with_attr_getters(
        graph,
        &[Config::EdgeNoLabel, Config::NodeNoLabel],
        &|g, edge| {
            let parent_invoke = g[edge.source()].invoke_count as f64;
            format!(
                "label = \"{:.2}\"",
                edge.weight().call_count as f64 / parent_invoke
            )
        },
        &|_g, (_, node)| {
            format!(
                "label = \"{}: {}\"",
                node.action_name.clone(),
                node.invoke_count
            )
        },
    );

    format!("{}", gv)
}

#[post("/logs")]
async fn post_log(
    mut entry: web::Json<LogEntry>,
    graph: CallGraph,
    application: web::Data<Application>,
    times: AppTimes,
    exp_1: Experiment1
) -> impl Responder {

    if entry.action == "run_application" {
        entry.action = "entry_point".into();
    }

    if !application.actions.contains_key(&entry.action) {
        info!("Ignoring action: {}", entry.action);
        return "Ok";
    }
    
    if entry.action == "entry_point" {
        info!("Logging start of application");
        times.lock().unwrap().current_sum = 1;
    } else if entry.action == "end_point" {
        let mut times = times.lock().unwrap();
        let duration = times.current_sum;
        if duration == 0 {
            info!("Ignoring duplicate invocation");
        } else {
            times.durations.add(duration);
            println!("Logging application latency of: {}", duration);
            times.current_sum = 0;
            exp_1.lock().unwrap().0 = duration;
        }
    } else {
        if times.lock().unwrap().current_sum > 0 {
            times.lock().unwrap().current_sum += entry.actual;
        } else {
            info!("Ignoring invocation before start");
        }
    }

    info!("Received: {:?}", *entry);

    let mut call_graph = graph.lock().unwrap();
    let (graph, indices) = &mut *call_graph;
    let index = if !indices.contains_key(&entry.action) {
        let action = ActionInfo {
            action_name: entry.action.clone(),
            invoke_count: 0,
            buffer: EpochCache::new(),
        };

        let index = graph.add_node(action);
        indices.insert(entry.action.clone(), index);
        index
    } else {
        indices[&entry.action]
    };

    graph[index].invoke_count += 1;
    graph[index].buffer.add(entry.actual);

    "Ok"
}

type NodeIndicies = HashMap<String, petgraph::prelude::NodeIndex>;
type CallGraph = web::Data<Mutex<(StableGraph<ActionInfo, EdgeInfo>, NodeIndicies)>>;
type AppTimes = web::Data<Mutex<Times>>;
type Experiment1 = web::Data<Mutex<Experiment1Container>>;

struct Times {
    current_sum: u64,
    durations: EpochCache
}

fn app_to_call_graph(app: &Application, file_name: &str) {
    let mut graph = StableGraph::new();
    let mut index_map = HashMap::new();

    for (name, _) in &app.actions {
        let index = graph.add_node(name);
        index_map.insert(name, index);
    }

    for (name, action) in &app.actions {
        let from_index = index_map[name];
        for edge in &action.actions {
            let to_index = index_map[&edge.action_name];
            let w = edge.probability;

            graph.add_edge(from_index, to_index, w);
        }
    }

    use petgraph::dot::*;

    let gv = Dot::with_attr_getters(
        &graph,
        &[Config::EdgeNoLabel, Config::NodeNoLabel],
        &|_, edge| {
            format!(
                "label = \"{:.2}\"",
                edge.weight()
            )
        },
        &|_g, (_, node)| {
            format!(
                "label = \"{}\"",
                node
            )
        },
    );

    std::fs::write(file_name, gv.to_string()).unwrap();
}

struct Experiment1Container(u64);


#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let args = Arguments::from_args();
    let port = 8000;

    let app: Application =
        serde_json::from_str(&std::fs::read_to_string(args.appplication_config)?).unwrap();

    app_to_call_graph(&app, "original.dot");

    let app = web::Data::new(app);
    let times = web::Data::new(Mutex::new(Times { 
        current_sum: 0,
        durations: EpochCache::new()
    } ));
    

    let call_graph: CallGraph = web::Data::new(Mutex::new((StableGraph::new(), HashMap::new())));


    let experiment_1_data = web::Data::new(Mutex::new(Experiment1Container(0)));

    env_logger::Builder::from_default_env().init();
    println!("Listening on http://0.0.0.0:{}/logs", port);

    HttpServer::new(move || {
        App::new()
            .service(post_log)
            .service(calls)
            .service(get_memory)
            .service(get_graph)
            .service(experiment_1)
            .app_data(app.clone())
            .app_data(call_graph.clone())
            .app_data(times.clone())
            .app_data(experiment_1_data.clone())
    })
    .bind(("0.0.0.0", port))?
    .run()
    .await
}
