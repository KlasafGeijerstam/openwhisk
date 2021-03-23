use actix_web::{get, post, web, App, HttpServer, Responder};
use log::{error, info};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Mutex;
use structopt::StructOpt;
use web::Path;

use petgraph::graph::DiGraph;

mod epoch_cache;
use epoch_cache::EpochCache;

mod flatten;
use flatten::*;

#[derive(StructOpt, Clone)]
struct Arguments {
    /// The port to listen on
    port: u16,
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

    let edge = match graph.find_edge(from, to) {
        Some(index) => index,
        _ => graph.add_edge(from, to, EdgeInfo::default()),
    };

    graph[edge].call_count += 1;

    "Ok"
}

#[get("/graph")]
async fn get_graph(graph: CallGraph) -> impl Responder {
    use petgraph::dot::*;
    use petgraph::visit::EdgeRef;

    let call_graph = graph.lock().unwrap();
    let (graph, _) = &*call_graph;

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
    entry: web::Json<LogEntry>,
    graph: CallGraph,
    application: web::Data<Application>,
) -> impl Responder {
    if !application.actions.contains_key(&entry.action) {
        info!("Ignoring action: {}", entry.action);
        return "Ok";
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
type CallGraph = web::Data<Mutex<(DiGraph<ActionInfo, EdgeInfo>, NodeIndicies)>>;

#[allow(unused)]
fn app_to_call_graph(app: &Application) -> CallGraph {
    let mut graph = DiGraph::new();
    let mut index_map = HashMap::new();

    for (name, action) in &app.actions {
        let action = ActionInfo {
            action_name: name.into(),
            invoke_count: 0,
            buffer: EpochCache::new(),
        };
        let index = graph.add_node(action);
        index_map.insert(name.to_owned(), index);
    }

    for (name, action) in &app.actions {
        let from_index = index_map[name];
        for edge in &action.actions {
            let to_index = index_map[&edge.action_name];
            let w = EdgeInfo { call_count: 0 };

            graph.add_edge(from_index, to_index, w);
        }
    }

    web::Data::new(Mutex::new((graph, index_map)))
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let args = Arguments::from_args();
    let port = args.port;

    let app: Application =
        serde_json::from_str(&std::fs::read_to_string(args.appplication_config)?).unwrap();

    let app = web::Data::new(app);

    let call_graph: CallGraph = web::Data::new(Mutex::new((DiGraph::new(), HashMap::new())));

    env_logger::Builder::from_default_env().init();
    println!("Listening on http://0.0.0.0:{}/logs", port);

    HttpServer::new(move || {
        App::new()
            .service(post_log)
            .service(calls)
            .service(get_memory)
            .service(get_graph)
            .app_data(app.clone())
            .app_data(call_graph.clone())
    })
    .bind(("0.0.0.0", port))?
    .run()
    .await
}
