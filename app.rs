use std::collections::HashMap;
use std::path::PathBuf;
use tauri::api::dialog::{dialog_open, DialogFilter};
use tauri::api::http::{format_request, request, RequestHandler};
use tauri::api::process::Command;
use tauri::api::tauri::{State, StateManager};
use tauri::Manager;
use tauri::{CustomMenuItem, Menu, MenuItemAttributes, Submenu};

use yew::prelude::*;

struct App {
    link: ComponentLink<Self>,
    input: String,
    output: String,
    loading: bool,
}

enum Msg {
    UpdateInput(String),
    UpdateOutput(String),
    Submit,
    UploadFile,
    FileSelected(PathBuf),
}

impl Component for App {
    type Message = Msg;
    type Properties = ();

    fn create(_: Self::Properties, link: ComponentLink<Self>) -> Self {
        App {
            link,
            input: "".to_string(),
            output: "".to_string(),
            loading: false,
        }
    }

    fn update(&mut self, msg: Self::Message) -> ShouldRender {
        match msg {
            Msg::UpdateInput(val) => {
                self.input = val;
                true
            }
            Msg::UpdateOutput(val) => {
                self.output = val;
                true
            }
            Msg::Submit => {
                let input = self.input.clone();
                let manager = StateManager::<Manager>::new();
                let state = manager.state().unwrap();
                let mut data = HashMap::new();
                data.insert("input", input);
                self.loading = true;
                request(
                    &format_request("sort_input", Some(data)),
                    state.config().tauri.url.as_str(),
                    state.config().tauri.timeout,
                    RequestHandler::Json(move |response| {
                        let output = response
                            .map(|data| data.into_string().unwrap_or_default())
                            .unwrap_or_default();
                        App::run_on_main_thread(move |component| {
                            component.loading = false;
                            component.output = output;
                        });
                    }),
                );
                false
            }
            Msg::UploadFile => {
                let options = DialogFilter::new().name("CSV files").extension("csv");
                if let Some(dialog) = dialog_open(options) {
                    let manager = StateManager::<Manager>::new();
                    let state = manager.state().unwrap();
                    let mut data = HashMap::new();
                    data.insert("file_path", dialog.path().unwrap_or_default());
                    self.loading = true;
                    request(
                        &format_request("sort_file", Some(data)),
                        state.config().tauri.url.as_str(),
                        state.config().tauri.timeout,
                        RequestHandler::Json(move |response| {
                            let output = response
                                .map(|data| data.into_string().unwrap_or_default())
                                .unwrap_or_default();
                            App::run_on_main_thread(move |component| {
                                component.loading = false;
                                component.output = output;
                            });
                        }),
                    );
                }
                false
            }
            Msg::FileSelected(path) => {
                let input = std::fs::read_to_string(&path).unwrap_or_default();
                self.input = input;
                true
            }
        }
    }

    fn fn view(&self) -> Html {
        let loading_text = if self.loading {
            html! { <span>{"Loading..."}</span> }
        } else {
            html! {}
        };
        html! {
            <>
                <div>
                    <h1>{"FootageSorter"}</h1>
                    <p>{"test file: --QuoteLength (3-90) --finalLengthSeconds(3-1800), --keywords prompt or note thr content, and upload a .mp4 file to sort your footage."}</p>
                    <div>
                        <label>{"Input:"}</label>
                        <textarea
                            value=&self.input
                            oninput=self.link.callback(|e: InputData| Msg::UpdateInput(e.value))
                        />
                        <button onclick=self.link.callback(|_| Msg::Submit)>{ "Submit" }</button>
                        <button onclick=self.link.callback(|_| Msg::UploadFile)>{ "Upload file" }</button>
                    </div>
                    <div>
                        <label>{"Output:"}</label>
                        <textarea readonly=true value=&self.output />
                        { loading_text }
                    </div>
                </div>
            </>
        }
    }
}
