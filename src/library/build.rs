use std::env;

fn main() {
    println!("cargo::rerun-if-env-changed=DOCS_RS");
    println!("cargo::rustc-check-cfg=cfg(axis_docs_rs)");

    if env::var_os("DOCS_RS").is_some() {
        println!("cargo::rustc-cfg=axis_docs_rs");
    }
}
