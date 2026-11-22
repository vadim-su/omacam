fn main() {
    if let Ok(home) = std::env::var("HOME") {
        let local_lib = format!("{home}/.local/lib");
        println!("cargo:rustc-link-search=native={local_lib}");
        println!("cargo:rustc-link-arg=-Wl,-rpath,{local_lib}");
    }
    println!("cargo:rustc-link-lib=dylib=ncnn");
    println!("cargo:rerun-if-changed=build.rs");
}
