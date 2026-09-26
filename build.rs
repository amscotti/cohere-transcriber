fn main() {
    // This binary targets Apple Silicon only: the MLX backend needs Metal.
    // Fail fast on other platforms with a clear message instead of obscure
    // Metal-linker errors.
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos")
        || std::env::var("CARGO_CFG_TARGET_ARCH").as_deref() != Ok("aarch64")
    {
        panic!(
            "cohere-transcriber requires Apple Silicon (aarch64 macOS): \
             the MLX backend links Metal/Accelerate and only runs there."
        );
    }
    if std::env::var("CARGO_FEATURE_MLX").is_ok() {
        build_mlx();
    } else {
        panic!("cohere-transcribe has no non-MLX backend: build with default features.");
    }
}

/// Build the vendored mlx-c tree via CMake and link it statically.
/// mlx-c fetches and builds the MLX C++ library, producing static libraries
/// and a compiled Metal shader library (mlx.metallib).
///
/// No Homebrew, no dylibbundler, no DYLD_LIBRARY_PATH — fully self-contained.
fn build_mlx() {
    let mlx_c_dir = std::path::PathBuf::from("mlx-c");
    if !mlx_c_dir.join("CMakeLists.txt").exists() {
        panic!(
            "mlx-c sources not found at ./mlx-c. \
             This tree vendors ml-explore/mlx-c v0.6.0; restore that directory. \
             See README.md."
        );
    }

    // Ensure CMake and Rust agree on the macOS deployment target.
    // Without this, CMake may compile C++ for macOS 15.x while Rust links
    // for macOS 11.0, causing `___isPlatformVersionAtLeast` linker errors.
    // The same variable is also exported to rustc so the final binary's
    // minimum OS matches the MLX objects (MLX requires macOS >= 14).
    let deployment_target =
        std::env::var("MACOSX_DEPLOYMENT_TARGET").unwrap_or_else(|_| "14.0".to_string());
    println!("cargo:rerun-if-env-changed=MACOSX_DEPLOYMENT_TARGET");
    // rustc does not read MACOSX_DEPLOYMENT_TARGET from `cargo:rustc-env`
    // (that only sets `env!` inside the crate). Pass the floor to the linker
    // so the binary's minimum OS matches the MLX objects.
    println!("cargo:rustc-link-arg=-mmacosx-version-min={deployment_target}");

    // Build mlx-c via CMake (fetches and builds MLX C++ as a dependency)
    let dst = cmake::Config::new(&mlx_c_dir)
        .define("MLX_BUILD_TESTS", "OFF")
        .define("MLX_BUILD_EXAMPLES", "OFF")
        .define("MLX_BUILD_BENCHMARKS", "OFF")
        // mlx-c's own examples default to ON and cost several extra
        // executables in an already slow build.
        .define("MLX_C_BUILD_EXAMPLES", "OFF")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("CMAKE_OSX_DEPLOYMENT_TARGET", &deployment_target)
        .build();

    // Link paths — CMake may output to lib/ or lib64/
    let lib_dir = dst.join("lib");
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    let lib64_dir = dst.join("lib64");
    if lib64_dir.exists() {
        println!("cargo:rustc-link-search=native={}", lib64_dir.display());
    }

    // Link mlx-c and mlx static libraries
    println!("cargo:rustc-link-lib=static=mlxc");
    println!("cargo:rustc-link-lib=static=mlx");

    // Apple system frameworks required by MLX
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=Accelerate");
    println!("cargo:rustc-link-lib=framework=MetalPerformanceShaders");

    // C++ standard library (MLX is C++)
    println!("cargo:rustc-link-lib=c++");

    // MLX C++ code uses @available() checks that reference ___isPlatformVersionAtLeast
    // from the compiler runtime. Rust passes -nodefaultlibs to the linker, so the
    // compiler runtime is not automatically linked. We must explicitly link it.
    // Find the clang resource directory to locate libclang_rt.osx.a.
    let clang_rt = std::process::Command::new("clang")
        .args(["--print-file-name", "libclang_rt.osx.a"])
        .output()
        .expect("failed to run clang --print-file-name");
    let clang_rt_path = String::from_utf8(clang_rt.stdout)
        .expect("non-utf8 clang output")
        .trim()
        .to_string();
    if std::path::Path::new(&clang_rt_path).exists() {
        println!("cargo:rustc-link-arg={}", clang_rt_path);
    } else {
        // A silent skip here resurfaces as an obscure undefined-symbol error
        // at link time. Point the linker at clang's resource lib directory
        // and fall back to -lclang_rt.osx before giving up.
        println!(
            "cargo:warning=libclang_rt.osx.a not found via `clang --print-file-name` \
             (got {:?}); falling back to the toolchain's resource library dir",
            clang_rt_path
        );
        let resource = std::process::Command::new("clang")
            .args(["--print-resource-dir"])
            .output();
        let mut linked = false;
        if let Ok(out) = resource {
            let dir = String::from_utf8_lossy(&out.stdout).trim().to_string();
            let libdir = std::path::Path::new(&dir).join("lib").join("darwin");
            if libdir.join("libclang_rt.osx.a").exists() {
                println!("cargo:rustc-link-search=native={}", libdir.display());
                println!("cargo:rustc-link-lib=static=clang_rt.osx");
                linked = true;
            }
        }
        if !linked {
            println!(
                "cargo:warning=could not locate libclang_rt.osx.a; expect linker errors about \
                 ___isPlatformVersionAtLeast"
            );
        }
    }

    // Copy mlx.metallib next to the output binaries so the MLX runtime
    // can find it at execution time.  MLX looks for the metallib in the
    // same directory as the running executable before falling back to the
    // compile-time METAL_PATH constant (which points into the cmake build
    // tree and won't exist after deployment).
    //
    // OUT_DIR is e.g. target/release/build/<pkg>-<hash>/out — go up 3
    // levels to reach the profile directory (target/release/) where Cargo
    // places the final binaries.
    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR not set by Cargo");
    let target_dir = std::path::Path::new(&out_dir)
        .parent() // build/<pkg>-<hash>/
        .and_then(|p| p.parent()) // build/
        .and_then(|p| p.parent()) // target/<profile>/
        .expect("cannot derive target dir from OUT_DIR");

    // Search the cmake build tree for mlx.metallib
    if let Some(metallib) = find_file(&dst, "mlx.metallib") {
        let dest = target_dir.join("mlx.metallib");
        std::fs::copy(&metallib, &dest).unwrap_or_else(|e| {
            panic!(
                "Failed to copy {} → {}: {}",
                metallib.display(),
                dest.display(),
                e
            )
        });
    } else {
        println!("cargo:warning=mlx.metallib not found in cmake build tree — Metal GPU ops will fail at runtime");
    }

    // Rebuild when the C sources change. The rest of mlx-c (docs, examples,
    // the Python generator) is not compiled into the binary.
    println!("cargo:rerun-if-changed=mlx-c/CMakeLists.txt");
    println!("cargo:rerun-if-changed=mlx-c/mlx");
    println!("cargo:rerun-if-changed=build.rs");
}

/// Recursively search for a file by name under a directory.
fn find_file(dir: &std::path::Path, name: &str) -> Option<std::path::PathBuf> {
    if !dir.is_dir() {
        return None;
    }
    for entry in std::fs::read_dir(dir).ok()? {
        let entry = entry.ok()?;
        let path = entry.path();
        if path.is_file() && path.file_name().is_some_and(|n| n == name) {
            return Some(path);
        }
        if path.is_dir() {
            if let Some(found) = find_file(&path, name) {
                return Some(found);
            }
        }
    }
    None
}
