use std::ffi::CString;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::sync::OnceLock;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

const LOG_FILE: &str = "spectra-viz-diagnostics.log";

static CRASH_FD: AtomicI32 = AtomicI32::new(-1);
static IN_HANDLER: AtomicBool = AtomicBool::new(false);
static START: OnceLock<Instant> = OnceLock::new();

#[repr(C, align(16))]
struct AltStack([u8; 64 * 1024]);
static ALT_STACK: AltStack = AltStack([0; 64 * 1024]);

pub fn exe_dir() -> Option<PathBuf> {
    std::env::current_exe()
        .ok()
        .and_then(|path| path.parent().map(|parent| parent.to_path_buf()))
}

pub fn clean_exit() -> bool {
    let Some(dir) = exe_dir() else {
        return false;
    };
    std::fs::read_to_string(dir.join(LOG_FILE))
        .map(|text| text.contains("run_native returned"))
        .unwrap_or(false)
}

pub fn install() {
    let path = exe_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(LOG_FILE);
    let cpath = CString::new(path.to_string_lossy().as_bytes()).unwrap_or_default();
    let fd = unsafe {
        libc::open(
            cpath.as_ptr(),
            libc::O_WRONLY | libc::O_CREAT | libc::O_TRUNC | libc::O_APPEND,
            0o644,
        )
    };
    CRASH_FD.store(fd, Ordering::SeqCst);
    let _ = START.set(Instant::now());
    write_file(&startup_header());
    install_handlers();
    mark("crash diagnostics installed");
}

fn write_file(text: &str) {
    let fd = CRASH_FD.load(Ordering::SeqCst);
    if fd >= 0 {
        unsafe {
            let _ = libc::write(fd, text.as_ptr() as *const libc::c_void, text.len());
        }
    }
}

pub fn mark(msg: &str) {
    let secs = START
        .get()
        .map(|started| started.elapsed().as_secs_f64())
        .unwrap_or(0.0);
    let line = format!("[{secs:9.3}s] {msg}\n");
    eprint!("{line}");
    write_file(&line);
}

fn startup_header() -> String {
    let mut out = String::new();
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    out.push_str(&format!("spectra-viz diagnostics, unix time {now}\n"));
    out.push_str(&format!("pid {}\n", std::process::id()));
    if let Ok(exe) = std::env::current_exe() {
        out.push_str(&format!("exe {}\n", exe.display()));
    }
    if let Ok(cwd) = std::env::current_dir() {
        out.push_str(&format!("cwd {}\n", cwd.display()));
    }
    let argv: Vec<String> = std::env::args().collect();
    out.push_str(&format!("argv {}\n", argv.join(" ")));
    out.push_str(&format!(
        "install_fn 0x{:x}\n",
        install as *const () as usize
    ));
    if let Ok(output) = std::process::Command::new("sw_vers")
        .arg("-productVersion")
        .output()
    {
        let version = String::from_utf8_lossy(&output.stdout).trim().to_string();
        out.push_str(&format!("macOS {version}\n"));
    }
    if std::env::var_os("SPECTRA_VIZ_DYLD_STRIPPED").is_some() {
        out.push_str("note: re-executed with DYLD_* environment stripped\n");
    }
    out.push_str(&rlimit_lines());
    out.push_str(&env_lines());
    out.push_str("----\n");
    out
}

fn rlimit_lines() -> String {
    let specs = [
        (libc::RLIMIT_STACK, "stack"),
        (libc::RLIMIT_DATA, "data"),
        (libc::RLIMIT_AS, "address-space"),
        (libc::RLIMIT_NOFILE, "nofile"),
    ];
    let mut out = String::new();
    unsafe {
        for (resource, name) in specs {
            let mut limit: libc::rlimit = std::mem::zeroed();
            if libc::getrlimit(resource, &mut limit) == 0 {
                let fmt = |v: u64| {
                    if v == libc::RLIM_INFINITY {
                        "unlimited".to_string()
                    } else {
                        v.to_string()
                    }
                };
                out.push_str(&format!(
                    "rlimit {name} cur={} max={}\n",
                    fmt(limit.rlim_cur),
                    fmt(limit.rlim_max)
                ));
            }
        }
    }
    out
}

fn env_lines() -> String {
    const VALUE_PREFIXES: [&str; 13] = [
        "DYLD", "MALLOC", "OBJC", "RUST", "TERM", "LC_", "LANG", "SSH", "NS", "XPC", "QT", "CI",
        "SHELL",
    ];
    let mut entries: Vec<(String, Option<String>)> = std::env::vars_os()
        .map(|(key, value)| {
            let key = key.to_string_lossy().into_owned();
            let value = value.to_string_lossy().into_owned();
            let safe = VALUE_PREFIXES.iter().any(|prefix| key.starts_with(prefix));
            if safe {
                (key, Some(value))
            } else {
                (key, None)
            }
        })
        .collect();
    entries.sort();
    let mut out = String::new();
    for (key, value) in entries {
        match value {
            Some(value) => out.push_str(&format!("env {key}={value}\n")),
            None => out.push_str(&format!("env {key} (value hidden)\n")),
        }
    }
    out
}

fn put(buf: &mut [u8], n: &mut usize, bytes: &[u8]) {
    let take = bytes.len().min(buf.len() - *n);
    buf[*n..*n + take].copy_from_slice(&bytes[..take]);
    *n += take;
}

fn put_hex(buf: &mut [u8], n: &mut usize, value: usize) {
    put(buf, n, b"0x");
    let mut shift: i32 = usize::BITS as i32 - 4;
    let mut started = false;
    while shift >= 0 {
        let nibble = (value >> shift as usize) & 0xF;
        if nibble != 0 || started || shift == 0 {
            let byte = if nibble < 10 {
                b'0' + nibble as u8
            } else {
                b'a' + (nibble as u8 - 10)
            };
            put(buf, n, &[byte]);
            started = true;
        }
        shift -= 4;
    }
}

unsafe extern "C" fn handle(sig: libc::c_int, info: *mut libc::siginfo_t, _ctx: *mut libc::c_void) {
    if IN_HANDLER.swap(true, Ordering::SeqCst) {
        libc::_exit(0x5a + sig);
    }
    let fault = (*info).si_addr() as usize;
    let name = match sig {
        libc::SIGBUS => "SIGBUS",
        libc::SIGSEGV => "SIGSEGV",
        libc::SIGILL => "SIGILL",
        _ => "?",
    };
    let mut buf = [0u8; 128];
    let mut n = 0;
    put(&mut buf, &mut n, b"SIGNAL ");
    put_hex(&mut buf, &mut n, sig as usize);
    put(&mut buf, &mut n, b" ");
    put(&mut buf, &mut n, name.as_bytes());
    put(&mut buf, &mut n, b" fault_addr=");
    put_hex(&mut buf, &mut n, fault);
    put(&mut buf, &mut n, b"\n");
    unsafe {
        libc::write(2, buf.as_ptr() as *const libc::c_void, n);
        let fd = CRASH_FD.load(Ordering::SeqCst);
        if fd >= 0 {
            libc::write(fd, buf.as_ptr() as *const libc::c_void, n);
        }
        libc::signal(sig, libc::SIG_DFL);
        libc::raise(sig);
    }
}

#[cfg(target_os = "macos")]
fn install_handlers() {
    unsafe {
        let stack = libc::stack_t {
            ss_sp: (&ALT_STACK.0 as *const u8).cast_mut() as *mut libc::c_void,
            ss_size: ALT_STACK.0.len(),
            ss_flags: 0,
        };
        let rc_stack = libc::sigaltstack(&stack, std::ptr::null_mut());
        let mut action: libc::sigaction = std::mem::zeroed();
        action.sa_sigaction = handle as *const () as usize;
        action.sa_flags = libc::SA_SIGINFO | libc::SA_ONSTACK;
        libc::sigemptyset(&mut action.sa_mask);
        let rc_bus = libc::sigaction(libc::SIGBUS, &action, std::ptr::null_mut());
        let rc_segv = libc::sigaction(libc::SIGSEGV, &action, std::ptr::null_mut());
        let rc_ill = libc::sigaction(libc::SIGILL, &action, std::ptr::null_mut());
        write_file(&format!(
            "handler install: sigaltstack rc={rc_stack}, sigaction rc: bus={rc_bus} segv={rc_segv} ill={rc_ill}, handler 0x{:x}\n",
            handle as *const () as usize
        ));
    }
}

#[cfg(not(target_os = "macos"))]
fn install_handlers() {}
