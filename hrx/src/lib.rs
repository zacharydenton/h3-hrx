//! A safe wrapper over libhrx: the GPU runtime this project dispatches Loom kernels through.
//!
//! The surface mirrors what the pipeline and the kernel launcher actually need — allocate, copy,
//! fill, load an executable, dispatch, synchronise — with owning handles that release on drop and
//! the `hrx_status_t` convention turned into `Result`.
pub mod sys;

use std::ffi::{c_void, CStr, CString};
use std::fmt;
use std::path::Path;

/// The target the kernels in this project are compiled for.
pub const TARGET_FAMILY: &str = "amdgpu";
pub const TARGET_KEY: &str = "gfx1151";

#[derive(Debug)]
pub struct Error(String);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

/// Turns a status into a `Result`, taking ownership of its message. A null status is success.
fn check(status: sys::Status, what: &str) -> Result<()> {
    if sys::is_ok(status) {
        return Ok(());
    }
    let mut message: *mut std::ffi::c_char = std::ptr::null_mut();
    let mut length: usize = 0;
    let text = unsafe {
        let to_string = sys::hrx_status_to_string(status, &mut message, &mut length);
        let text = if sys::is_ok(to_string) && !message.is_null() {
            let owned = CStr::from_ptr(message).to_string_lossy().into_owned();
            sys::hrx_status_free_message(message);
            owned
        } else {
            sys::hrx_status_ignore(to_string);
            "unknown error".to_string()
        };
        sys::hrx_status_ignore(status);
        text
    };
    Err(Error(format!("{what}: {text}")))
}

/// The device and stream themselves, released when the last thing referencing them goes away.
///
/// Buffers and kernels hold one of these, so the stream cannot be released while an allocation or an
/// executable still names its device. The C implementation sidestepped the question by never tearing
/// down at all — its runtime was a function-local static that lived to process exit — which is not
/// something a safe API can leave to the caller to arrange.
struct Inner {
    device: sys::Device,
    stream: sys::Stream,
}

impl Drop for Inner {
    fn drop(&mut self) {
        // Only the stream is ours: `hrx_stream_create` hands back a reference, while
        // `hrx_gpu_device_get` returns a borrowed pointer into libhrx's own device array
        // (`*device = &g_gpu.devices[index]`) without retaining. Releasing that would be an
        // over-release of the global registry, which aborts inside libhrx.
        unsafe { sys::hrx_stream_release(self.stream) };
    }
}

// Safety: this only holds handles. They carry atomic reference counts and no thread-local state, so
// both moving one between threads and releasing it on another are sound. It is `Sync` because holding
// the handle is not using it: every stream operation goes through `&Gpu`, and `Gpu` is deliberately not
// `Sync`, so no two threads can drive one stream at once.
unsafe impl Send for Inner {}
unsafe impl Sync for Inner {}

/// The process's GPU device and its stream. libhrx initialises globally, so this is created once.
pub struct Gpu {
    inner: std::sync::Arc<Inner>,
    /// `Gpu` may be moved between threads but not shared between them: a stream's timepoint and
    /// pending command buffer are ordinary mutable fields, so two threads dispatching through one
    /// would race. `Cell` is `Send` and not `Sync`, which says exactly that.
    _not_sync: std::marker::PhantomData<std::cell::Cell<()>>,
}



impl Gpu {
    pub fn open() -> Result<Self> {
        unsafe {
            check(sys::hrx_gpu_initialize(0), "hrx_gpu_initialize")?;
            let mut count = 0;
            check(sys::hrx_gpu_device_count(&mut count), "hrx_gpu_device_count")?;
            if count < 1 {
                return Err(Error(
                    "no GPU device (is the HSA runtime on LD_LIBRARY_PATH? see docs/setup.md)".into(),
                ));
            }
            let mut device = std::ptr::null_mut();
            check(sys::hrx_gpu_device_get(0, &mut device), "hrx_gpu_device_get")?;
            let mut stream = std::ptr::null_mut();
            check(sys::hrx_stream_create(device, 0, &mut stream), "hrx_stream_create")?;
            Ok(Self {
                inner: std::sync::Arc::new(Inner { device, stream }),
                _not_sync: std::marker::PhantomData,
            })
        }
    }

    pub fn sync(&self) -> Result<()> {
        unsafe { check(sys::hrx_stream_synchronize(self.inner.stream), "hrx_stream_synchronize") }
    }

    /// A device-local allocation. Zero bytes is rounded to one so every argument has an address.
    pub fn alloc(&self, bytes: usize) -> Result<Buffer> {
        let mut buffer = std::ptr::null_mut();
        unsafe {
            check(
                sys::hrx_buffer_allocate(
                    self.inner.stream,
                    bytes.max(1),
                    sys::MEMORY_TYPE_DEVICE_LOCAL,
                    sys::BUFFER_USAGE_DEFAULT,
                    &mut buffer,
                ),
                "hrx_buffer_allocate",
            )?;
        }
        Ok(Buffer { raw: buffer, bytes: bytes.max(1), _device: self.inner.clone() })
    }

    /// Only the low byte of `value` is meaningful: the fill pattern is one byte wide.
    pub fn memset(&self, dst: &Buffer, value: u8, bytes: usize) -> Result<()> {
        unsafe {
            check(
                sys::hrx_stream_fill_buffer(
                    self.inner.stream,
                    dst.raw,
                    0,
                    bytes,
                    &value as *const u8 as *const c_void,
                    1,
                ),
                "hrx_stream_fill_buffer",
            )
        }
    }

    /// The synchronous transfers bypass the stream's pending commands, so the stream is drained first.
    pub fn h2d(&self, dst: &Buffer, src: &[u8]) -> Result<()> {
        self.h2d_at(dst, 0, src)
    }

    /// As [`Gpu::h2d`], writing at an offset: a large weight is uploaded in chunks so the host never
    /// stages more than one chunk of it.
    pub fn h2d_at(&self, dst: &Buffer, offset: usize, src: &[u8]) -> Result<()> {
        if src.is_empty() {
            return Ok(());
        }
        if offset + src.len() > dst.bytes {
            return Err(Error(format!(
                "upload of {} bytes at {offset} overruns a {}-byte allocation",
                src.len(),
                dst.bytes
            )));
        }
        self.sync()?;
        unsafe {
            check(
                sys::hrx_synchronous_h2d(
                    self.inner.device,
                    src.as_ptr() as *const c_void,
                    dst.raw,
                    offset,
                    src.len(),
                ),
                "hrx_synchronous_h2d",
            )
        }
    }

    pub fn d2h(&self, src: &Buffer, dst: &mut [u8]) -> Result<()> {
        if dst.is_empty() {
            return Ok(());
        }
        if dst.len() > src.bytes {
            return Err(Error(format!(
                "read of {} bytes from a {}-byte allocation",
                dst.len(),
                src.bytes
            )));
        }
        self.sync()?;
        unsafe {
            check(
                sys::hrx_synchronous_d2h(self.inner.device, src.raw, 0, dst.as_mut_ptr() as *mut c_void, dst.len()),
                "hrx_synchronous_d2h",
            )
        }
    }

    /// Reads back from a view rather than a whole allocation, which is how the pipeline inspects rows
    /// it handed a kernel at an offset.
    pub fn d2h_ref(&self, src: sys::BufferRef, dst: &mut [u8]) -> Result<()> {
        if dst.is_empty() {
            return Ok(());
        }
        if dst.len() > src.length {
            return Err(Error(format!(
                "read of {} bytes from a {}-byte view",
                dst.len(),
                src.length
            )));
        }
        self.sync()?;
        unsafe {
            check(
                sys::hrx_synchronous_d2h(
                    self.inner.device,
                    src.buffer,
                    src.offset,
                    dst.as_mut_ptr() as *mut c_void,
                    dst.len(),
                ),
                "hrx_synchronous_d2h",
            )
        }
    }

    pub fn d2d(&self, dst: &Buffer, src: &Buffer, bytes: usize) -> Result<()> {
        self.d2d_at(dst, 0, src, 0, bytes)
    }

    /// A device copy into or out of the middle of an allocation — a row range of a packed sequence,
    /// or one half of a two-row table.
    pub fn d2d_at(
        &self,
        dst: &Buffer,
        dst_offset: usize,
        src: &Buffer,
        src_offset: usize,
        bytes: usize,
    ) -> Result<()> {
        unsafe {
            check(
                sys::hrx_stream_copy_buffer(
                    self.inner.stream,
                    src.raw,
                    src_offset,
                    dst.raw,
                    dst_offset,
                    bytes,
                ),
                "hrx_stream_copy_buffer",
            )
        }
    }

    pub fn load(&self, path: &Path, symbol: &str) -> Result<Kernel> {
        let c_path = CString::new(path.as_os_str().as_encoded_bytes())
            .map_err(|_| Error(format!("{} contains a NUL", path.display())))?;
        let c_family = CString::new(TARGET_FAMILY).unwrap();
        let c_key = CString::new(TARGET_KEY).unwrap();
        let c_symbol = CString::new(symbol).map_err(|_| Error(format!("{symbol} contains a NUL")))?;
        unsafe {
            let mut executable = std::ptr::null_mut();
            check(
                sys::hrx_executable_load_file(
                    self.inner.device,
                    c_path.as_ptr(),
                    c_family.as_ptr(),
                    c_key.as_ptr(),
                    &mut executable,
                ),
                &format!("loading {}", path.display()),
            )?;
            let kernel = (|| {
                let mut ordinal = 0u32;
                check(
                    sys::hrx_executable_lookup_export_by_name(executable, c_symbol.as_ptr(), &mut ordinal),
                    &format!("looking up {symbol} in {}", path.display()),
                )?;
                let mut info = sys::ExportInfo::default();
                check(
                    sys::hrx_executable_export_info(executable, ordinal, &mut info),
                    &format!("export info for {symbol}"),
                )?;
                Ok(Kernel {
                    executable,
                    ordinal,
                    info,
                    symbol: symbol.to_string(),
                    _device: self.inner.clone(),
                })
            })();
            if kernel.is_err() {
                sys::hrx_executable_release(executable);
            }
            kernel
        }
    }

    /// Dispatch, with the export's own metadata deciding how the constants are packed.
    ///
    /// `scalars` are the kernel's by-value arguments in declaration order and `bindings` its buffer
    /// arguments in declaration order. The export reports how many bytes of constants it wants and how
    /// many bindings it has; a mismatch is a caller error and is reported as one rather than dispatched.
    pub fn dispatch(
        &self,
        kernel: &Kernel,
        grid: [u32; 3],
        block: [u32; 3],
        scalars: &[u32],
        bindings: &[sys::BufferRef],
    ) -> Result<()> {
        let info = &kernel.info;
        if bindings.len() != info.binding_count as usize {
            return Err(Error(format!(
                "{} takes {} buffer arguments, {} given",
                kernel.symbol,
                info.binding_count,
                bindings.len()
            )));
        }
        let mut constants = [0u8; 256];
        let size = info.constant_byte_length as usize;
        if size > constants.len() {
            return Err(Error(format!("{} wants {size} constant bytes", kernel.symbol)));
        }
        if !scalars.is_empty() {
            if size % scalars.len() != 0 {
                return Err(Error(format!(
                    "{} wants {size} constant bytes, not divisible by {} scalars",
                    kernel.symbol,
                    scalars.len()
                )));
            }
            let width = size / scalars.len();
            if width != 4 && width != 8 {
                return Err(Error(format!("{} implies a {width}-byte scalar slot", kernel.symbol)));
            }
            for (i, v) in scalars.iter().enumerate() {
                constants[i * width..i * width + 4].copy_from_slice(&v.to_le_bytes());
            }
        } else if size != 0 {
            return Err(Error(format!("{} wants {size} constant bytes, none given", kernel.symbol)));
        }
        let config = sys::DispatchConfig {
            workgroup_count: grid,
            workgroup_size: block,
            subgroup_size: 32,
        };
        unsafe {
            check(
                sys::hrx_stream_dispatch(
                    self.inner.stream,
                    kernel.executable,
                    kernel.ordinal,
                    &config,
                    constants.as_ptr() as *const c_void,
                    size,
                    bindings.as_ptr(),
                    bindings.len(),
                    0,
                ),
                &format!("dispatching {}", kernel.symbol),
            )
        }
    }
}

/// A device allocation. libhrx buffers have no host-visible address, so this is the handle itself
/// rather than a pointer; bindings are made with [`Buffer::binding`].
pub struct Buffer {
    raw: sys::Buffer,
    bytes: usize,
    /// Keeps the device alive: releasing a buffer after its device is gone would be a use-after-free.
    _device: std::sync::Arc<Inner>,
}

// Safety: `hrx_buffer_s` is an immutable handle after allocation — a HAL buffer pointer, its device and
// its length — behind an atomic reference count, so moving one to another thread and releasing it there
// is sound. It is `Sync` as well because nothing mutates it through a shared reference: every transfer
// and fill goes through `&Gpu`, which is deliberately not `Sync`, so two threads cannot reach the same
// allocation concurrently through this API.
unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

impl Buffer {
    pub fn bytes(&self) -> usize {
        self.bytes
    }
    pub fn binding(&self) -> sys::BufferRef {
        sys::BufferRef { buffer: self.raw, offset: 0, length: self.bytes }
    }

    /// A binding into part of the allocation. The pipeline hands kernels views at row offsets, which
    /// the C did with pointer arithmetic; a device buffer here has no host-visible address, so the
    /// offset travels in the binding instead.
    pub fn slice(&self, offset: usize, length: usize) -> sys::BufferRef {
        assert!(offset + length <= self.bytes, "slice past the allocation");
        sys::BufferRef { buffer: self.raw, offset, length }
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe { sys::hrx_buffer_release(self.raw) }
    }
}

pub struct Kernel {
    executable: sys::Executable,
    ordinal: u32,
    info: sys::ExportInfo,
    symbol: String,
    /// As for a buffer: the executable names its device.
    _device: std::sync::Arc<Inner>,
}

// Safety: `hrx_executable_s` is fixed once loaded — a retained HAL executable, its device and a
// snapshot of its export names — behind an atomic reference count. Everything this type exposes is
// read-only, and dispatching with it needs `&Gpu`.
unsafe impl Send for Kernel {}
unsafe impl Sync for Kernel {}

impl Kernel {
    pub fn info(&self) -> &sys::ExportInfo {
        &self.info
    }
    pub fn symbol(&self) -> &str {
        &self.symbol
    }
}

impl Drop for Kernel {
    fn drop(&mut self) {
        unsafe { sys::hrx_executable_release(self.executable) }
    }
}
