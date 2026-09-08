//! libhrx's C API, declared by hand from `libhrx/include/hrx_runtime.h`. No bindgen, so the whole
//! surface this project uses is readable in one file.
//!
//! Two conventions from the header matter and are easy to get wrong:
//!   * `hrx_status_t` is a pointer, and **NULL means success**. `hrx_status_is_ok` is a `static inline`
//!     in the header, so it is not a linkable symbol — [`is_ok`] reimplements it.
//!   * handles are opaque pointers; a null handle is never valid.
use std::ffi::{c_char, c_int, c_void};

pub type Status = *mut c_void;
pub type Device = *mut c_void;
pub type Stream = *mut c_void;
pub type Buffer = *mut c_void;
pub type Executable = *mut c_void;

/// `hrx_status_is_ok`, which the header defines as `static inline` rather than exporting.
#[inline]
pub fn is_ok(status: Status) -> bool {
    status.is_null()
}

pub const MEMORY_TYPE_DEVICE_LOCAL: u32 = 0x0000_0030;
pub const BUFFER_USAGE_DEFAULT: u32 = 0x0000_0C03;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct ExportInfo {
    pub name: *const c_char,
    pub flags: u32,
    pub constant_byte_length: u32,
    pub binding_count: u32,
    pub parameter_count: u32,
    pub workgroup_size: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DispatchConfig {
    pub workgroup_count: [u32; 3],
    pub workgroup_size: [u32; 3],
    pub subgroup_size: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct BufferRef {
    pub buffer: Buffer,
    pub offset: usize,
    pub length: usize,
}

#[link(name = "hrx")]
extern "C" {
    pub fn hrx_status_to_string(status: Status, out_message: *mut *mut c_char, out_length: *mut usize) -> Status;
    pub fn hrx_status_free_message(message: *mut c_char);
    pub fn hrx_status_ignore(status: Status);

    pub fn hrx_gpu_initialize(flags: u32) -> Status;
    pub fn hrx_gpu_device_count(count: *mut c_int) -> Status;
    pub fn hrx_gpu_device_get(index: c_int, device: *mut Device) -> Status;

    pub fn hrx_stream_create(device: Device, flags: u32, out_stream: *mut Stream) -> Status;
    pub fn hrx_stream_release(stream: Stream);
    pub fn hrx_stream_synchronize(stream: Stream) -> Status;

    pub fn hrx_buffer_allocate(
        stream: Stream,
        size: usize,
        memory_type: u32,
        usage: u32,
        out_buffer: *mut Buffer,
    ) -> Status;
    pub fn hrx_buffer_release(buffer: Buffer);
    pub fn hrx_stream_fill_buffer(
        stream: Stream,
        buffer: Buffer,
        offset: usize,
        size: usize,
        pattern: *const c_void,
        pattern_size: usize,
    ) -> Status;
    pub fn hrx_stream_copy_buffer(
        stream: Stream,
        src: Buffer,
        src_offset: usize,
        dst: Buffer,
        dst_offset: usize,
        size: usize,
    ) -> Status;
    pub fn hrx_synchronous_h2d(
        device: Device,
        host_src: *const c_void,
        dst: Buffer,
        dst_offset: usize,
        size: usize,
    ) -> Status;
    pub fn hrx_synchronous_d2h(
        device: Device,
        src: Buffer,
        src_offset: usize,
        host_dst: *mut c_void,
        size: usize,
    ) -> Status;

    pub fn hrx_executable_load_file(
        device: Device,
        path: *const c_char,
        target_family: *const c_char,
        target_key: *const c_char,
        out_executable: *mut Executable,
    ) -> Status;
    pub fn hrx_executable_release(executable: Executable);
    pub fn hrx_executable_lookup_export_by_name(
        executable: Executable,
        name: *const c_char,
        out_ordinal: *mut u32,
    ) -> Status;
    pub fn hrx_executable_export_info(
        executable: Executable,
        ordinal: u32,
        out_info: *mut ExportInfo,
    ) -> Status;
    #[allow(clippy::too_many_arguments)]
    pub fn hrx_stream_dispatch(
        stream: Stream,
        executable: Executable,
        ordinal: u32,
        config: *const DispatchConfig,
        constants: *const c_void,
        constants_size: usize,
        bindings: *const BufferRef,
        binding_count: usize,
        flags: u32,
    ) -> Status;
}
