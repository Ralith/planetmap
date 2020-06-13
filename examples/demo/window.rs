//! Based on ash's examples/lib.rs

use ash::extensions::{
    ext::DebugReport,
    khr::{Surface, Swapchain},
};

pub use ash::version::{DeviceV1_0, EntryV1_0, InstanceV1_0};
use ash::{vk, Device, Entry, Instance};
use std::default::Default;
use std::ffi::{CStr, CString};
use std::ops::Drop;
use std::os::raw::{c_char, c_void};
use std::sync::Arc;

pub fn record_submit_commandbuffer<T, D: DeviceV1_0, F: FnOnce() -> T>(
    device: &D,
    command_buffer: vk::CommandBuffer,
    submit_queue: vk::Queue,
    wait_mask: &[vk::PipelineStageFlags],
    wait_semaphores: &[vk::Semaphore],
    signal_semaphores: &[vk::Semaphore],
    f: F,
) -> T {
    unsafe {
        device
            .reset_command_buffer(
                command_buffer,
                vk::CommandBufferResetFlags::RELEASE_RESOURCES,
            )
            .expect("Reset command buffer failed.");

        let command_buffer_begin_info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        device
            .begin_command_buffer(command_buffer, &command_buffer_begin_info)
            .expect("Begin commandbuffer");
        let x = f();
        device
            .end_command_buffer(command_buffer)
            .expect("End commandbuffer");

        let submit_fence = device
            .create_fence(&vk::FenceCreateInfo::default(), None)
            .expect("Create fence failed.");

        let command_buffers = vec![command_buffer];

        let submit_info = vk::SubmitInfo::builder()
            .wait_semaphores(wait_semaphores)
            .wait_dst_stage_mask(wait_mask)
            .command_buffers(&command_buffers)
            .signal_semaphores(signal_semaphores);

        device
            .queue_submit(submit_queue, &[submit_info.build()], submit_fence)
            .expect("queue submit failed.");
        device
            .wait_for_fences(&[submit_fence], true, std::u64::MAX)
            .expect("Wait for fence failed.");
        device.destroy_fence(submit_fence, None);
        x
    }
}

unsafe extern "system" fn vulkan_debug_callback(
    flags: vk::DebugReportFlagsEXT,
    _: vk::DebugReportObjectTypeEXT,
    _: u64,
    _: usize,
    _: i32,
    _: *const c_char,
    p_message: *const c_char,
    _: *mut c_void,
) -> u32 {
    eprintln!(
        "{:?} {}",
        flags,
        CStr::from_ptr(p_message).to_string_lossy()
    );
    vk::FALSE
}

pub fn find_memorytype_index(
    memory_req: &vk::MemoryRequirements,
    memory_prop: &vk::PhysicalDeviceMemoryProperties,
    flags: vk::MemoryPropertyFlags,
) -> Option<u32> {
    // Try to find an exactly matching memory flag
    let best_suitable_index =
        find_memorytype_index_f(memory_req, memory_prop, flags, |property_flags, flags| {
            property_flags == flags
        });
    if best_suitable_index.is_some() {
        return best_suitable_index;
    }
    // Otherwise find a memory flag that works
    find_memorytype_index_f(memory_req, memory_prop, flags, |property_flags, flags| {
        property_flags & flags == flags
    })
}

pub fn find_memorytype_index_f<F: Fn(vk::MemoryPropertyFlags, vk::MemoryPropertyFlags) -> bool>(
    memory_req: &vk::MemoryRequirements,
    memory_prop: &vk::PhysicalDeviceMemoryProperties,
    flags: vk::MemoryPropertyFlags,
    f: F,
) -> Option<u32> {
    let mut memory_type_bits = memory_req.memory_type_bits;
    for (index, ref memory_type) in memory_prop.memory_types.iter().enumerate() {
        if memory_type_bits & 1 == 1 {
            if f(memory_type.property_flags, flags) {
                return Some(index as u32);
            }
        }
        memory_type_bits = memory_type_bits >> 1;
    }
    None
}

pub struct ExampleBase {
    pub entry: Entry,
    pub instance: Instance,
    pub device: Arc<Device>,
    pub surface_loader: Surface,
    pub swapchain_loader: Arc<Swapchain>,
    pub debug_report_loader: DebugReport,
    pub window: winit::window::Window,
    pub debug_call_back: vk::DebugReportCallbackEXT,

    pub pdevice: vk::PhysicalDevice,
    pub device_memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub queue_family_index: u32,
    pub present_queue: vk::Queue,

    pub surface: vk::SurfaceKHR,
    pub surface_format: vk::SurfaceFormatKHR,

    pub pool: vk::CommandPool,
    pub draw_command_buffer: vk::CommandBuffer,

    pub present_complete_semaphore: vk::Semaphore,
    pub rendering_complete_semaphore: vk::Semaphore,
}

impl ExampleBase {
    pub fn new(window_width: u32, window_height: u32) -> (Self, winit::event_loop::EventLoop<()>) {
        unsafe {
            let event_loop = winit::event_loop::EventLoop::new();
            let window = winit::window::WindowBuilder::new()
                .with_title("planetmap example")
                .with_inner_size(winit::dpi::LogicalSize::new(
                    window_width as f64,
                    window_height as f64,
                ))
                .build(&event_loop)
                .unwrap();
            let entry = Entry::new().unwrap();
            let app_name = CString::new("planetmap example").unwrap();

            let mut extension_names_raw = vec![
                Surface::name().as_ptr(),
                DebugReport::name().as_ptr(),
                b"VK_KHR_get_physical_device_properties2\0".as_ptr() as _,
            ];

            extension_names_raw.extend(
                ash_window::enumerate_required_extensions(&window)
                    .unwrap()
                    .into_iter()
                    .map(|x| x.as_ptr()),
            );

            let appinfo = vk::ApplicationInfo::builder()
                .application_name(&app_name)
                .application_version(0)
                .engine_name(&app_name)
                .engine_version(0)
                .api_version(vk::make_version(1, 0, 36));

            let create_info = vk::InstanceCreateInfo::builder()
                .application_info(&appinfo)
                .enabled_extension_names(&extension_names_raw);

            let instance: Instance = entry
                .create_instance(&create_info, None)
                .expect("Instance creation error");

            let debug_info = vk::DebugReportCallbackCreateInfoEXT::builder()
                .flags(
                    vk::DebugReportFlagsEXT::ERROR
                        | vk::DebugReportFlagsEXT::WARNING
                        | vk::DebugReportFlagsEXT::PERFORMANCE_WARNING,
                )
                .pfn_callback(Some(vulkan_debug_callback));

            let debug_report_loader = DebugReport::new(&entry, &instance);
            let debug_call_back = debug_report_loader
                .create_debug_report_callback(&debug_info, None)
                .unwrap();
            let surface = ash_window::create_surface(&entry, &instance, &window, None).unwrap();
            let pdevices = instance
                .enumerate_physical_devices()
                .expect("Physical device error");
            let surface_loader = Surface::new(&entry, &instance);
            let (pdevice, queue_family_index) = pdevices
                .iter()
                .map(|pdevice| {
                    instance
                        .get_physical_device_queue_family_properties(*pdevice)
                        .iter()
                        .enumerate()
                        .filter_map(|(index, ref info)| {
                            let supports_graphic_and_surface =
                                info.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                                    && surface_loader
                                        .get_physical_device_surface_support(
                                            *pdevice,
                                            index as u32,
                                            surface,
                                        )
                                        .unwrap();
                            match supports_graphic_and_surface {
                                true => Some((*pdevice, index)),
                                _ => None,
                            }
                        })
                        .nth(0)
                })
                .filter_map(|v| v)
                .next()
                .expect("Couldn't find suitable device.");
            let queue_family_index = queue_family_index as u32;
            let device_extension_names_raw = [
                Swapchain::name().as_ptr(),
                b"VK_EXT_descriptor_indexing\0".as_ptr() as _,
                b"VK_KHR_maintenance3\0".as_ptr() as _,
            ];
            let mut indexing = vk::PhysicalDeviceDescriptorIndexingFeaturesEXT {
                shader_sampled_image_array_non_uniform_indexing: vk::TRUE,
                ..Default::default()
            };
            let mut features = vk::PhysicalDeviceFeatures2 {
                p_next: &mut indexing as *mut _ as *mut _,
                features: vk::PhysicalDeviceFeatures {
                    fill_mode_non_solid: vk::TRUE,
                    dual_src_blend: vk::TRUE,
                    ..Default::default()
                },
                ..Default::default()
            };

            let priorities = [1.0];
            let queue_info = [vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(queue_family_index)
                .queue_priorities(&priorities)
                .build()];

            let device_create_info = vk::DeviceCreateInfo::builder()
                .queue_create_infos(&queue_info)
                .enabled_extension_names(&device_extension_names_raw)
                .push_next(&mut features);

            let device = Arc::new(
                instance
                    .create_device(pdevice, &device_create_info, None)
                    .unwrap(),
            );
            let present_queue = device.get_device_queue(queue_family_index as u32, 0);

            let swapchain_loader = Arc::new(Swapchain::new(&instance, &*device));

            let pool_create_info = vk::CommandPoolCreateInfo::builder()
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                .queue_family_index(queue_family_index);

            let pool = device.create_command_pool(&pool_create_info, None).unwrap();

            let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::builder()
                .command_buffer_count(1)
                .command_pool(pool)
                .level(vk::CommandBufferLevel::PRIMARY);

            let command_buffers = device
                .allocate_command_buffers(&command_buffer_allocate_info)
                .unwrap();
            let draw_command_buffer = command_buffers[0];

            let semaphore_create_info = vk::SemaphoreCreateInfo::default();

            let device_memory_properties = instance.get_physical_device_memory_properties(pdevice);

            let surface_formats = surface_loader
                .get_physical_device_surface_formats(pdevice, surface)
                .unwrap();
            let surface_format = surface_formats
                .iter()
                .map(|sfmt| match sfmt.format {
                    vk::Format::UNDEFINED => vk::SurfaceFormatKHR {
                        format: vk::Format::B8G8R8_SRGB,
                        color_space: sfmt.color_space,
                    },
                    _ => sfmt.clone(),
                })
                .nth(0)
                .expect("Unable to find suitable surface format.");

            let present_complete_semaphore = device
                .create_semaphore(&semaphore_create_info, None)
                .unwrap();
            let rendering_complete_semaphore = device
                .create_semaphore(&semaphore_create_info, None)
                .unwrap();
            (
                ExampleBase {
                    entry,
                    instance,
                    device,
                    queue_family_index,
                    pdevice,
                    device_memory_properties,
                    window,
                    surface_loader,
                    swapchain_loader,
                    present_queue,
                    pool,
                    draw_command_buffer,
                    present_complete_semaphore,
                    rendering_complete_semaphore,
                    surface,
                    surface_format,
                    debug_call_back,
                    debug_report_loader,
                },
                event_loop,
            )
        }
    }
}

impl Drop for ExampleBase {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().unwrap();
            self.device
                .destroy_semaphore(self.present_complete_semaphore, None);
            self.device
                .destroy_semaphore(self.rendering_complete_semaphore, None);
            self.device.destroy_command_pool(self.pool, None);
            self.device.destroy_device(None);
            self.surface_loader.destroy_surface(self.surface, None);
            self.debug_report_loader
                .destroy_debug_report_callback(self.debug_call_back, None);
            self.instance.destroy_instance(None);
        }
    }
}

pub struct SwapchainState {
    device: Arc<Device>,
    loader: Arc<Swapchain>,
    pub extent: vk::Extent2D,
    pub handle: vk::SwapchainKHR,

    pub frames: Vec<Frame>,

    depth_image: vk::Image,
    pub depth_image_view: vk::ImageView,
    depth_image_memory: vk::DeviceMemory,
}

impl SwapchainState {
    pub fn new(base: &ExampleBase, render_pass: vk::RenderPass, old: Option<Self>) -> Self {
        unsafe {
            let capabilities = base
                .surface_loader
                .get_physical_device_surface_capabilities(base.pdevice, base.surface)
                .unwrap();

            let surface_capabilities = base
                .surface_loader
                .get_physical_device_surface_capabilities(base.pdevice, base.surface)
                .unwrap();
            let extent = match surface_capabilities.current_extent.width {
                std::u32::MAX => vk::Extent2D {
                    width: 1280,
                    height: 1024,
                },
                _ => surface_capabilities.current_extent,
            };
            let pre_transform = if surface_capabilities
                .supported_transforms
                .contains(vk::SurfaceTransformFlagsKHR::IDENTITY)
            {
                vk::SurfaceTransformFlagsKHR::IDENTITY
            } else {
                surface_capabilities.current_transform
            };
            let present_modes = base
                .surface_loader
                .get_physical_device_surface_present_modes(base.pdevice, base.surface)
                .unwrap();
            let present_mode = present_modes
                .iter()
                .cloned()
                .find(|&mode| mode == vk::PresentModeKHR::MAILBOX)
                .unwrap_or(vk::PresentModeKHR::FIFO);

            let image_count = if capabilities.max_image_count > 0 {
                capabilities
                    .max_image_count
                    .min(capabilities.min_image_count + 1)
            } else {
                capabilities.min_image_count + 1
            };

            let handle = base
                .swapchain_loader
                .create_swapchain(
                    &vk::SwapchainCreateInfoKHR::builder()
                        .surface(base.surface)
                        .min_image_count(image_count)
                        .image_color_space(base.surface_format.color_space)
                        .image_format(base.surface_format.format)
                        .image_extent(extent)
                        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
                        .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                        .pre_transform(pre_transform)
                        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                        .present_mode(present_mode)
                        .clipped(true)
                        .image_array_layers(1)
                        .old_swapchain(
                            old.as_ref()
                                .map_or_else(vk::SwapchainKHR::null, |x| x.handle),
                        ),
                    None,
                )
                .unwrap();

            let device_memory_properties = base
                .instance
                .get_physical_device_memory_properties(base.pdevice);
            let depth_image_create_info = vk::ImageCreateInfo::builder()
                .image_type(vk::ImageType::TYPE_2D)
                .format(vk::Format::D32_SFLOAT)
                .extent(vk::Extent3D {
                    width: extent.width,
                    height: extent.height,
                    depth: 1,
                })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(
                    vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
                        | vk::ImageUsageFlags::INPUT_ATTACHMENT,
                )
                .sharing_mode(vk::SharingMode::EXCLUSIVE);

            let depth_image = base
                .device
                .create_image(&depth_image_create_info, None)
                .unwrap();
            let depth_image_memory_req = base.device.get_image_memory_requirements(depth_image);
            let depth_image_memory_index = find_memorytype_index(
                &depth_image_memory_req,
                &device_memory_properties,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )
            .expect("Unable to find suitable memory index for depth image.");

            let depth_image_allocate_info = vk::MemoryAllocateInfo::builder()
                .allocation_size(depth_image_memory_req.size)
                .memory_type_index(depth_image_memory_index);

            let depth_image_memory = base
                .device
                .allocate_memory(&depth_image_allocate_info, None)
                .unwrap();

            base.device
                .bind_image_memory(depth_image, depth_image_memory, 0)
                .expect("Unable to bind depth image memory");

            let depth_image_view_info = vk::ImageViewCreateInfo::builder()
                .subresource_range(
                    vk::ImageSubresourceRange::builder()
                        .aspect_mask(vk::ImageAspectFlags::DEPTH)
                        .level_count(1)
                        .layer_count(1)
                        .build(),
                )
                .image(depth_image)
                .format(depth_image_create_info.format)
                .view_type(vk::ImageViewType::TYPE_2D);

            let depth_image_view = base
                .device
                .create_image_view(&depth_image_view_info, None)
                .unwrap();

            let frames = base
                .swapchain_loader
                .get_swapchain_images(handle)
                .unwrap()
                .into_iter()
                .map(|image| {
                    let view = base
                        .device
                        .create_image_view(
                            &vk::ImageViewCreateInfo::builder()
                                .view_type(vk::ImageViewType::TYPE_2D)
                                .format(base.surface_format.format)
                                .components(vk::ComponentMapping {
                                    r: vk::ComponentSwizzle::R,
                                    g: vk::ComponentSwizzle::G,
                                    b: vk::ComponentSwizzle::B,
                                    a: vk::ComponentSwizzle::A,
                                })
                                .subresource_range(vk::ImageSubresourceRange {
                                    aspect_mask: vk::ImageAspectFlags::COLOR,
                                    base_mip_level: 0,
                                    level_count: 1,
                                    base_array_layer: 0,
                                    layer_count: 1,
                                })
                                .image(image),
                            None,
                        )
                        .unwrap();
                    let buffer = base
                        .device
                        .create_framebuffer(
                            &vk::FramebufferCreateInfo::builder()
                                .render_pass(render_pass)
                                .attachments(&[view, depth_image_view])
                                .width(extent.width)
                                .height(extent.height)
                                .layers(1),
                            None,
                        )
                        .unwrap();
                    Frame {
                        image,
                        view,
                        buffer,
                    }
                })
                .collect();

            Self {
                device: base.device.clone(),
                loader: base.swapchain_loader.clone(),
                extent,
                handle,
                frames,

                depth_image,
                depth_image_memory,
                depth_image_view,
            }
        }
    }

    pub unsafe fn acquire_next_image(&self, sem: vk::Semaphore) -> Result<(u32, bool), vk::Result> {
        self.loader
            .acquire_next_image(self.handle, std::u64::MAX, sem, vk::Fence::null())
    }
}

impl Drop for SwapchainState {
    fn drop(&mut self) {
        unsafe {
            self.device.free_memory(self.depth_image_memory, None);
            self.device.destroy_image_view(self.depth_image_view, None);
            self.device.destroy_image(self.depth_image, None);
            for frame in &self.frames {
                self.device.destroy_image_view(frame.view, None);
                self.device.destroy_framebuffer(frame.buffer, None);
            }
            self.loader.destroy_swapchain(self.handle, None);
        }
    }
}

pub struct Frame {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub buffer: vk::Framebuffer,
}
