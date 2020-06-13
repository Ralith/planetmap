//! Helpers for rendering planet maps using Vulkan
//!
//! Requires the `ash` feature to be enabled.

use std::sync::Arc;
use std::{mem, ptr, slice};

use ash::version::{DeviceV1_0, InstanceV1_0};
use ash::{vk, Device, Instance};

use crate::{cache, Chunk};

/// A GPU-resident streaming LoD implementation for per-`Chunk` texture data
pub struct Cache {
    mgr: cache::Manager,

    instance_memory: vk::DeviceMemory,
    instance_buffer: vk::Buffer,
    instances: *mut [u8],
    shared: Arc<Shared>,
}

unsafe impl Send for Cache {}

impl Drop for Cache {
    fn drop(&mut self) {
        unsafe {
            self.shared
                .device
                .destroy_buffer(self.instance_buffer, None);
            self.shared.device.free_memory(self.instance_memory, None);
        }
    }
}

impl Cache {
    /// Construct a cache for `textures`, to be sampled from a queue in the family at index
    /// `sample_queue_family`.
    pub fn new(
        instance: &Instance,
        pdevice: vk::PhysicalDevice,
        device: Arc<Device>,
        config: cache::Config,
        textures: &[TextureKind],
        sample_queue_family: u32,
    ) -> Self {
        let slots = config.slots_needed() * 3 / 2;
        let mgr = cache::Manager::new(slots, config);

        unsafe {
            let device_memory_properties = instance.get_physical_device_memory_properties(pdevice);

            let instances_size = std::mem::size_of::<ChunkInstance>() * slots;
            let instance_buffer = device
                .create_buffer(
                    &vk::BufferCreateInfo {
                        size: instances_size as u64,
                        usage: vk::BufferUsageFlags::VERTEX_BUFFER,
                        sharing_mode: vk::SharingMode::EXCLUSIVE,
                        ..Default::default()
                    },
                    None,
                )
                .unwrap();

            let instance_memory = {
                let req = device.get_buffer_memory_requirements(instance_buffer);

                let memory_index = find_memorytype_index(
                    &req,
                    &device_memory_properties,
                    vk::MemoryPropertyFlags::HOST_VISIBLE,
                )
                .expect("no suitable memory type for instance buffer");

                device
                    .allocate_memory(
                        &vk::MemoryAllocateInfo {
                            allocation_size: req.size,
                            memory_type_index: memory_index,
                            ..Default::default()
                        },
                        None,
                    )
                    .unwrap()
            };
            device
                .bind_buffer_memory(instance_buffer, instance_memory, 0)
                .unwrap();

            let instances_ptr = device
                .map_memory(
                    instance_memory,
                    0,
                    vk::WHOLE_SIZE,
                    vk::MemoryMapFlags::empty(),
                )
                .unwrap() as *mut u8;

            let limits = instance.get_physical_device_properties(pdevice).limits;

            let array_count = (slots / limits.max_image_array_layers as usize) as u32;
            let last_array_size = (slots % limits.max_image_array_layers as usize) as u32;
            let array_size = slots.min(limits.max_image_array_layers as usize) as u32;
            let arrays = textures
                .iter()
                .map(|x| ArraySet {
                    extent: x.extent,
                    stages: x.stages,
                    arrays: (0..array_count)
                        .map(|_| {
                            TextureArray::new(&device, &device_memory_properties, x, array_size)
                        })
                        .chain(if last_array_size != 0 {
                            Some(TextureArray::new(
                                &device,
                                &device_memory_properties,
                                x,
                                last_array_size,
                            ))
                        } else {
                            None
                        })
                        .collect(),
                })
                .collect();

            Self {
                mgr,
                instance_memory,
                instance_buffer,
                instances: slice::from_raw_parts_mut(instances_ptr, instances_size),
                shared: Arc::new(Shared {
                    device,
                    array_size,
                    arrays,
                    sample_queue_family,
                }),
            }
        }
    }

    /// Construct a transfer engine for use on a queue in the family at index
    /// `transfer_queue_family`.
    pub fn transfer_engine(&self, transfer_queue_family: u32) -> TransferEngine {
        TransferEngine {
            shared: self.shared.clone(),
            transfer_queue_family,
        }
    }

    /// Update instance buffer and compute transfers to initiate for a new `view` transform
    ///
    /// Returns the number of elements now in the instance buffer and a list of non-resident chunks
    /// that are wanted.
    pub fn update(
        &mut self,
        sphere_radius: f64,
        view: &na::IsometryMatrix3<f64>,
    ) -> (u32, Vec<Chunk>) {
        let viewpoint = view.inverse() * na::Point3::origin(); // Camera position in world space
        let state = self.mgr.update(&[viewpoint / sphere_radius]);
        let count = state.render.len() as u32;
        unsafe {
            for (mem, (chunk, neighborhood, slot)) in (*self.instances)
                .chunks_mut(mem::size_of::<ChunkInstance>())
                .zip(state.render.into_iter())
            {
                let (origin, worldview) = chunk.worldview(sphere_radius, view);
                ptr::write_unaligned(
                    mem.as_mut_ptr() as *mut _,
                    ChunkInstance {
                        worldview: worldview.to_homogeneous(),
                        coords: [chunk.coords.x, chunk.coords.y],
                        depth: chunk.depth as u32,
                        slot,
                        origin,
                        neighborhood: (neighborhood.nx as u32) << 24
                            | (neighborhood.ny as u32) << 16
                            | (neighborhood.px as u32) << 8
                            | neighborhood.py as u32,
                    },
                );
            }
            self.shared
                .device
                .flush_mapped_memory_ranges(&[vk::MappedMemoryRange {
                    memory: self.instance_memory,
                    offset: 0,
                    size: vk::WHOLE_SIZE,
                    ..Default::default()
                }])
                .unwrap()
        }
        (count, state.transfer)
    }

    /// Allocate a cache slot to transfer texture data into
    pub fn allocate(&mut self, chunk: Chunk) -> Option<u32> {
        self.mgr.allocate(chunk)
    }

    /// Record commands into `cmd` to recover ownership of `slot` from a transfer queue
    ///
    /// If `slot` was transferred on queue family other than the `sample_queue_family` supplied to
    /// `new`, the commands recorded by this method must be executed before a draw is performed
    /// following the slot's release and a successive `update` call.
    pub fn acquire_ownership(&self, slot: u32, transfer_queue_family: u32, cmd: vk::CommandBuffer) {
        let array = slot / self.shared.array_size;
        let layer = slot % self.shared.array_size;
        let device = &self.shared.device;
        for kind in &self.shared.arrays {
            let image = kind.arrays[array as usize].image;
            unsafe {
                device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    kind.stages,
                    Default::default(),
                    &[],
                    &[],
                    &[vk::ImageMemoryBarrier {
                        dst_access_mask: vk::AccessFlags::SHADER_READ,
                        old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        src_queue_family_index: transfer_queue_family,
                        dst_queue_family_index: self.shared.sample_queue_family,
                        image,
                        subresource_range: vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: layer,
                            layer_count: 1,
                        },
                        ..Default::default()
                    }],
                );
            }
        }
    }

    /// Indicate that texture transfer into `slot` is complete
    #[inline]
    pub fn transferred(&mut self, slot: u32) {
        self.mgr.release(slot);
    }

    /// Buffer containing a sequence of `Instance`s to be rendered
    ///
    /// The number of instances to render is determined by calling `update`.
    #[inline]
    pub fn instance_buffer(&self) -> vk::Buffer {
        self.instance_buffer
    }

    /// For each element in `textures`, yields an iterator of `vk::ImageView`s of texture arrays
    pub fn array_views<'a>(
        &'a self,
    ) -> impl Iterator<Item = impl Iterator<Item = vk::ImageView> + 'a> + 'a {
        self.shared
            .arrays
            .iter()
            .map(|x| x.arrays.iter().map(|x| x.view))
    }

    /// For each element in `textures`, yields an iterator of `vk::Image`s of texture arrays
    pub fn arrays<'a>(&'a self) -> impl Iterator<Item = impl Iterator<Item = vk::Image> + 'a> {
        self.shared
            .arrays
            .iter()
            .map(|x| x.arrays.iter().map(|x| x.image))
    }

    /// The number of slots per `vk::ImageView` in `array_views`
    ///
    /// Use this constant to compute the exact `vk::ImageView` and layer corresponding to a given
    /// `ChunkInstance.slot`.
    #[inline]
    pub fn array_size(&self) -> u32 {
        self.shared.array_size
    }

    /// The number of texture arrays making up the cache (per texture type)
    #[inline]
    pub fn array_count(&self) -> u32 {
        self.shared.arrays[0].arrays.len() as u32
    }
}

/// A kind of per-chunk textures to be cached
///
/// A typical application might use three different kinds: one each for heightmaps, normals, and
/// color.
#[derive(Debug, Copy, Clone)]
pub struct TextureKind {
    pub format: vk::Format,
    pub extent: vk::Extent2D,
    /// Pipeline stages at which the caller will sample from this type of texture
    pub stages: vk::PipelineStageFlags,
}

struct ArraySet {
    extent: vk::Extent2D,
    stages: vk::PipelineStageFlags,
    arrays: Vec<TextureArray>,
}

struct TextureArray {
    memory: vk::DeviceMemory,
    image: vk::Image,
    view: vk::ImageView,
}

impl TextureArray {
    pub fn new(
        device: &Device,
        props: &vk::PhysicalDeviceMemoryProperties,
        info: &TextureKind,
        slots: u32,
    ) -> Self {
        unsafe {
            let image = device
                .create_image(
                    &vk::ImageCreateInfo {
                        image_type: vk::ImageType::TYPE_2D,
                        format: info.format,
                        extent: vk::Extent3D {
                            width: info.extent.width,
                            height: info.extent.height,
                            depth: 1,
                        },
                        mip_levels: 1,
                        array_layers: slots,
                        samples: vk::SampleCountFlags::TYPE_1,
                        tiling: vk::ImageTiling::OPTIMAL,
                        usage: vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
                        sharing_mode: vk::SharingMode::EXCLUSIVE,
                        initial_layout: vk::ImageLayout::UNDEFINED,
                        ..Default::default()
                    },
                    None,
                )
                .unwrap();
            let req = device.get_image_memory_requirements(image);
            let memory_index =
                find_memorytype_index(&req, props, vk::MemoryPropertyFlags::DEVICE_LOCAL)
                    .expect("unable to find suitable memory index for image");

            let memory = device
                .allocate_memory(
                    &vk::MemoryAllocateInfo {
                        allocation_size: req.size,
                        memory_type_index: memory_index,
                        ..Default::default()
                    },
                    None,
                )
                .unwrap();
            device.bind_image_memory(image, memory, 0).unwrap();

            let view = device
                .create_image_view(
                    &vk::ImageViewCreateInfo {
                        view_type: vk::ImageViewType::TYPE_2D_ARRAY,
                        format: info.format,
                        components: vk::ComponentMapping {
                            r: vk::ComponentSwizzle::R,
                            g: vk::ComponentSwizzle::G,
                            b: vk::ComponentSwizzle::B,
                            a: vk::ComponentSwizzle::A,
                        },
                        subresource_range: vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            level_count: 1,
                            layer_count: slots,
                            ..Default::default()
                        },
                        image,
                        ..Default::default()
                    },
                    None,
                )
                .unwrap();

            Self {
                image,
                memory,
                view,
            }
        }
    }
}

/// Structure contained in the `Cache`'s instance buffer
#[repr(C)]
pub struct ChunkInstance {
    pub worldview: na::Matrix4<f32>,
    pub coords: [u32; 2],
    pub depth: u32,
    pub slot: u32,
    pub origin: na::Point3<f32>,
    pub neighborhood: u32,
}

fn find_memorytype_index(
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

fn find_memorytype_index_f<F: Fn(vk::MemoryPropertyFlags, vk::MemoryPropertyFlags) -> bool>(
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

/// Source to transfer a chunk's texture from
///
pub struct TransferSource {
    /// Which `TextureKind` this corresponds to
    pub texture: usize,
    /// Staging buffer containing the texture data
    pub buffer: vk::Buffer,
    /// Where the texture begins within the buffer
    pub offset: vk::DeviceSize,
    /// See `vk::BufferImageCopy`
    pub row_length: u32,
    /// See `vk::BufferImageCopy`
    pub image_height: u32,
}

/// Helper for asynchronously loading texture data into the cache
///
/// When a separate transfer queue is used, this can be sent to another thread.
pub struct TransferEngine {
    shared: Arc<Shared>,
    transfer_queue_family: u32,
}

impl TransferEngine {
    /// Record commands into `cmd` to transfer a texture from a staging buffer identified by `src` into `slot`
    ///
    /// Must be called once for each kind of texture in use. If performed on a distinct queue, an
    /// execution dependency must be defined between reads taking place prior to the `Cache::update`
    /// call that yielded the `Chunk` for this `Slot` and the commands recorded here.
    pub fn transfer(&self, cmd: vk::CommandBuffer, src: TransferSource, slot: u32) {
        let TransferSource {
            texture,
            buffer,
            offset,
            row_length,
            image_height,
        } = src;
        let array = slot / self.shared.array_size;
        let layer = slot % self.shared.array_size;
        let kind = &self.shared.arrays[texture];
        let image = kind.arrays[array as usize].image;
        let device = &self.shared.device;
        unsafe {
            device.cmd_pipeline_barrier(
                cmd,
                kind.stages,
                vk::PipelineStageFlags::TRANSFER,
                Default::default(),
                &[],
                &[],
                &[vk::ImageMemoryBarrier {
                    src_access_mask: vk::AccessFlags::SHADER_READ,
                    dst_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                    old_layout: vk::ImageLayout::UNDEFINED,
                    new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    src_queue_family_index: self.shared.sample_queue_family,
                    dst_queue_family_index: self.transfer_queue_family,
                    image,
                    subresource_range: vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: layer,
                        layer_count: 1,
                    },
                    ..Default::default()
                }],
            );

            device.cmd_copy_buffer_to_image(
                cmd,
                buffer,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[vk::BufferImageCopy {
                    buffer_offset: offset,
                    buffer_row_length: row_length,
                    buffer_image_height: image_height,
                    image_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: layer,
                        layer_count: 1,
                    },
                    image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                    image_extent: vk::Extent3D {
                        width: kind.extent.width,
                        height: kind.extent.height,
                        depth: 1,
                    },
                }],
            );

            if self.shared.sample_queue_family != self.transfer_queue_family {
                device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    Default::default(),
                    &[],
                    &[],
                    &[vk::ImageMemoryBarrier {
                        src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                        old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        src_queue_family_index: self.transfer_queue_family,
                        dst_queue_family_index: self.shared.sample_queue_family,
                        image,
                        subresource_range: vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: layer,
                            layer_count: 1,
                        },
                        ..Default::default()
                    }],
                );
            } else {
                device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::TRANSFER,
                    kind.stages,
                    Default::default(),
                    &[],
                    &[],
                    &[vk::ImageMemoryBarrier {
                        src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                        dst_access_mask: vk::AccessFlags::SHADER_READ,
                        old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                        dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                        image,
                        subresource_range: vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: layer,
                            layer_count: 1,
                        },
                        ..Default::default()
                    }],
                );
            }
        }
    }
}

struct Shared {
    device: Arc<Device>,
    array_size: u32,
    arrays: Vec<ArraySet>,
    /// The family textures will be sampled from
    sample_queue_family: u32,
}

impl Drop for Shared {
    fn drop(&mut self) {
        for array in self.arrays.iter().flat_map(|x| x.arrays.iter()) {
            unsafe {
                self.device.destroy_image_view(array.view, None);
                self.device.destroy_image(array.image, None);
                self.device.free_memory(array.memory, None);
            }
        }
    }
}
