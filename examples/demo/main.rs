mod planet;
mod window;

use std::ffi::CStr;
use std::time::Instant;
use std::{mem, slice};

use ash::vk;
use half::f16;
use memoffset::offset_of;
use vk_shader_macros::include_glsl;

use planetmap::ash::ChunkInstance;

use planet::Planet;
use window::*;

const VERT: &[u32] = include_glsl!("examples/demo/terrain.vert", debug);
const FRAG: &[u32] = include_glsl!("examples/demo/terrain.frag", debug);

/// Number of samples along one edge of a chunk. Must be at least 2.
const CHUNK_HEIGHT_SIZE: u32 = 17;
/// Number of quads along one edge of a chunk. Must be a power of two for stitching to work.
const CHUNK_QUADS: u32 = CHUNK_HEIGHT_SIZE - 1;
const CHUNK_NORMALS_SIZE: u32 = CHUNK_HEIGHT_SIZE * 3;
const CHUNK_COLORS_SIZE: u32 = CHUNK_HEIGHT_SIZE * 3;
/// Amount of CPU-side staging memory to allocate for originating transfers
const STAGING_BUFFER_LENGTH: u32 = 256;

fn main() {
    unsafe {
        let mut base = ExampleBase::new(512, 512);
        let planet = Planet::new();

        let atmosphere_builder = fuzzyblue::Builder::new(
            &base.instance,
            base.device.clone(),
            vk::PipelineCache::null(),
            base.pdevice,
            base.queue_family_index,
            None,
        );

        let atmosphere_cmd = base
            .device
            .allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::builder()
                    .command_buffer_count(1)
                    .command_pool(base.pool)
                    .level(vk::CommandBufferLevel::PRIMARY),
            )
            .unwrap()[0];
        let atmosphere = record_submit_commandbuffer(
            &*base.device,
            atmosphere_cmd,
            base.present_queue,
            &[],
            &[],
            &[],
            || {
                atmosphere_builder.build(
                    atmosphere_cmd,
                    &fuzzyblue::Params {
                        r_planet: planet.radius(),
                        ..Default::default()
                    },
                )
            },
        );
        let atmosphere = atmosphere.assert_ready();

        let mut cache = planetmap::ash::Cache::new(
            &base.instance,
            base.pdevice,
            base.device.clone(),
            planetmap::cache::Config { max_depth: 12 },
            &[
                planetmap::ash::TextureKind {
                    format: vk::Format::R16_SFLOAT,
                    extent: vk::Extent2D {
                        width: CHUNK_HEIGHT_SIZE,
                        height: CHUNK_HEIGHT_SIZE,
                    },
                    stages: vk::PipelineStageFlags::VERTEX_SHADER,
                },
                planetmap::ash::TextureKind {
                    format: vk::Format::R8G8_SNORM,
                    extent: vk::Extent2D {
                        width: CHUNK_NORMALS_SIZE,
                        height: CHUNK_NORMALS_SIZE,
                    },
                    stages: vk::PipelineStageFlags::FRAGMENT_SHADER,
                },
                planetmap::ash::TextureKind {
                    format: vk::Format::R8G8B8A8_SRGB,
                    extent: vk::Extent2D {
                        width: CHUNK_COLORS_SIZE,
                        height: CHUNK_COLORS_SIZE,
                    },
                    stages: vk::PipelineStageFlags::FRAGMENT_SHADER,
                },
            ],
            base.queue_family_index,
        );
        let engine = cache.transfer_engine(base.queue_family_index);

        let renderpass_attachments = [
            vk::AttachmentDescription {
                format: base.surface_format.format,
                samples: vk::SampleCountFlags::TYPE_1,
                load_op: vk::AttachmentLoadOp::CLEAR,
                store_op: vk::AttachmentStoreOp::STORE,
                final_layout: vk::ImageLayout::PRESENT_SRC_KHR,
                ..Default::default()
            },
            vk::AttachmentDescription {
                format: vk::Format::D32_SFLOAT,
                samples: vk::SampleCountFlags::TYPE_1,
                load_op: vk::AttachmentLoadOp::CLEAR,
                initial_layout: vk::ImageLayout::UNDEFINED,
                final_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                ..Default::default()
            },
        ];
        let color_attachment_refs = [vk::AttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        }];
        let depth_attachment_ref = vk::AttachmentReference {
            attachment: 1,
            layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        };
        let dependencies = [vk::SubpassDependency {
            src_subpass: vk::SUBPASS_EXTERNAL,
            src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_READ
                | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            ..Default::default()
        }];

        let subpasses = [vk::SubpassDescription::builder()
            .color_attachments(&color_attachment_refs)
            .depth_stencil_attachment(&depth_attachment_ref)
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .build()];

        let renderpass_create_info = vk::RenderPassCreateInfo::builder()
            .attachments(&renderpass_attachments)
            .subpasses(&subpasses)
            .dependencies(&dependencies);

        let renderpass = base
            .device
            .create_render_pass(&renderpass_create_info, None)
            .unwrap();

        let atmosphere_renderer = fuzzyblue::Renderer::new(
            base.device.clone(),
            vk::PipelineCache::null(),
            true,
            renderpass,
        );

        let uniform_buffer_info = vk::BufferCreateInfo {
            size: mem::size_of::<Uniforms>() as u64,
            usage: vk::BufferUsageFlags::UNIFORM_BUFFER,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };
        let uniform_buffer = base
            .device
            .create_buffer(&uniform_buffer_info, None)
            .unwrap();
        let uniform_buffer_memory_req = base.device.get_buffer_memory_requirements(uniform_buffer);
        let uniform_buffer_memory_index = find_memorytype_index(
            &uniform_buffer_memory_req,
            &base.device_memory_properties,
            vk::MemoryPropertyFlags::HOST_VISIBLE,
        )
        .expect("Unable to find suitable memory type for the uniform buffer.");

        let uniform_buffer_allocate_info = vk::MemoryAllocateInfo {
            allocation_size: uniform_buffer_memory_req.size,
            memory_type_index: uniform_buffer_memory_index,
            ..Default::default()
        };
        let uniform_buffer_memory = base
            .device
            .allocate_memory(&uniform_buffer_allocate_info, None)
            .unwrap();
        let uniform_ptr = base
            .device
            .map_memory(
                uniform_buffer_memory,
                0,
                uniform_buffer_memory_req.size,
                vk::MemoryMapFlags::empty(),
            )
            .unwrap() as *mut Uniforms;
        base.device
            .bind_buffer_memory(uniform_buffer, uniform_buffer_memory, 0)
            .unwrap();
        let uniforms = &mut *uniform_ptr;

        let staging_buffer_info = vk::BufferCreateInfo {
            size: (STAGED_CHUNK_SIZE * STAGING_BUFFER_LENGTH) as u64,
            usage: vk::BufferUsageFlags::TRANSFER_SRC,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };
        let staging_buffer = base
            .device
            .create_buffer(&staging_buffer_info, None)
            .unwrap();
        let staging_buffer_memory_req = base.device.get_buffer_memory_requirements(staging_buffer);
        let staging_buffer_memory_index = find_memorytype_index(
            &staging_buffer_memory_req,
            &base.device_memory_properties,
            vk::MemoryPropertyFlags::HOST_VISIBLE,
        )
        .expect("Unable to find suitable memorytype for the staging buffer.");

        let staging_buffer_allocate_info = vk::MemoryAllocateInfo {
            allocation_size: staging_buffer_memory_req.size,
            memory_type_index: staging_buffer_memory_index,
            ..Default::default()
        };
        let staging_buffer_memory = base
            .device
            .allocate_memory(&staging_buffer_allocate_info, None)
            .unwrap();
        let staging_ptr = base
            .device
            .map_memory(
                staging_buffer_memory,
                0,
                staging_buffer_memory_req.size,
                vk::MemoryMapFlags::empty(),
            )
            .unwrap() as *mut [u8; STAGED_CHUNK_SIZE as usize];
        base.device
            .bind_buffer_memory(staging_buffer, staging_buffer_memory, 0)
            .unwrap();
        let staging = slice::from_raw_parts_mut(staging_ptr, STAGING_BUFFER_LENGTH as usize);

        let sampler_info = vk::SamplerCreateInfo {
            mag_filter: vk::Filter::LINEAR,
            min_filter: vk::Filter::LINEAR,
            mipmap_mode: vk::SamplerMipmapMode::NEAREST,
            address_mode_u: vk::SamplerAddressMode::CLAMP_TO_EDGE,
            address_mode_v: vk::SamplerAddressMode::CLAMP_TO_EDGE,
            address_mode_w: vk::SamplerAddressMode::CLAMP_TO_EDGE,
            max_anisotropy: 1.0,
            border_color: vk::BorderColor::FLOAT_TRANSPARENT_BLACK,
            compare_op: vk::CompareOp::NEVER,
            ..Default::default()
        };

        let sampler = base.device.create_sampler(&sampler_info, None).unwrap();

        let descriptor_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: 1,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: cache.array_count() * 3,
            },
        ];
        let descriptor_pool_info = vk::DescriptorPoolCreateInfo::builder()
            .pool_sizes(&descriptor_sizes)
            .max_sets(1);

        let descriptor_pool = base
            .device
            .create_descriptor_pool(&descriptor_pool_info, None)
            .unwrap();
        let samplers = (0..cache.array_count())
            .map(|_| sampler)
            .collect::<Vec<_>>();
        let desc_layout_bindings = [
            vk::DescriptorSetLayoutBinding {
                descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                ..Default::default()
            },
            vk::DescriptorSetLayoutBinding {
                binding: 1,
                descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: cache.array_count(),
                stage_flags: vk::ShaderStageFlags::VERTEX,
                p_immutable_samplers: samplers.as_ptr(),
                ..Default::default()
            },
            vk::DescriptorSetLayoutBinding {
                binding: 2,
                descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: cache.array_count(),
                stage_flags: vk::ShaderStageFlags::FRAGMENT,
                p_immutable_samplers: samplers.as_ptr(),
                ..Default::default()
            },
            vk::DescriptorSetLayoutBinding {
                binding: 3,
                descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: cache.array_count(),
                stage_flags: vk::ShaderStageFlags::FRAGMENT,
                p_immutable_samplers: samplers.as_ptr(),
                ..Default::default()
            },
        ];

        let desc_set_layouts = [base
            .device
            .create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::builder().bindings(&desc_layout_bindings),
                None,
            )
            .unwrap()];

        let desc_alloc_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&desc_set_layouts);
        let descriptor_set = base
            .device
            .allocate_descriptor_sets(&desc_alloc_info)
            .unwrap()
            .into_iter()
            .next()
            .unwrap();

        let mut cache_views = cache.array_views();

        base.device.update_descriptor_sets(
            &[
                vk::WriteDescriptorSet {
                    dst_set: descriptor_set,
                    dst_binding: 0,
                    descriptor_count: 1,
                    descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                    p_buffer_info: &vk::DescriptorBufferInfo {
                        buffer: uniform_buffer,
                        offset: 0,
                        range: mem::size_of::<Uniforms>() as u64,
                    },
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    dst_set: descriptor_set,
                    dst_binding: 1,
                    descriptor_count: cache.array_count(),
                    descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    p_image_info: cache_views
                        .next()
                        .unwrap()
                        .map(|x| vk::DescriptorImageInfo {
                            sampler: vk::Sampler::null(),
                            image_view: x,
                            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        })
                        .collect::<Vec<_>>()[..]
                        .as_ptr(),
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    dst_set: descriptor_set,
                    dst_binding: 2,
                    descriptor_count: cache.array_count(),
                    descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    p_image_info: cache_views
                        .next()
                        .unwrap()
                        .map(|x| vk::DescriptorImageInfo {
                            sampler: vk::Sampler::null(),
                            image_view: x,
                            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        })
                        .collect::<Vec<_>>()[..]
                        .as_ptr(),
                    ..Default::default()
                },
                vk::WriteDescriptorSet {
                    dst_set: descriptor_set,
                    dst_binding: 3,
                    descriptor_count: cache.array_count(),
                    descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    p_image_info: cache_views
                        .next()
                        .unwrap()
                        .map(|x| vk::DescriptorImageInfo {
                            sampler: vk::Sampler::null(),
                            image_view: x,
                            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        })
                        .collect::<Vec<_>>()[..]
                        .as_ptr(),
                    ..Default::default()
                },
            ],
            &[],
        );
        assert!(cache_views.next().is_none());
        mem::drop(cache_views);

        let vertex_shader_info = vk::ShaderModuleCreateInfo::builder().code(&VERT);
        let frag_shader_info = vk::ShaderModuleCreateInfo::builder().code(&FRAG);

        let vertex_shader_module = base
            .device
            .create_shader_module(&vertex_shader_info, None)
            .expect("Vertex shader module error");

        let fragment_shader_module = base
            .device
            .create_shader_module(&frag_shader_info, None)
            .expect("Fragment shader module error");

        let pipeline_layout = base
            .device
            .create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::builder().set_layouts(&desc_set_layouts),
                None,
            )
            .unwrap();

        let vertex_input_binding_descriptions = [vk::VertexInputBindingDescription {
            binding: 0,
            stride: mem::size_of::<ChunkInstance>() as u32,
            input_rate: vk::VertexInputRate::INSTANCE,
        }];
        let vertex_input_attribute_descriptions = [
            vk::VertexInputAttributeDescription {
                location: 0,
                binding: 0,
                format: vk::Format::R32G32B32A32_SFLOAT,
                offset: offset_of!(ChunkInstance, worldview) as u32,
            },
            vk::VertexInputAttributeDescription {
                location: 1,
                binding: 0,
                format: vk::Format::R32G32B32A32_SFLOAT,
                offset: offset_of!(ChunkInstance, worldview) as u32 + 16,
            },
            vk::VertexInputAttributeDescription {
                location: 2,
                binding: 0,
                format: vk::Format::R32G32B32A32_SFLOAT,
                offset: offset_of!(ChunkInstance, worldview) as u32 + 32,
            },
            vk::VertexInputAttributeDescription {
                location: 3,
                binding: 0,
                format: vk::Format::R32G32B32A32_SFLOAT,
                offset: offset_of!(ChunkInstance, worldview) as u32 + 48,
            },
            vk::VertexInputAttributeDescription {
                location: 4,
                binding: 0,
                format: vk::Format::R32G32B32A32_UINT, // Incorporates depth and slot as well
                offset: offset_of!(ChunkInstance, coords) as u32,
            },
            vk::VertexInputAttributeDescription {
                location: 5,
                binding: 0,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: offset_of!(ChunkInstance, origin) as u32,
            },
            vk::VertexInputAttributeDescription {
                location: 6,
                binding: 0,
                format: vk::Format::R32_UINT,
                offset: offset_of!(ChunkInstance, neighborhood) as u32,
            },
        ];

        let vertex_input_state_info = vk::PipelineVertexInputStateCreateInfo {
            vertex_attribute_description_count: vertex_input_attribute_descriptions.len() as u32,
            p_vertex_attribute_descriptions: vertex_input_attribute_descriptions.as_ptr(),
            vertex_binding_description_count: vertex_input_binding_descriptions.len() as u32,
            p_vertex_binding_descriptions: vertex_input_binding_descriptions.as_ptr(),
            ..Default::default()
        };
        let vertex_input_assembly_state_info = vk::PipelineInputAssemblyStateCreateInfo {
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            ..Default::default()
        };

        let rasterization_info = vk::PipelineRasterizationStateCreateInfo {
            front_face: vk::FrontFace::COUNTER_CLOCKWISE,
            line_width: 1.0,
            polygon_mode: vk::PolygonMode::FILL,
            cull_mode: vk::CullModeFlags::BACK,
            ..Default::default()
        };
        let multisample_state_info = vk::PipelineMultisampleStateCreateInfo {
            rasterization_samples: vk::SampleCountFlags::TYPE_1,
            ..Default::default()
        };
        let noop_stencil_state = vk::StencilOpState {
            fail_op: vk::StencilOp::KEEP,
            pass_op: vk::StencilOp::KEEP,
            depth_fail_op: vk::StencilOp::KEEP,
            compare_op: vk::CompareOp::ALWAYS,
            ..Default::default()
        };
        let depth_state_info = vk::PipelineDepthStencilStateCreateInfo {
            depth_test_enable: 1,
            depth_write_enable: 1,
            depth_compare_op: vk::CompareOp::GREATER_OR_EQUAL,
            front: noop_stencil_state.clone(),
            back: noop_stencil_state.clone(),
            max_depth_bounds: 1.0,
            ..Default::default()
        };
        let color_blend_attachment_states = [vk::PipelineColorBlendAttachmentState {
            blend_enable: vk::FALSE,
            color_write_mask: vk::ColorComponentFlags::all(),
            ..Default::default()
        }];
        let color_blend_state = vk::PipelineColorBlendStateCreateInfo::builder()
            .logic_op(vk::LogicOp::CLEAR)
            .attachments(&color_blend_attachment_states);

        let dynamic_state = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state_info =
            vk::PipelineDynamicStateCreateInfo::builder().dynamic_states(&dynamic_state);

        let specialization = mem::transmute::<_, [u8; 16]>(Specialization {
            quad_count: CHUNK_QUADS,
            heightmap_array_size: cache.array_size(),
            heightmap_array_count: cache.array_count(),
            radius: planet.radius(),
        });
        let specialization_map = [
            vk::SpecializationMapEntry {
                constant_id: 0,
                offset: offset_of!(Specialization, quad_count) as u32,
                size: 4,
            },
            vk::SpecializationMapEntry {
                constant_id: 1,
                offset: offset_of!(Specialization, heightmap_array_size) as u32,
                size: 4,
            },
            vk::SpecializationMapEntry {
                constant_id: 2,
                offset: offset_of!(Specialization, heightmap_array_count) as u32,
                size: 4,
            },
            vk::SpecializationMapEntry {
                constant_id: 3,
                offset: offset_of!(Specialization, radius) as u32,
                size: 4,
            },
        ];
        let specialization_info = vk::SpecializationInfo::builder()
            .data(&specialization)
            .map_entries(&specialization_map);

        let shader_entry_name = CStr::from_bytes_with_nul(b"main\0").unwrap();
        let graphics_pipeline = base
            .device
            .create_graphics_pipelines(
                vk::PipelineCache::null(),
                &[vk::GraphicsPipelineCreateInfo::builder()
                    .stages(&[
                        vk::PipelineShaderStageCreateInfo {
                            module: vertex_shader_module,
                            p_name: shader_entry_name.as_ptr(),
                            stage: vk::ShaderStageFlags::VERTEX,
                            p_specialization_info: &*specialization_info,
                            ..Default::default()
                        },
                        vk::PipelineShaderStageCreateInfo {
                            module: fragment_shader_module,
                            p_name: shader_entry_name.as_ptr(),
                            stage: vk::ShaderStageFlags::FRAGMENT,
                            p_specialization_info: &*specialization_info,
                            ..Default::default()
                        },
                    ])
                    .vertex_input_state(&vertex_input_state_info)
                    .input_assembly_state(&vertex_input_assembly_state_info)
                    .viewport_state(&vk::PipelineViewportStateCreateInfo {
                        scissor_count: 1,
                        viewport_count: 1,
                        ..Default::default()
                    })
                    .rasterization_state(&rasterization_info)
                    .multisample_state(&multisample_state_info)
                    .depth_stencil_state(&depth_state_info)
                    .color_blend_state(&color_blend_state)
                    .dynamic_state(&dynamic_state_info)
                    .layout(pipeline_layout)
                    .render_pass(renderpass)
                    .build()],
                None,
            )
            .expect("Unable to create graphics pipeline")
            .into_iter()
            .next()
            .unwrap();

        let mut swapchain = SwapchainState::new(&base, renderpass, None);

        let mut camera = na::IsometryMatrix3::from_parts(
            na::Translation3::from(na::Vector3::new(0.0, planet.radius() as f64 + 1e4, 0.0)),
            na::Rotation3::identity(),
        );

        use winit::ElementState::*;
        let mut panning = Released;
        let mut left = Released;
        let mut right = Released;
        let mut forward = Released;
        let mut back = Released;
        let mut up = Released;
        let mut down = Released;
        let mut roll_left = Released;
        let mut roll_right = Released;
        let mut sprint = Released;
        let mut walk = Released;

        let mut t0 = Instant::now();
        loop {
            let mut keep_going = true;
            base.events_loop.poll_events(|e| {
                use winit::WindowEvent::*;
                use winit::*;
                match e {
                    Event::WindowEvent { event, .. } => match event {
                        CloseRequested => {
                            keep_going = false;
                        }
                        MouseInput {
                            button: MouseButton::Left,
                            state,
                            ..
                        } => {
                            panning = state;
                        }
                        WindowEvent::KeyboardInput {
                            input:
                                winit::KeyboardInput {
                                    state,
                                    virtual_keycode: Some(key),
                                    ..
                                },
                            ..
                        } => {
                            use VirtualKeyCode::*;
                            match key {
                                A => {
                                    left = state;
                                }
                                D => {
                                    right = state;
                                }
                                W => {
                                    forward = state;
                                }
                                S => {
                                    back = state;
                                }
                                R => {
                                    up = state;
                                }
                                F => {
                                    down = state;
                                }
                                Q => {
                                    roll_left = state;
                                }
                                E => {
                                    roll_right = state;
                                }
                                LShift => {
                                    sprint = state;
                                }
                                LControl => {
                                    walk = state;
                                }
                                _ => {}
                            }
                        }
                        _ => {}
                    },
                    Event::DeviceEvent { event, .. } => {
                        use winit::DeviceEvent::*;
                        match event {
                            MouseMotion { delta: (x, y) } if panning == Pressed => {
                                camera = camera
                                    * na::Rotation3::from_axis_angle(
                                        &na::Vector3::y_axis(),
                                        -x * 0.003,
                                    )
                                    * na::Rotation3::from_axis_angle(
                                        &na::Vector3::x_axis(),
                                        -y * 0.003,
                                    );
                            }
                            _ => {}
                        }
                    }
                    _ => {}
                }
            });
            if !keep_going {
                break;
            }

            let t1 = Instant::now();
            let dt = t1 - t0;
            let dt = dt.as_secs() as f64 + dt.subsec_nanos() as f64 * 1e-9;
            t0 = t1;

            camera = camera
                * na::Rotation3::from_axis_angle(
                    &na::Vector3::z_axis(),
                    (if roll_left == Pressed { 1.0 } else { 0.0 }
                     + if roll_right == Pressed { -1.0 } else { 0.0 })
                        * dt,
                );

            let mut motion = na::Vector3::zeros();
            if left == Pressed {
                motion.x -= 1.0;
            }
            if right == Pressed {
                motion.x += 1.0;
            }
            if forward == Pressed {
                motion.z -= 1.0;
            }
            if back == Pressed {
                motion.z += 1.0;
            }
            if up == Pressed {
                motion.y += 1.0;
            }
            if down == Pressed {
                motion.y -= 1.0;
            }
            let altitude = camera.translation.vector.norm() - planet.radius() as f64;
            let speed = altitude
                * if sprint == Pressed { 3.0 } else { 1.0 }
                * if walk == Pressed { 1.0 / 3.0 } else { 1.0 };
            camera = camera
                * na::Translation3::from(motion * dt * if speed > 1e8 { 1e8 } else { speed });

            let swapchain_suboptimal;
            let present_index = loop {
                match swapchain.acquire_next_image(base.present_complete_semaphore) {
                    Ok((idx, suboptimal)) => {
                        swapchain_suboptimal = suboptimal;
                        break idx;
                    }
                    Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                        swapchain = SwapchainState::new(&base, renderpass, Some(swapchain));
                    }
                    Err(e) => {
                        panic!("{}", e);
                    }
                }
            };
            let clear_values = [
                vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.0, 0.0, 0.0, 0.0],
                    },
                },
                vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 0.0,
                        stencil: 0,
                    },
                },
            ];

            let viewports = [vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: swapchain.extent.width as f32,
                height: swapchain.extent.height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            }];
            let scissors = [vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: swapchain.extent,
            }];

            let render_pass_begin_info = vk::RenderPassBeginInfo::builder()
                .render_pass(renderpass)
                .framebuffer(swapchain.frames[present_index as usize].buffer)
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: swapchain.extent,
                })
                .clear_values(&clear_values);

            let viewport =
                Viewport::from_vertical_fov(swapchain.extent, std::f32::consts::FRAC_PI_2);
            uniforms.projection = viewport.projection(1e-2);
            uniforms.view = na::convert(camera.inverse());
            let view = camera.inverse();
            let (instances, transfers) = cache.update(planet.radius() as f64, &view);

            let mut transfer_slots = Vec::new();
            record_submit_commandbuffer(
                &*base.device,
                base.draw_command_buffer,
                base.present_queue,
                &[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT],
                &[base.present_complete_semaphore],
                &[base.rendering_complete_semaphore],
                || {
                    let device = &*base.device;
                    let cmd = base.draw_command_buffer;
                    if !transfers.is_empty() {
                        let base = staging.as_ptr() as usize;
                        for (stage, chunk) in staging.iter_mut().zip(transfers) {
                            let stage = stage.as_ptr() as *mut StagedChunk;
                            let slot = cache.allocate(chunk).unwrap();
                            planet.generate_chunk(
                                &chunk,
                                CHUNK_HEIGHT_SIZE,
                                &mut (*stage).heights.0[..],
                                CHUNK_NORMALS_SIZE,
                                &mut (*stage).normals.0[..],
                                CHUNK_COLORS_SIZE,
                                &mut (*stage).colors.0[..],
                            );
                            let offset = stage as usize - base;
                            engine.transfer(
                                cmd,
                                planetmap::ash::TransferSource {
                                    texture: 0,
                                    buffer: staging_buffer,
                                    offset: (offset + offset_of!(StagedChunk, heights))
                                        as vk::DeviceSize,
                                    row_length: 0,
                                    image_height: 0,
                                },
                                slot,
                            );
                            engine.transfer(
                                cmd,
                                planetmap::ash::TransferSource {
                                    texture: 1,
                                    buffer: staging_buffer,
                                    offset: (offset + offset_of!(StagedChunk, normals))
                                        as vk::DeviceSize,
                                    row_length: 0,
                                    image_height: 0,
                                },
                                slot,
                            );
                            engine.transfer(
                                cmd,
                                planetmap::ash::TransferSource {
                                    texture: 2,
                                    buffer: staging_buffer,
                                    offset: (offset + offset_of!(StagedChunk, colors))
                                        as vk::DeviceSize,
                                    row_length: 0,
                                    image_height: 0,
                                },
                                slot,
                            );
                            transfer_slots.push(slot);
                        }
                        device
                            .flush_mapped_memory_ranges(&[vk::MappedMemoryRange {
                                memory: staging_buffer_memory,
                                offset: 0,
                                size: vk::WHOLE_SIZE,
                                ..Default::default()
                            }])
                            .unwrap();
                    }

                    device.cmd_begin_render_pass(
                        cmd,
                        &render_pass_begin_info,
                        vk::SubpassContents::INLINE,
                    );
                    if instances > 0 {
                        device.cmd_bind_descriptor_sets(
                            cmd,
                            vk::PipelineBindPoint::GRAPHICS,
                            pipeline_layout,
                            0,
                            &[descriptor_set],
                            &[],
                        );
                        device.cmd_bind_pipeline(
                            cmd,
                            vk::PipelineBindPoint::GRAPHICS,
                            graphics_pipeline,
                        );
                        device.cmd_set_viewport(cmd, 0, &viewports);
                        device.cmd_set_scissor(cmd, 0, &scissors);
                        device.cmd_bind_vertex_buffers(cmd, 0, &[cache.instance_buffer()], &[0]);
                        device.cmd_draw(cmd, CHUNK_QUADS * CHUNK_QUADS * 6, instances, 0, 0);
                    }
                    let (zenith, height) = na::Unit::new_and_get(camera.translation.vector);
                    atmosphere_renderer.draw(
                        cmd,
                        &atmosphere,
                        &fuzzyblue::DrawParams {
                            inverse_viewproj: (*(uniforms.projection
                                * na::convert::<_, na::Isometry3<f32>>(view))
                            .inverse()
                            .matrix())
                            .into(),
                            zenith: na::convert::<_, na::Vector3<f32>>(zenith.into_inner()).into(),
                            sun_direction: [0.0, 1.0, 0.0],
                            height: (height - planet.radius() as f64) as f32,
                            mie_anisotropy: fuzzyblue::MIE_ANISOTROPY_AIR,
                            // Scale down to usable SDR values
                            solar_irradiance: [
                                fuzzyblue::SOL_IRRADIANCE[0] * 6e-3,
                                fuzzyblue::SOL_IRRADIANCE[1] * 6e-3,
                                fuzzyblue::SOL_IRRADIANCE[2] * 6e-3,
                            ],
                        },
                        swapchain.extent,
                    );
                    device.cmd_end_render_pass(cmd);
                },
            );

            // Commands complete, so is the transfer.
            for slot in transfer_slots.drain(..) {
                cache.transferred(slot);
            }

            let out_of_date = match base.swapchain_loader.queue_present(
                base.present_queue,
                &vk::PresentInfoKHR::builder()
                    .wait_semaphores(&[base.rendering_complete_semaphore])
                    .swapchains(&[swapchain.handle])
                    .image_indices(&[present_index]),
            ) {
                Ok(true) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => true,
                Ok(false) => swapchain_suboptimal,
                Err(e) => panic!("{}", e),
            };
            if out_of_date {
                // Wait for present to finish
                base.device.queue_wait_idle(base.present_queue).unwrap();
                swapchain = SwapchainState::new(&base, renderpass, Some(swapchain));
            }
        }

        base.device.device_wait_idle().unwrap();
        base.device.destroy_pipeline(graphics_pipeline, None);
        base.device.destroy_pipeline_layout(pipeline_layout, None);
        base.device
            .destroy_shader_module(vertex_shader_module, None);
        base.device
            .destroy_shader_module(fragment_shader_module, None);
        base.device.free_memory(staging_buffer_memory, None);
        base.device.destroy_buffer(staging_buffer, None);
        base.device.free_memory(uniform_buffer_memory, None);
        base.device.destroy_buffer(uniform_buffer, None);
        for &descriptor_set_layout in desc_set_layouts.iter() {
            base.device
                .destroy_descriptor_set_layout(descriptor_set_layout, None);
        }
        base.device.destroy_descriptor_pool(descriptor_pool, None);
        base.device.destroy_sampler(sampler, None);
        base.device.destroy_render_pass(renderpass, None);
    }
}

#[repr(C)]
struct Uniforms {
    projection: na::Projective3<f32>,
    view: na::Projective3<f32>,
}

struct Viewport {
    left: f32,
    right: f32,
    bottom: f32,
    top: f32,
}

#[repr(align(4))]
struct TransferAligned<T: Copy>(T);

#[repr(C)]
struct StagedChunk {
    heights: TransferAligned<[f16; (CHUNK_HEIGHT_SIZE * CHUNK_HEIGHT_SIZE) as usize]>,
    normals: TransferAligned<[[i8; 2]; (CHUNK_NORMALS_SIZE * CHUNK_NORMALS_SIZE) as usize]>,
    colors: TransferAligned<[[u8; 4]; (CHUNK_COLORS_SIZE * CHUNK_COLORS_SIZE) as usize]>,
}

const STAGED_CHUNK_SIZE: u32 = least_greater_multiple(mem::size_of::<StagedChunk>() as u32, 4);

struct Specialization {
    quad_count: u32,
    heightmap_array_size: u32,
    heightmap_array_count: u32,
    radius: f32,
}

impl Viewport {
    fn from_vertical_fov(extent: vk::Extent2D, vfov: f32) -> Self {
        let aspect = extent.width as f32 / extent.height as f32;
        let top = (vfov / 2.0).tan();
        let right = aspect * top;
        Viewport {
            left: -right,
            right,
            bottom: -top,
            top,
        }
    }

    #[rustfmt::skip]
    fn projection(&self, znear: f32) -> na::Projective3<f32> {
        let idx = 1.0 / (self.right - self.left);
        let idy = 1.0 / (self.bottom - self.top);
        let sx = self.right + self.left;
        let sy = self.bottom + self.top;
        na::Projective3::from_matrix_unchecked(
            na::Matrix4::new(
                2.0 * idx,       0.0, sx * idx, 0.0,
                0.0, 2.0 * idy, sy * idy, 0.0,
                0.0,       0.0,      0.0, znear,
                0.0,       0.0,     -1.0, 0.0))
    }
}

/// Compute smallest multiple of `factor` which is >= `x`
const fn least_greater_multiple(x: u32, factor: u32) -> u32 {
    x + (factor - x % factor)
}
