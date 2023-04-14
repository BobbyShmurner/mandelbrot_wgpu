use wgpu::util::DeviceExt;
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

use bracket_color::prelude::*;
use tracing::{error, info, warn};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

trait ToWGPUColor {
    fn to_wgpu(&self) -> wgpu::Color;
}

impl ToWGPUColor for RGBA {
    fn to_wgpu(&self) -> wgpu::Color {
        wgpu::Color {
            r: self.r as f64,
            g: self.g as f64,
            b: self.b as f64,
            a: self.a as f64,
        }
    }
}

impl ToWGPUColor for RGB {
    fn to_wgpu(&self) -> wgpu::Color {
        wgpu::Color {
            r: self.r as f64,
            g: self.g as f64,
            b: self.b as f64,
            a: 1.0,
        }
    }
}

impl ToWGPUColor for HSV {
    fn to_wgpu(&self) -> wgpu::Color {
        self.to_rgb().to_wgpu()
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    color: [f32; 3],
}

impl Vertex {
    const ATTRIBS: [wgpu::VertexAttribute; 2] =
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3];

    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        use std::mem;

        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct View {
    position: [f32; 3],
    zoom: f32,
}

// impl View {
//     fn desc<'a>() -> wgpu::Buffer<'a> {
//         use std::mem;

//         wgpu::BufferDescriptor {
//             label: Some("View Buffer"),
//             size: mem::size_of::<Self>() as u64,
//             usage: wgpu::BufferUsages::UNIFORM,
//             mapped_at_creation: true,
//         }
//     }
// }

const VERTICES: &[Vertex] = &[
    Vertex {
        position: [-1.0, -1.0, 0.0],
        color: [0.5, 0.0, 0.5],
    },
    Vertex {
        position: [1.0, -1.0, 0.0],
        color: [0.5, 0.0, 0.5],
    },
    Vertex {
        position: [1.0, 1.0, 0.0],
        color: [0.5, 0.0, 0.5],
    },
    Vertex {
        position: [-1.0, 1.0, 0.0],
        color: [0.5, 0.0, 0.5],
    },
];

const INDICES: &[u16] = &[0, 1, 2, 2, 3, 0];

struct State {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    window: Window,
    view: View,
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    view_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    num_indices: u32,
    clear_col: RGBA,
}

impl State {
    // Creating some of the wgpu types requires async code
    async fn new(window: Window) -> Self {
        // The instance is a handle to our GPU
        // Backends::all => Vulkan + Metal + DX12 + Browser WebGPU
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            dx12_shader_compiler: Default::default(),
        });

        // # Safety
        //
        // The surface needs to live as long as the window that created it.
        // State owns the window so this should be safe.
        let surface = unsafe { instance.create_surface(&window) }.unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        // let backends = wgpu::util::backend_bits_from_env().unwrap_or_else(wgpu::Backends::all);
        // let dx12_shader_compiler = wgpu::util::dx12_shader_compiler_from_env().unwrap_or_default();

        // let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        //     backends,
        //     dx12_shader_compiler,
        // });
        // let (size, surface) = unsafe {
        //     let size = window.inner_size();
        //     let surface = instance.create_surface(&window).unwrap();

        //     (size, surface)
        // };
        // let adapter =
        //     wgpu::util::initialize_adapter_from_env_or_default(&instance, backends, Some(&surface))
        //         .await
        //         .expect("No suitable GPU adapters found on the system!");

        info!("Adapter: {:#?}", adapter.get_info());
        info!("Adapter Features: {:?}", adapter.features());
        info!(
            "Adapter Downlevel Capabilities: {:#?}",
            adapter.get_downlevel_capabilities()
        );
        info!("Adapter Limits: {:#?}", adapter.limits());

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    features: wgpu::Features::empty(),
                    limits: if cfg!(target_arch = "wasm32") {
                        wgpu::Limits::downlevel_webgl2_defaults()
                    } else {
                        wgpu::Limits::default()
                    },
                    // limits: wgpu::Limits::default(),
                    label: None,
                },
                None,
            )
            .await
            .expect("Unable to find a suitable GPU adapter!");

        info!("Got Device");

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.describe().srgb)
            .unwrap_or(surface_caps.formats[0]);

        let size = window.inner_size();

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &config);

        let shader = device.create_shader_module(wgpu::include_wgsl!("../shaders/shader.wgsl"));

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
            label: Some("Bind Group Layout"),
        });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[Vertex::desc()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(VERTICES),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(INDICES),
            usage: wgpu::BufferUsages::INDEX,
        });

        let view = View {
            position: [-1.0, -0.0, 0.0],
            zoom: 0.01,
        };

        let view_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("View Buffer"),
            contents: bytemuck::cast_slice(&[view]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind Group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: view_buffer.as_entire_binding(),
            }],
        });

        let num_indices = INDICES.len() as u32;

        Self {
            window,
            surface,
            device,
            queue,
            config,
            size,
            view,
            render_pipeline,
            vertex_buffer,
            index_buffer,
            view_buffer,
            bind_group,
            num_indices,
            clear_col: RED.into(),
        }
    }

    pub fn window(&self) -> &Window {
        &self.window
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
        } else {
            error!(
                "Failed to resize window: Invalid window size ({}x{})!",
                new_size.width, new_size.height
            );
        }
    }

    fn input(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::CursorMoved { position, .. } => {
                let hue = (position.x / (self.size.width as f64)
                    + position.y / (self.size.height as f64))
                    * 0.5;

                let mut hsv = HSV::from(self.clear_col);
                hsv.h = hue as f32;

                self.clear_col = hsv.into();

                true
            }
            _ => false,
        }
    }

    fn update(&mut self) {
        self.view.zoom *= 0.99;

        self.queue
            .write_buffer(&self.view_buffer, 0, bytemuck::cast_slice(&[self.view]));
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(self.clear_col.to_wgpu()),
                        store: true,
                    },
                })],
                depth_stencil_attachment: None,
            });

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);

            render_pass.draw_indexed(0..self.num_indices, 0, 0..1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen(start))]
pub async fn run() {
    cfg_if::cfg_if! {
        if #[cfg(target_arch = "wasm32")] {
            console_error_panic_hook::set_once();
            tracing_wasm::set_as_global_default();
        } else {
            tracing_subscriber::fmt::init();
        }
    }

    info!("Setup Logger");

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new()
        .with_title("Learn wgpu!")
        .build(&event_loop)
        .unwrap();

    #[cfg(target_arch = "wasm32")]
    {
        // Winit prevents sizing with CSS, so we have to set
        // the size manually when on web.
        use winit::dpi::PhysicalSize;
        use winit::platform::web::WindowExtWebSys;

        window.set_inner_size(PhysicalSize::new(450, 400));
        web_sys::window()
            .and_then(|win| win.document())
            .and_then(|doc| {
                let dst = doc.get_element_by_id("wasm-body")?;
                let canvas = web_sys::Element::from(window.canvas());
                dst.append_child(&canvas).ok()?;
                Some(())
            })
            .expect("Couldn't append canvas to document body.");
    }

    info!("Created Window");

    let mut state = State::new(window).await;
    info!("Created State");

    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent {
            ref event,
            window_id,
        } if window_id == state.window.id() => {
            if !state.input(event) {
                match event {
                    WindowEvent::CloseRequested
                    | WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                state: ElementState::Pressed,
                                virtual_keycode: Some(VirtualKeyCode::Escape),
                                ..
                            },
                        ..
                    } => {
                        info!("Exiting...");
                        *control_flow = ControlFlow::Exit;
                    }
                    WindowEvent::Resized(physical_size) => state.resize(*physical_size),
                    WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                        state.resize(**new_inner_size)
                    }
                    _ => {}
                }
            }
        }
        Event::RedrawRequested(window_id) if window_id == state.window().id() => {
            state.update();
            match state.render() {
                Ok(_) => {}
                // Reconfigure the surface if lost
                Err(wgpu::SurfaceError::Lost) => {
                    warn!("Lost Surface, resizing... (Render will occur next frame)");
                    state.resize(state.size);
                }
                // The system is out of memory, we should probably quit
                Err(wgpu::SurfaceError::OutOfMemory) => {
                    error!("Out of Memory!");
                    *control_flow = ControlFlow::Exit;
                }
                // All other errors (Outdated, Timeout) should be resolved by the next frame
                Err(e) => error!("{:?}", e),
            }
        }
        Event::MainEventsCleared => {
            // RedrawRequested will only trigger once, unless we manually
            // request it.
            state.window().request_redraw();
        }
        _ => {}
    });
}
