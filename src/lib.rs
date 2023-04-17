use rand::Rng;
use wgpu::util::DeviceExt;
use winit::{
    dpi::{PhysicalPosition, PhysicalSize},
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

use bracket_color::prelude::*;
use tracing::{error, info, warn};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

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

const PIXEL_ZOOM_SENSITVITY: f64 = 0.01;
const LINE_ZOOM_SENSITVITY: f64 = 0.05;
const MAX_STEP_COUNT: u32 = 500;

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

#[derive(Debug)]
struct Camera {
    position: [f64; 2],
    zoom: f64,
    detail: u32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            position: [0.0, 0.0],
            zoom: 0.005,
            detail: MAX_STEP_COUNT,
        }
    }
}

impl Camera {
    fn zoom(&mut self, amount: f64, mouse_pos: PhysicalPosition<f64>) {
        let old_zoom = self.zoom;
        let world_mouse_pos = self.clip_to_world_pos(mouse_pos.x, mouse_pos.y);

        // self.detail = {
        //     let new_detial = DETAIL_SCALE_FACTOR * amount * 1920.0;
        //     if new_detial < 0.0 {
        //         error!("Failed to zoom out: Fractle Steps (Camera Detail) is negative!");
        //         return;
        //     }

        //     info!("Detial: {}, Amount: {}", self.detail, amount);

        //     new_detial.floor() as u32
        // };

        self.zoom *= 1.0 - amount;

        self.position[0] -= (world_mouse_pos.0 / self.zoom) - (world_mouse_pos.0 / old_zoom);
        self.position[1] -= (world_mouse_pos.1 / self.zoom) - (world_mouse_pos.1 / old_zoom);
    }

    fn clip_to_world_pos(&self, clip_x: f64, clip_y: f64) -> (f64, f64) {
        (
            (clip_x - self.position[0]) * self.zoom,
            (clip_y - self.position[1]) * self.zoom,
        )
    }
}

struct State {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: PhysicalSize<u32>,
    window: Window,
    camera: Camera,
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    num_indices: u32,
    mouse_down: bool,
    mouse_pos: PhysicalPosition<f64>,
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

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.describe().srgb)
            .unwrap_or(surface_caps.formats[0]);

        let size = {
            cfg_if::cfg_if! {
                if #[cfg(target_arch = "wasm32")] {
                    get_js_window_size()
                } else {
                    window.inner_size()
                }
            }
        };

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

        let grad = colorgrad::turbo();
        let grad = colorgrad::CustomGradient::new()
            .colors(&grad.colors(MAX_STEP_COUNT as usize))
            .domain(&[0.0, MAX_STEP_COUNT as f64])
            .build()
            .unwrap();

        let main_width = 800;
        let main_height = 600;

        let camera = Camera {
            // position: [main_width as f64 * 0.5, main_height as f64 * 0.5],
            // ..Default::default()
            position: [19693.422, 28863.879],
            zoom: 0.00002087276,
            ..Default::default()
        };

        let max_val_squared = 4.0;
        let sub_samples = 16;
        let mut rng = rand::thread_rng();

        let mut pixels: Vec<u8> = Vec::with_capacity((main_width * main_height * 4) as usize);

        for y in 0..main_height {
            for x in 0..main_width {
                let x = x as f64 - camera.position[0];
                let y = y as f64 - camera.position[1];

                let mut avg_r = 0.0;
                let mut avg_g = 0.0;
                let mut avg_b = 0.0;

                for _i in 0..sub_samples {
                    let cx = (x + rng.gen_range(0.0..1.0)) * camera.zoom;
                    let cy = (y + rng.gen_range(0.0..1.0)) * camera.zoom;

                    let mut zx = 0.0;
                    let mut zy = 0.0;

                    for i in 0..=camera.detail {
                        let z_squared_x = zx * zx - zy * zy;
                        let z_squared_y = 2.0 * zx * zy;

                        zx = cx + z_squared_x;
                        zy = cy + z_squared_y;

                        let mag_squared = zx * zx + zy * zy;

                        if mag_squared > max_val_squared {
                            let mag = mag_squared.sqrt();

                            let col = grad.at(i as f64 - mag.log2().max(1.0).log2());

                            avg_r += col.r;
                            avg_g += col.g;
                            avg_b += col.b;

                            break;
                        }
                    }
                }

                pixels.push((avg_r / sub_samples as f64 * 255.0) as u8);
                pixels.push((avg_g / sub_samples as f64 * 255.0) as u8);
                pixels.push((avg_b / sub_samples as f64 * 255.0) as u8);
                pixels.push(255);
            }
        }

        let main_size = wgpu::Extent3d {
            width: main_width,
            height: main_height,
            depth_or_array_layers: 1,
        };

        let main_texture = device.create_texture(&wgpu::TextureDescriptor {
            size: main_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            label: Some("main texture"),
            view_formats: &[],
        });

        let main_view = main_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let main_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &main_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &pixels,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: std::num::NonZeroU32::new(4 * main_width),
                rows_per_image: std::num::NonZeroU32::new(main_height),
            },
            main_size,
        );

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
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

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&main_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&main_sampler),
                },
            ],
        });

        let num_indices = INDICES.len() as u32;

        #[allow(unused_mut)] // We add this to prevent the warning when not targeting wasm
        let mut state = Self {
            window,
            surface,
            device,
            queue,
            config,
            size,
            camera,
            render_pipeline,
            vertex_buffer,
            index_buffer,
            bind_group,
            num_indices,
            mouse_down: false,
            mouse_pos: PhysicalPosition { x: 0.0, y: 0.0 },
        };

        #[cfg(target_arch = "wasm32")]
        state.force_canvas_resize(None);

        state
    }

    #[cfg(target_arch = "wasm32")]
    fn check_resize_canvas(&mut self) {
        let new_size = get_js_window_size();

        if self.size == new_size {
            return;
        }

        self.force_canvas_resize(Some(new_size));
    }

    /// Forces set the canvas size. if `new_size` is `None`, the canvas is resized to the window size
    #[cfg(target_arch = "wasm32")]
    fn force_canvas_resize(&mut self, new_size: Option<PhysicalSize<u32>>) {
        self.size = new_size.unwrap_or(get_js_window_size());
        self.window.set_inner_size(self.size);
    }

    fn update(&mut self) {
        #[cfg(target_arch = "wasm32")]
        self.check_resize_canvas();
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
            WindowEvent::MouseWheel {
                delta: MouseScrollDelta::LineDelta(_, y),
                ..
            } => {
                self.camera
                    .zoom(*y as f64 * LINE_ZOOM_SENSITVITY, self.mouse_pos);
                true
            }
            WindowEvent::MouseWheel {
                delta: MouseScrollDelta::PixelDelta(delta),
                ..
            } => {
                self.camera
                    .zoom(delta.y * PIXEL_ZOOM_SENSITVITY, self.mouse_pos);
                true
            }
            WindowEvent::CursorMoved { position, .. } => {
                if !self.mouse_down {
                    return false;
                }

                let delta = (position.x - self.mouse_pos.x, position.y - self.mouse_pos.y);

                self.camera.position[0] += delta.0;
                self.camera.position[1] += delta.1;

                false
            }
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } => {
                self.mouse_down = match state {
                    ElementState::Pressed => true,
                    ElementState::Released => false,
                };

                true
            }
            _ => false,
        }
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
                        load: wgpu::LoadOp::Clear(RGB::from(WHITE).to_wgpu()),
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

#[cfg(target_arch = "wasm32")]
pub fn get_js_window_size() -> PhysicalSize<u32> {
    let win_js = web_sys::window().expect("Couldn't get js window");

    let width = win_js.inner_width().unwrap().as_f64().unwrap() as u32;
    let height = win_js.inner_height().unwrap().as_f64().unwrap() as u32;

    PhysicalSize::new(width, height)
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
        .with_title("Mandelbrot WGPU")
        .build(&event_loop)
        .unwrap();

    #[cfg(target_arch = "wasm32")]
    {
        // Winit prevents sizing with CSS, so we have to set
        // the size manually when on web.
        use winit::platform::web::WindowExtWebSys;

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

    info!("Starting Event Loop");
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
                    WindowEvent::CursorMoved { position, .. } => {
                        state.mouse_pos = *position;
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
