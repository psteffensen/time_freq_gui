#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release

use eframe::egui;
use egui_plot::{Legend, Line, Plot, PlotPoints};
use realfft::num_complex;
use realfft::RealFftPlanner;
use std::collections::VecDeque;

fn generate_test_signal(start_idx: usize, end_idx: usize) -> Vec<[f64; 2]> {
    let sample_rate = 44100.0; // Standard CD quality sample rate
    let duration = 2.0; // 1 second
    let num_samples = (sample_rate * duration) as usize;

    let center_frequency = 512.0;
    let start_lfo = 20.0;
    let end_lfo = 5.0;
    let start_deviation = 10.0;
    let end_deviation = 30.0;
    let attack_duration = 0.05; // 50ms attack

    let mut signal = Vec::with_capacity(num_samples);

    for n in start_idx..end_idx {
        let time = n as f64 / sample_rate;

        // Linearly interpolate LFO and deviation
        let lfo = start_lfo + (end_lfo - start_lfo) * time / duration;
        let deviation = start_deviation + (end_deviation - start_deviation) * time / duration;

        // Calculate current frequency
        let current_frequency =
            center_frequency + deviation * (lfo * time * 2.0 * std::f64::consts::PI).sin();

        // Calculate amplitude (including attack phase)
        let amplitude = if time < attack_duration {
            time / attack_duration
        } else {
            1.0 - (time - attack_duration) / (duration - attack_duration)
        };

        // Generate sine wave
        let value = amplitude * (current_frequency * time * 2.0 * std::f64::consts::PI).sin();
        signal.push([n as f64, value]);
    }
    signal

    // Here you can process the signal vector as needed
    // For example, save it as a WAV file, plot it, or apply a wavelet transform
}

fn fft(signal: Vec<[f64; 2]>) -> Vec<[f64; 2]> {
    let mut signal_1d: Vec<f64> = signal.iter().map(|x| x[1]).collect();
    let mut real_planner = RealFftPlanner::<f64>::new();

    // create a FFT
    let r2c = real_planner.plan_fft_forward(44100);
    let mut spectrum = r2c.make_output_vec();

    // forward transform the signal
    r2c.process(&mut signal_1d, &mut spectrum).unwrap();

    vec_apply_hanning_window(&mut spectrum);

    let spectrum_abs: Vec<f64> = spectrum.iter().map(|x| abs(*x)).collect();
    let result: Vec<[f64; 2]> = spectrum_abs
        .iter()
        .map(|&x| {
            if x > 0.0 {
                20.0 * x.log10()
            } else {
                -std::f64::INFINITY
            }
        }) // Use x.log10() here
        .enumerate()
        .map(|(count, x)| [count as f64, x])
        .collect();

    result
}

fn abs(c: num_complex::Complex<f64>) -> f64 {
    let real = c.re;
    let imag = c.im;

    // return f32::sqrt(real * real + imag * imag);
    return (real * real + imag * imag).powf(0.7);
}

// Define the Hanning window function
fn hanning_window(n: usize) -> f64 {
    let alpha = 0.2;
    let pi = core::f64::consts::PI;
    alpha - (1.0 - alpha) * (2.0 * pi * n as f64 / (44100 - 1) as f64).cos()
}

// Apply the Hanning window to the input signal
fn apply_hanning_window(signal: &mut [f64; 44100]) {
    for n in 0..44100 {
        signal[n] *= hanning_window(n);
    }
}

// Apply the Hanning window to the input signal
fn vec_apply_hanning_window(signal: &mut Vec<num_complex::Complex<f64>>) {
    for (n, value) in signal.iter_mut().enumerate() {
        let window_value = hanning_window(n);
        *value = num_complex::Complex::new(value.re * window_value, value.im * window_value);
    }
}

fn main() -> Result<(), eframe::Error> {
    env_logger::init(); // Log to stderr (if you run with `RUST_LOG=debug`).
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1200.0, 1000.0]),
        ..Default::default()
    };
    eframe::run_native(
        "My test app",
        options,
        Box::new(|cc| Box::new(MyApp::new(cc))),
        // Box::new(|_cc| {
        //     // This gives us image support:
        //     // egui_extras::install_image_loaders(&cc.egui_ctx);
        //     Box::<MyApp>::default()
        // }),
    )
}
struct MyApp {
    signal_buffer: VecDeque<[f64; 2]>,
    buffer_size: usize,
    sample_rate: f64,
    last_update_time: std::time::Instant, // Used to track time for periodic updates
    current_index: usize,                 // Keeps track of the current index in the signal
}

impl MyApp {
    // Constructor for MyApp
    fn new(cc: &eframe::CreationContext) -> Self {
        cc.egui_ctx.set_visuals(egui::Visuals::dark());

        let mut signal_buffer: VecDeque<[f64; 2]> = VecDeque::with_capacity(44100);

        for _ in 0..44100 {
            signal_buffer.push_back([0.0, 0.0]);
        }

        MyApp {
            signal_buffer: signal_buffer,
            buffer_size: 44100, // 0.1 second of data at 44100 Hz
            sample_rate: 44100.0,
            last_update_time: std::time::Instant::now(),
            current_index: 0,
        }
    }

    // fn update_buffer(&mut self) {
    //     let now = std::time::Instant::now();
    //     if now.duration_since(self.last_update_time).as_secs_f64() >= 0.1 {
    //         // Update buffer every 0.1 seconds
    //         self.last_update_time = now;
    //         let new_data = generate_test_signal(self.current_index, self.current_index + 4410);
    //         self.current_index += 4410;
    //         self.signal_buffer.extend(new_data);
    //         if self.signal_buffer.len() > self.buffer_size {
    //             self.signal_buffer
    //                 .drain(0..self.signal_buffer.len() - self.buffer_size);
    //         }
    //         // Reset index if the end of the signal is reached
    //         if self.current_index >= self.sample_rate as usize {
    //             self.current_index = 0;
    //         }
    //     }
    // }

    fn update_buffer(&mut self, new_data: Vec<[f64; 2]>) {
        for point in new_data {
            self.signal_buffer.push_back(point);
            if self.signal_buffer.len() > self.buffer_size {
                self.signal_buffer.pop_front(); // Remove the oldest data point
            }
        }
    }

    // fn update_buffer(&mut self) {
    //     self.signal_buffer = vec![[0.0, 0.0]; self.buffer_size * 2];
    // }
}

// impl Default for MyApp {
//     fn default() -> Self {
//         Self {
//             signal_buffer: Vec::new(),
//             buffer_size: 4410,
//         }
//     }
// }

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // ctx.set_visuals(egui::Visuals::dark());

        // self.update_buffer();
        let now = std::time::Instant::now();
        if now.duration_since(self.last_update_time).as_secs_f64() >= 0.01 {
            self.last_update_time = now;
            let new_data = generate_test_signal(self.current_index, self.current_index + 441);
            self.current_index += 441;
            if self.current_index >= self.sample_rate as usize * 2 {
                self.current_index = 0;
            }
            self.update_buffer(new_data);
        }

        // if self.signal_buffer.len() >= self.buffer_size {
        if self.signal_buffer.len() == self.buffer_size {
            ctx.request_repaint();
            // let signal_slice = self.signal_buffer.clone();
            // let signal_fft: Vec<[f64; 2]> = fft(signal_slice.clone());
            // let signal_fft: Vec<[f64; 2]> = fft(self.signal_buffer.clone());
            let signal_fft: Vec<[f64; 2]> = fft(Vec::from(self.signal_buffer.clone()));
            // let signal_slice = &self.signal_buffer[self.signal_buffer.len() - self.buffer_size..];
            // let signal_fft: Vec<[f64; 2]> = fft(signal_slice.to_vec().clone());

            let mut plot_rect_sp = None;
            let mut plot_rect_wp = None;

            egui::CentralPanel::default().show(ctx, |ui| {
                egui::widgets::global_dark_light_mode_buttons(ui);

                ui.heading("Time");
                ui.horizontal(|ui| {
                    let height = match ctx.input(|i| i.viewport().outer_rect) {
                        Some(rect) => rect.height() / 2.5,
                        None => 100.0, // Default height is now 100.0
                    };

                    let signal_plot = Plot::new("Signal")
                        .auto_bounds([true, false].into())
                        .legend(Legend::default())
                        .height(height);
                    let sp = signal_plot.show(ui, |plot_ui| {
                        let signal_slice = &self.signal_buffer;

                        // Prepare a vector for the last 100 plot points
                        let mut plot_points: Vec<[f64; 2]> = Vec::new();

                        if signal_slice.len() >= 2 * 441 {
                            // Get the last 100 points from the signal_slice
                            for i in (signal_slice.len() - 2 * 441)..signal_slice.len() {
                                if let Some(value) = signal_slice.get(i) {
                                    // Assuming the x-coordinate is the index and y-coordinate is the second element of the array
                                    plot_points.push([i as f64, value[1]]);
                                }
                            }
                        } else {
                            // If there are less than 100 elements, plot all of them
                            for (i, &value) in signal_slice.iter().enumerate() {
                                plot_points.push([i as f64, value[1]]);
                            }
                        }
                        plot_ui.line(Line::new(PlotPoints::from(plot_points)).name("curve"));
                        // plot_ui.line(
                        // Line::new(PlotPoints::from(Vec::from(signal_slice.clone())))
                        // .name("curve"),
                        // );
                        // plot_ui
                        // .line(Line::new(PlotPoints::from(signal_slice.to_vec())).name("curve"));
                    });
                    // Remember the position of the plot
                    plot_rect_sp = Some(sp.response.rect);
                });
                ui.heading("Frequency");
                ui.horizontal(|ui| {
                    let height = match ctx.input(|i| i.viewport().outer_rect) {
                        Some(rect) => rect.height() / 2.5,
                        None => 100.0, // Default height is now 100.0
                    };
                    let wavelet_plot = Plot::new("Wavelet")
                        .include_x(2000.0)
                        .include_y(1000)
                        // .auto_bounds([false, false].into())
                        // .with_x_bounds(0.0..2000.0) // Set X axis bounds
                        // .with_y_bounds(0.0..12000.0) // Set Y axis bounds
                        .legend(Legend::default())
                        .height(height);
                    let wp = wavelet_plot.show(ui, |plot_ui| {
                        plot_ui.line(Line::new(PlotPoints::from(signal_fft)).name("curve"));
                    });
                    // Remember the position of the plot
                    plot_rect_wp = Some(wp.response.rect);
                });
            });
        } else {
            // println!(
            //     "Buffer does not have enough data. Buffer size: {}, Required size: {}",
            //     self.signal_buffer.len(),
            //     self.buffer_size
            // );
            println!("yo");
        }
    }
}
