#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release

use eframe::egui;
use egui_plot::{Legend, Line, Plot, PlotPoints};
use realfft::num_complex;
use realfft::RealFftPlanner;

fn generate_test_signal() -> Vec<[f64; 2]> {
    let sample_rate = 44100.0; // Standard CD quality sample rate
    let duration = 1.0; // 1 second
    let num_samples = (sample_rate * duration) as usize;

    let center_frequency = 512.0;
    let start_lfo = 20.0;
    let end_lfo = 5.0;
    let start_deviation = 10.0;
    let end_deviation = 30.0;
    let attack_duration = 0.05; // 50ms attack

    let mut signal = Vec::with_capacity(num_samples);

    for n in 0..num_samples {
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
        .enumerate()
        .map(|(count, x)| [count as f64, *x])
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
struct MyApp {}

impl MyApp {
    // Constructor for MyApp
    fn new(cc: &eframe::CreationContext) -> Self {
        cc.egui_ctx.set_visuals(egui::Visuals::dark());
        MyApp {
            // Initialize your fields here
        }
    }
}

impl Default for MyApp {
    fn default() -> Self {
        Self {
            // name: "Arthur".to_owned(),
            // age: 42,
        }
    }
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // ctx.set_visuals(egui::Visuals::dark());
        let mut plot_rect_sp = None;
        let mut plot_rect_wp = None;
        egui::CentralPanel::default().show(ctx, |ui| {
            egui::widgets::global_dark_light_mode_buttons(ui);
            let signal: Vec<[f64; 2]> = generate_test_signal();
            let signal_fft: Vec<[f64; 2]> = fft(signal.clone());
            ui.heading("Time");
            ui.horizontal(|ui| {
                let height = match ctx.input(|i| i.viewport().outer_rect) {
                    Some(rect) => rect.height() / 2.5,
                    None => 100.0, // Default height is now 100.0
                };

                let signal_plot = Plot::new("Signal").legend(Legend::default()).height(height);
                let sp = signal_plot.show(ui, |plot_ui| {
                    plot_ui.line(Line::new(PlotPoints::from(signal)).name("curve"));
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
                    .legend(Legend::default())
                    .height(height);
                let wp = wavelet_plot.show(ui, |plot_ui| {
                    plot_ui.line(Line::new(PlotPoints::from(signal_fft)).name("curve"));
                });
                // Remember the position of the plot
                plot_rect_wp = Some(wp.response.rect);
            });
        });
    }
}
