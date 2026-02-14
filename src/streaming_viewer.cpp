#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <libbladeRF.h>
#include <fftw3.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
#include <chrono>
#include <thread>

#define RX_GAIN 10
#define CHANNEL BLADERF_CHANNEL_RX(0)
#define FFT_SIZE 8192
#define TIME_AVERAGE 10
#define MAX_FFTS_MEMORY 10000

struct FFTHeader {
    char magic[4];
    uint32_t version;
    uint32_t fft_size;
    uint32_t sample_rate;
    uint64_t center_frequency;
    uint32_t num_ffts;
    uint32_t time_average;
    float power_min;
    float power_max;
    float reserved[8];
};

class FFTViewer {
public:
    FFTHeader header;
    std::vector<int8_t> fft_data;
    
    int current_fft_idx = 0;
    int fft_index_step = 1;
    float freq_zoom = 1.0f;
    float freq_pan = 0.0f;
    float display_power_min = 0.0f;
    float display_power_max = 0.0f;
    float spectrum_height_ratio = 0.4f;
    
    bool is_playing = false;
    bool is_looping = false;
    std::chrono::steady_clock::time_point play_start_time;
    double total_duration = 0.0;
    
    std::vector<std::vector<float>> waterfall_line_cache;
    float last_cached_freq_pan = -999.0f;
    float last_cached_freq_zoom = -999.0f;
    int last_cached_num_pixels = -1;
    
    std::vector<float> spectrum_cache;
    int cached_spectrum_idx = -1;
    float cached_spectrum_freq_pan = -999.0f;
    float cached_spectrum_freq_zoom = -999.0f;
    int cached_spectrum_pixels = -1;
    
    float last_cached_power_min = -999.0f;
    float last_cached_power_max = -999.0f;
    
    std::chrono::steady_clock::time_point last_input_time;
    bool high_fps_mode = true;
    
    struct bladerf *dev = nullptr;
    fftwf_plan fft_plan = nullptr;
    fftwf_complex *fft_in = nullptr;
    fftwf_complex *fft_out = nullptr;
    bool is_running = true;
    int total_ffts_captured = 0;
    
    std::string window_title;

    bool initialize_bladerf(float center_freq_mhz, float sample_rate_msps) {
        int status = bladerf_open(&dev, nullptr);
        if (status != 0) {
            fprintf(stderr, "Failed to open device: %s\n", bladerf_strerror(status));
            return false;
        }

        status = bladerf_set_frequency(dev, CHANNEL, static_cast<uint64_t>(center_freq_mhz * 1e6));
        if (status != 0) {
            fprintf(stderr, "Failed to set frequency: %s\n", bladerf_strerror(status));
            return false;
        }

        status = bladerf_set_sample_rate(dev, CHANNEL, static_cast<uint32_t>(sample_rate_msps * 1e6), nullptr);
        if (status != 0) {
            fprintf(stderr, "Failed to set sample rate: %s\n", bladerf_strerror(status));
            return false;
        }

        status = bladerf_set_gain(dev, CHANNEL, RX_GAIN);
        if (status != 0) {
            fprintf(stderr, "Failed to set gain: %s\n", bladerf_strerror(status));
            return false;
        }

        status = bladerf_enable_module(dev, CHANNEL, true);
        if (status != 0) {
            fprintf(stderr, "Failed to enable RX: %s\n", bladerf_strerror(status));
            return false;
        }

        status = bladerf_sync_config(dev, BLADERF_RX_X1, BLADERF_FORMAT_SC16_Q11, 
                                     512, 16384, 32, 5000);
        if (status != 0) {
            fprintf(stderr, "Failed to configure sync: %s\n", bladerf_strerror(status));
            return false;
        }

        printf("BladeRF initialized: %.2f MHz, %.2f MSPS\n", center_freq_mhz, sample_rate_msps);
        
        std::memcpy(header.magic, "FFTD", 4);
        header.version = 1;
        header.fft_size = FFT_SIZE;
        header.sample_rate = static_cast<uint32_t>(sample_rate_msps * 1e6);
        header.center_frequency = static_cast<uint64_t>(center_freq_mhz * 1e6);
        header.time_average = TIME_AVERAGE;
        header.power_min = -80.0f;
        header.power_max = 30.0f;
        header.num_ffts = 0;
        
        fft_data.resize(MAX_FFTS_MEMORY * FFT_SIZE);
        waterfall_line_cache.resize(MAX_FFTS_MEMORY);
        
        char title[256];
        snprintf(title, sizeof(title), "Real-time FFT Viewer - %.2f MHz", center_freq_mhz);
        window_title = title;
        
        display_power_min = header.power_min;
        display_power_max = header.power_max;
        
        fft_in = fftwf_alloc_complex(FFT_SIZE);
        fft_out = fftwf_alloc_complex(FFT_SIZE);
        fft_plan = fftwf_plan_dft_1d(FFT_SIZE, fft_in, fft_out, FFTW_FORWARD, FFTW_MEASURE);
        
        last_input_time = std::chrono::steady_clock::now();
        
        return true;
    }

    void capture_and_process() {
        int16_t *iq_buffer = new int16_t[FFT_SIZE * 2];
        std::vector<float> power_accum(FFT_SIZE, 0.0f);
        int fft_count = 0;

        while (is_running) {
            int status = bladerf_sync_rx(dev, iq_buffer, FFT_SIZE, nullptr, 5000);
            if (status != 0) {
                fprintf(stderr, "RX error: %s\n", bladerf_strerror(status));
                continue;
            }

            for (int i = 0; i < FFT_SIZE; i++) {
                fft_in[i][0] = iq_buffer[i * 2] / 2048.0f;
                fft_in[i][1] = iq_buffer[i * 2 + 1] / 2048.0f;
            }

            fftwf_execute(fft_plan);

            for (int i = 0; i < FFT_SIZE; i++) {
                float mag_sq = fft_out[i][0] * fft_out[i][0] + fft_out[i][1] * fft_out[i][1];
                float power_db = 10.0f * log10(mag_sq + 1e-10f);
                power_accum[i] += power_db;
            }

            fft_count++;

            if (fft_count >= TIME_AVERAGE) {
                int fft_idx = total_ffts_captured % MAX_FFTS_MEMORY;
                int8_t *fft_row = fft_data.data() + fft_idx * FFT_SIZE;

                for (int i = 0; i < FFT_SIZE; i++) {
                    float avg_power = power_accum[i] / fft_count;
                    float normalized = (avg_power - header.power_min) / (header.power_max - header.power_min);
                    normalized = std::max(-1.0f, std::min(1.0f, normalized));
                    fft_row[i] = static_cast<int8_t>(normalized * 127);
                }

                current_fft_idx = fft_idx;
                total_ffts_captured++;
                header.num_ffts = std::min(total_ffts_captured, MAX_FFTS_MEMORY);
                last_cached_freq_pan = -999.0f;
                last_cached_num_pixels = -1;

                std::fill(power_accum.begin(), power_accum.end(), 0.0f);
                fft_count = 0;
            }
        }

        delete[] iq_buffer;
    }

    float get_freq_from_bin(int bin, float sr_mhz) {
        int n = header.fft_size;
        if (bin == 0) return 0.0f;
        if (bin <= n / 2) {
            return bin * sr_mhz / n;
        } else {
            return (bin - n) * sr_mhz / n;
        }
    }

    void check_adaptive_fps(GLFWwindow* window) {
        auto now = std::chrono::steady_clock::now();
        double time_since_input = std::chrono::duration<double>(now - last_input_time).count();
        
        if (time_since_input > 0.5 && high_fps_mode) {
            high_fps_mode = false;
            glfwSwapInterval(5);
        } else if (time_since_input <= 0.1 && !high_fps_mode) {
            high_fps_mode = true;
            glfwSwapInterval(1);
        }
    }

    void register_input() {
        last_input_time = std::chrono::steady_clock::now();
    }

    void update_playback() {
        if (header.num_ffts > 0) {
            current_fft_idx = header.num_ffts - 1;
        }
    }

    void compute_fft_line(int fft_idx, int num_pixels, int bin_skip, float sr_mhz,
                         float disp_start, float disp_end, std::vector<float>& pixel_powers) {
        const int8_t *spec = fft_data.data() + fft_idx * header.fft_size;
        
        for (int bin = 0; bin < static_cast<int>(header.fft_size); bin += bin_skip) {
            float freq = get_freq_from_bin(bin, sr_mhz);
            
            if (freq < disp_start || freq > disp_end) continue;
            
            float norm_freq = (freq - disp_start) / (disp_end - disp_start);
            int px = static_cast<int>(norm_freq * num_pixels);
            
            if (px >= 0 && px < num_pixels) {
                float power = header.power_min + (spec[bin] / 127.0f) * (header.power_max - header.power_min);
                pixel_powers[px] = std::max(pixel_powers[px], power);
            }
        }
    }

    void draw_spectrum(float canvas_width, float canvas_height);
    void draw_power_grid(ImDrawList *draw_list, ImVec2 pos, ImVec2 size, float range);
    void draw_freq_grid(ImDrawList *draw_list, ImVec2 pos, ImVec2 size, float start, float end);
    void draw_waterfall_canvas(ImDrawList *draw_list, ImVec2 plot_pos, ImVec2 plot_size);

    ImU32 get_color(float v) {
        v = std::max(0.0f, std::min(1.0f, v));
        float r, g, b;
        if (v < 0.25f) { r=0; g=0; b=v/0.25f; }
        else if (v < 0.5f) { r=0; g=(v-0.25f)/0.25f; b=1.0f; }
        else if (v < 0.75f) { r=(v-0.5f)/0.25f; g=1.0f; b=1.0f-(v-0.5f)/0.25f; }
        else { r=1.0f; g=1.0f-(v-0.75f)/0.25f; b=0; }
        return IM_COL32(r*255, g*255, b*255, 255);
    }

    ~FFTViewer() {
        is_running = false;
        
        if (fft_plan) fftwf_destroy_plan(fft_plan);
        if (fft_in) fftwf_free(fft_in);
        if (fft_out) fftwf_free(fft_out);
        
        if (dev) {
            bladerf_enable_module(dev, CHANNEL, false);
            bladerf_close(dev);
        }
    }
};

void FFTViewer::draw_spectrum(float canvas_width, float canvas_height) {
    ImDrawList *draw_list = ImGui::GetWindowDrawList();
    ImVec2 canvas_pos = ImGui::GetCursorScreenPos();
    ImVec2 canvas_size(canvas_width, canvas_height);

    ImVec2 plot_pos(canvas_pos.x + 40, canvas_pos.y);
    ImVec2 plot_size(canvas_size.x - 40, canvas_size.y - 25);

    draw_list->AddRectFilled(plot_pos, ImVec2(plot_pos.x + plot_size.x, plot_pos.y + plot_size.y), 
                             IM_COL32(20, 20, 20, 255));

    if (header.num_ffts == 0) return;

    const int8_t *spectrum = fft_data.data() + current_fft_idx * header.fft_size;

    float nyquist = header.sample_rate / 2.0f / 1e6f;
    float total_range = 2.0f * nyquist;
    
    float disp_start = -nyquist + freq_pan * total_range;
    float disp_width = total_range / freq_zoom;
    float disp_end = disp_start + disp_width;
    
    disp_start = std::max(-nyquist, disp_start);
    disp_end = std::min(nyquist, disp_end);

    float power_range = display_power_max - display_power_min;
    
    draw_power_grid(draw_list, plot_pos, plot_size, power_range);
    draw_freq_grid(draw_list, plot_pos, plot_size, disp_start, disp_end);

    float sr_mhz = header.sample_rate / 1e6f;
    int num_pixels = static_cast<int>(plot_size.x);
    
    bool use_spectrum_cache = (cached_spectrum_idx == current_fft_idx &&
                              cached_spectrum_freq_pan == freq_pan &&
                              cached_spectrum_freq_zoom == freq_zoom &&
                              cached_spectrum_pixels == num_pixels);
    
    if (!use_spectrum_cache) {
        spectrum_cache.assign(num_pixels, -1e10f);
        for (int bin = 0; bin < static_cast<int>(header.fft_size); bin++) {
            float freq = get_freq_from_bin(bin, sr_mhz);
            
            if (freq < disp_start || freq > disp_end) continue;
            
            float norm_freq = (freq - disp_start) / (disp_end - disp_start);
            int px = static_cast<int>(norm_freq * num_pixels);
            
            if (px >= 0 && px < num_pixels) {
                float power = header.power_min + (spectrum[bin] / 127.0f) * (header.power_max - header.power_min);
                spectrum_cache[px] = std::max(spectrum_cache[px], power);
            }
        }
        cached_spectrum_idx = current_fft_idx;
        cached_spectrum_freq_pan = freq_pan;
        cached_spectrum_freq_zoom = freq_zoom;
        cached_spectrum_pixels = num_pixels;
    }

    for (int px = 0; px < num_pixels - 1; px++) {
        float np1 = (spectrum_cache[px] - display_power_min) / power_range;
        float np2 = (spectrum_cache[px + 1] - display_power_min) / power_range;
        np1 = std::max(0.0f, std::min(1.0f, np1));
        np2 = std::max(0.0f, std::min(1.0f, np2));

        float x1 = plot_pos.x + px;
        float x2 = plot_pos.x + px + 1;
        float y1 = plot_pos.y + plot_size.y * (1.0f - np1);
        float y2 = plot_pos.y + plot_size.y * (1.0f - np2);

        draw_list->AddLine(ImVec2(x1, y1), ImVec2(x2, y2), get_color(np1), 2.0f);
    }

    ImGui::InvisibleButton("spectrum_canvas", canvas_size);
    if (ImGui::IsItemHovered()) {
        ImGuiIO& io = ImGui::GetIO();
        ImVec2 mouse = ImGui::GetMousePos();
        float mx = (mouse.x - plot_pos.x) / plot_size.x;
        mx = std::max(0.0f, std::min(1.0f, mx));
        float freq_mouse = disp_start + mx * (disp_end - disp_start);
        
        if (io.MouseWheel != 0.0f) {
            register_input();
            freq_zoom *= (1.0f + io.MouseWheel * 0.1f);
            freq_zoom = std::max(1.0f, std::min(10.0f, freq_zoom));
            
            float new_width = total_range / freq_zoom;
            float new_start = freq_mouse - (mx * new_width);
            freq_pan = (new_start + nyquist) / total_range;
            freq_pan = std::max(0.0f, std::min(1.0f - 1.0f / freq_zoom, freq_pan));
        }
    }
}

void FFTViewer::draw_power_grid(ImDrawList *draw_list, ImVec2 pos, ImVec2 size, float range) {
    ImU32 col = IM_COL32(150, 150, 150, 150);
    for (int i = 0; i <= 12; i++) {
        float ny = i * 10.0f / range;
        if (ny > 1.0f) break;
        float y = pos.y + size.y * (1.0f - ny);
        draw_list->AddLine(ImVec2(pos.x, y), ImVec2(pos.x + size.x, y), col, 1.0f);
        char label[16];
        snprintf(label, sizeof(label), "%.0f", display_power_min + i * 10.0f);
        draw_list->AddText(ImVec2(pos.x - 35, y - 8), IM_COL32(0, 255, 0, 255), label);
    }
}

void FFTViewer::draw_freq_grid(ImDrawList *draw_list, ImVec2 pos, ImVec2 size, float start, float end) {
    ImU32 col = IM_COL32(150, 150, 150, 150);
    float range = end - start;
    
    int tick_start = static_cast<int>(std::floor(start));
    int tick_end = static_cast<int>(std::ceil(end));
    
    for (int tick = tick_start; tick <= tick_end; tick++) {
        float nx = (tick - start) / range;
        if (nx < -0.05f || nx > 1.05f) continue;
        nx = std::max(0.0f, std::min(1.0f, nx));
        float x = pos.x + nx * size.x;
        draw_list->AddLine(ImVec2(x, pos.y), ImVec2(x, pos.y + size.y), col, 1.0f);
        float abs_f = tick + header.center_frequency / 1e6f;
        char label[16];
        snprintf(label, sizeof(label), "%.0f", abs_f);
        draw_list->AddText(ImVec2(x - 15, pos.y + size.y + 5), IM_COL32(0, 255, 0, 255), label);
    }
}

void FFTViewer::draw_waterfall_canvas(ImDrawList *draw_list, ImVec2 plot_pos, ImVec2 plot_size) {
    draw_list->AddRectFilled(plot_pos, ImVec2(plot_pos.x + plot_size.x, plot_pos.y + plot_size.y), 
                             IM_COL32(10, 10, 10, 255));

    float nyquist = header.sample_rate / 2.0f / 1e6f;
    float total_range = 2.0f * nyquist;
    float disp_start = -nyquist + freq_pan * total_range;
    float disp_width = total_range / freq_zoom;
    float disp_end = disp_start + disp_width;
    
    disp_start = std::max(-nyquist, disp_start);
    disp_end = std::min(nyquist, disp_end);

    int display_rows = std::min(static_cast<int>(header.num_ffts), static_cast<int>(plot_size.y / 1));
    float sr_mhz = header.sample_rate / 1e6f;
    int num_pixels = static_cast<int>(plot_size.x);

    float visible_bins = header.fft_size / freq_zoom;
    int bin_skip = std::max(1, static_cast<int>(visible_bins / num_pixels));

    bool use_cache = (last_cached_freq_pan == freq_pan && 
                     last_cached_freq_zoom == freq_zoom &&
                     last_cached_num_pixels == num_pixels);

    if (!use_cache) {
        for (int fft_idx = 0; fft_idx < static_cast<int>(header.num_ffts); fft_idx++) {
            waterfall_line_cache[fft_idx].assign(num_pixels, -1e10f);
            compute_fft_line(fft_idx, num_pixels, bin_skip, sr_mhz, disp_start, disp_end, waterfall_line_cache[fft_idx]);
        }
        last_cached_freq_pan = freq_pan;
        last_cached_freq_zoom = freq_zoom;
        last_cached_num_pixels = num_pixels;
    }

    bool power_changed = (last_cached_power_min != display_power_min || 
                         last_cached_power_max != display_power_max);

    for (int display_row = 0; display_row < display_rows; display_row++) {
        int fft_idx = current_fft_idx - display_rows + 1 + display_row;
        
        if (fft_idx < 0 || fft_idx > current_fft_idx) continue;

        float row_y = plot_pos.y + (display_rows - 1 - display_row) * plot_size.y / display_rows;
        float row_h = plot_size.y / display_rows;

        std::vector<float>& line_data = waterfall_line_cache[fft_idx];

        for (int px = 0; px < num_pixels && px < static_cast<int>(line_data.size()); px++) {
            float power_range = display_power_max - display_power_min;
            float np = (line_data[px] - display_power_min) / power_range;
            np = std::max(0.0f, std::min(1.0f, np));

            float bx = plot_pos.x + px;
            draw_list->AddRectFilled(ImVec2(bx, row_y), ImVec2(bx + 1, row_y + row_h), get_color(np));
        }
    }
    
    if (power_changed) {
        last_cached_power_min = display_power_min;
        last_cached_power_max = display_power_max;
    }
    
    ImGui::InvisibleButton("waterfall_canvas", plot_size);
    if (ImGui::IsItemHovered()) {
        ImGuiIO& io = ImGui::GetIO();
        ImVec2 mouse = ImGui::GetMousePos();
        float mx = (mouse.x - plot_pos.x) / plot_size.x;
        mx = std::max(0.0f, std::min(1.0f, mx));
        
        if (io.MouseWheel != 0.0f) {
            register_input();
            float disp_start_local = -nyquist + freq_pan * total_range;
            float freq_mouse = disp_start_local + mx * (total_range / freq_zoom);
            
            freq_zoom *= (1.0f + io.MouseWheel * 0.1f);
            freq_zoom = std::max(1.0f, std::min(10.0f, freq_zoom));
            
            float new_width = total_range / freq_zoom;
            float new_start = freq_mouse - (mx * new_width);
            freq_pan = (new_start + nyquist) / total_range;
            freq_pan = std::max(0.0f, std::min(1.0f - 1.0f / freq_zoom, freq_pan));
        }
    }
}

void run_streaming_viewer() {
    float center_freq, sample_rate;
    
    printf("Enter center frequency (MHz): ");
    scanf("%f", &center_freq);
    
    printf("Enter sample rate (MSPS): ");
    scanf("%f", &sample_rate);
    
    FFTViewer viewer;
    
    if (!viewer.initialize_bladerf(center_freq, sample_rate)) {
        printf("Failed to initialize BladeRF\n");
        return;
    }

    std::thread capture_thread(&FFTViewer::capture_and_process, &viewer);

    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow *window = glfwCreateWindow(1400, 900, viewer.window_title.c_str(), nullptr, nullptr);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    glewExperimental = GL_TRUE;
    glewInit();

    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    while (!glfwWindowShouldClose(window)) {
        viewer.check_adaptive_fps(window);
        
        glfwPollEvents();
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        viewer.update_playback();

        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize);

        if (ImGui::Begin("Real-time FFT Viewer", nullptr, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoScrollbar)) {
            ImGui::BeginGroup();
            ImGui::SliderInt("FFT Index", &viewer.current_fft_idx, 0, viewer.header.num_ffts > 0 ? viewer.header.num_ffts - 1 : 0);
            ImGui::EndGroup();
            ImGui::Separator();

            float w = ImGui::GetContentRegionAvail().x;
            float total_h = ImGui::GetIO().DisplaySize.y - 140;
            float divider_h = 10.0f;
            float h1 = (total_h - divider_h) * viewer.spectrum_height_ratio;
            float h2 = (total_h - divider_h) * (1.0f - viewer.spectrum_height_ratio);

            if (ImGui::CollapsingHeader("Power Spectrum", ImGuiTreeNodeFlags_DefaultOpen)) {
                viewer.draw_spectrum(w, h1);
            }
            
            ImVec2 divider_pos = ImGui::GetCursorScreenPos();
            ImGui::InvisibleButton("divider", ImVec2(w, divider_h));
            
            ImDrawList *draw_list = ImGui::GetWindowDrawList();
            draw_list->AddLine(ImVec2(divider_pos.x, divider_pos.y + divider_h/2), 
                              ImVec2(divider_pos.x + w, divider_pos.y + divider_h/2), 
                              IM_COL32(100, 100, 100, 200), 2.0f);
            
            if (ImGui::IsItemHovered()) {
                ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNS);
            }
            
            if (ImGui::IsItemActive()) {
                viewer.register_input();
                ImGuiIO& io = ImGui::GetIO();
                float delta = io.MouseDelta.y;
                viewer.spectrum_height_ratio += delta / total_h;
                viewer.spectrum_height_ratio = std::max(0.2f, std::min(0.8f, viewer.spectrum_height_ratio));
            }
            
            if (ImGui::CollapsingHeader("Waterfall", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::PushItemWidth(25);
                ImGui::VSliderFloat("##max", ImVec2(25, h2 * 0.45f), &viewer.display_power_max, 
                                   viewer.header.power_max, viewer.header.power_min, "");
                ImGui::SameLine();
                
                ImDrawList *wf_draw = ImGui::GetWindowDrawList();
                ImVec2 canvas_pos = ImGui::GetCursorScreenPos();
                ImVec2 canvas_size(w - 40, h2);
                
                viewer.draw_waterfall_canvas(wf_draw, canvas_pos, canvas_size);
                
                ImGui::SetCursorPosY(ImGui::GetCursorPosY() - h2 * 0.45f);
                ImGui::VSliderFloat("##min", ImVec2(25, h2 * 0.45f), &viewer.display_power_min, 
                                   viewer.header.power_max, viewer.header.power_min, "");
                ImGui::PopItemWidth();
            }
            ImGui::End();
        }

        ImGui::Render();
        int dw, dh;
        glfwGetFramebufferSize(window, &dw, &dh);
        glViewport(0, 0, dw, dh);
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    viewer.is_running = false;
    capture_thread.join();

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwTerminate();
    
    printf("Streaming viewer closed\n");
}