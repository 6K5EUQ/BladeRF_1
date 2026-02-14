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
#include <mutex>

#define RX_GAIN 20
#define CHANNEL BLADERF_CHANNEL_RX(0)
#define FFT_SIZE 8192
#define TIME_AVERAGE 50
#define MAX_FFTS_MEMORY 1000
#define FFT_UPDATE_FPS 15
#define FFT_UPDATE_INTERVAL_MS (1000 / FFT_UPDATE_FPS)

#define AXIS_LABEL_WIDTH 50
#define SLIDER_WIDTH 40
#define BOTTOM_LABEL_HEIGHT 30

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

void apply_hann_window(fftwf_complex *fft_in, int fft_size) {
    for (int i = 0; i < fft_size; i++) {
        float window = 0.5f * (1.0f - cosf(2.0f * M_PI * i / (fft_size - 1)));
        fft_in[i][0] *= window;
        fft_in[i][1] *= window;
    }
}

class FFTViewer {
public:
    FFTHeader header;
    std::vector<int8_t> fft_data;
    std::vector<float> waterfall_texture_data;
    GLuint waterfall_texture = 0;
    
    int current_fft_idx = 0;
    int last_waterfall_update_idx = -1;
    int fft_index_step = 1;
    float freq_zoom = 1.0f;
    float freq_pan = 0.0f;
    float display_power_min = 0.0f;
    float display_power_max = 0.0f;
    float spectrum_height_ratio = 0.2f;
    
    bool is_playing = false;
    bool is_looping = false;
    std::chrono::steady_clock::time_point play_start_time;
    double total_duration = 0.0;
    
    std::vector<float> current_spectrum;
    int cached_spectrum_idx = -1;
    float cached_spectrum_freq_pan = -999.0f;
    float cached_spectrum_freq_zoom = -999.0f;
    int cached_spectrum_pixels = -1;
    float cached_spectrum_power_min = -999.0f;
    float cached_spectrum_power_max = -999.0f;
    
    std::chrono::steady_clock::time_point last_input_time;
    std::chrono::steady_clock::time_point last_fft_update_time;
    bool high_fps_mode = true;
    
    struct bladerf *dev = nullptr;
    fftwf_plan fft_plan = nullptr;
    fftwf_complex *fft_in = nullptr;
    fftwf_complex *fft_out = nullptr;
    bool is_running = true;
    int total_ffts_captured = 0;
    
    std::string window_title;
    std::mutex data_mutex;
    int pending_new_fft_idx = -1;

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

        // oversample mode 필요 확인
        bool needs_oversample = (sample_rate_msps > 61.44f);
        
        if (needs_oversample) {
            fprintf(stderr, "Attempting oversample mode for %.2f MSPS\n", sample_rate_msps);
        }

        // set_sample_rate 시도
        uint32_t actual_rate = 0;
        status = bladerf_set_sample_rate(dev, CHANNEL, static_cast<uint32_t>(sample_rate_msps * 1e6), &actual_rate);
        
        if (status != 0) {
            fprintf(stderr, "Failed to set sample rate: %s\n", bladerf_strerror(status));
            if (needs_oversample) {
                fprintf(stderr, "Note: Oversample mode may require FPGA bitstream loaded\n");
                fprintf(stderr, "Check: bladerf-cli -p (look for 'FPGA loaded: Yes')\n");
            }
            bladerf_close(dev);
            return false;
        }

        if (actual_rate != static_cast<uint32_t>(sample_rate_msps * 1e6)) {
            fprintf(stderr, "Warning: Requested rate %.2f MSPS, got %.2f MSPS\n", 
                    sample_rate_msps, actual_rate / 1e6f);
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
                                     512, 16384, 128, 5000);
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
        header.power_min = -40.0f;
        header.power_max = 30.0f;
        header.num_ffts = 0;
        
        fft_data.resize(MAX_FFTS_MEMORY * FFT_SIZE);
        waterfall_texture_data.resize(MAX_FFTS_MEMORY * FFT_SIZE, 0.0f);
        current_spectrum.resize(FFT_SIZE, -80.0f);
        
        char title[256];
        snprintf(title, sizeof(title), "Real-time FFT Viewer - %.2f MHz", center_freq_mhz);
        window_title = title;
        
        display_power_min = header.power_min;
        display_power_max = header.power_max;
        
        fft_in = fftwf_alloc_complex(FFT_SIZE);
        fft_out = fftwf_alloc_complex(FFT_SIZE);
        fft_plan = fftwf_plan_dft_1d(FFT_SIZE, fft_in, fft_out, FFTW_FORWARD, FFTW_MEASURE);
        
        last_input_time = std::chrono::steady_clock::now();
        last_fft_update_time = std::chrono::steady_clock::now();
        
        return true;
    }

    void create_waterfall_texture() {
        if (waterfall_texture != 0) {
            glDeleteTextures(1, &waterfall_texture);
        }
        
        glGenTextures(1, &waterfall_texture);
        glBindTexture(GL_TEXTURE_2D, waterfall_texture);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, FFT_SIZE, MAX_FFTS_MEMORY, 
                     0, GL_RED, GL_FLOAT, waterfall_texture_data.data());
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glBindTexture(GL_TEXTURE_2D, 0);
    }

    void update_waterfall_row(int fft_idx) {
        if (waterfall_texture == 0) return;
        
        int mem_idx = fft_idx % MAX_FFTS_MEMORY;
        int8_t *fft_row = fft_data.data() + mem_idx * FFT_SIZE;
        
        std::vector<float> row_float(FFT_SIZE);
        
        // 올바른 FFT 정렬: 음수주파수(왼쪽) | CF(중앙) | 양수주파수(오른쪽)
        int half = FFT_SIZE / 2;
        
        // 음수 주파수: bin FFT_SIZE/2+1 ~ FFT_SIZE-1 → 화면 왼쪽
        for (int i = 0; i < half; i++) {
            int bin = half + 1 + i;
            float power_db = (fft_row[bin] / 127.0f) * (header.power_max - header.power_min) + header.power_min;
            float normalized = (power_db - display_power_min) / (display_power_max - display_power_min);
            row_float[i] = std::max(0.0f, std::min(1.0f, normalized));
        }
        
        // CF: bin 0 → 화면 중앙
        float power_db = (fft_row[0] / 127.0f) * (header.power_max - header.power_min) + header.power_min;
        float normalized = (power_db - display_power_min) / (display_power_max - display_power_min);
        row_float[half] = std::max(0.0f, std::min(1.0f, normalized));
        
        // 양수 주파수: bin 1 ~ FFT_SIZE/2 → 화면 오른쪽
        for (int i = 1; i <= half; i++) {
            power_db = (fft_row[i] / 127.0f) * (header.power_max - header.power_min) + header.power_min;
            normalized = (power_db - display_power_min) / (display_power_max - display_power_min);
            row_float[half + i] = std::max(0.0f, std::min(1.0f, normalized));
        }
        
        glBindTexture(GL_TEXTURE_2D, waterfall_texture);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, mem_idx, FFT_SIZE, 1, 
                        GL_RED, GL_FLOAT, row_float.data());
        glBindTexture(GL_TEXTURE_2D, 0);
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

            apply_hann_window(fft_in, FFT_SIZE);
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

                {
                    std::lock_guard<std::mutex> lock(data_mutex);
                    for (int i = 0; i < FFT_SIZE; i++) {
                        float avg_power = power_accum[i] / fft_count;
                        float normalized = (avg_power - header.power_min) / (header.power_max - header.power_min);
                        normalized = std::max(-1.0f, std::min(1.0f, normalized));
                        fft_row[i] = static_cast<int8_t>(normalized * 127);
                        current_spectrum[i] = avg_power;
                    }
                    
                    total_ffts_captured++;
                    current_fft_idx = total_ffts_captured - 1;
                    header.num_ffts = std::min(total_ffts_captured, MAX_FFTS_MEMORY);
                    cached_spectrum_idx = -1;
                }

                std::fill(power_accum.begin(), power_accum.end(), 0.0f);
                fft_count = 0;
            }
        }

        delete[] iq_buffer;
    }


    ImU32 get_color(float value) {
        if (value < 0.0f) value = 0.0f;
        if (value > 1.0f) value = 1.0f;
        
        float h = value * 240.0f / 360.0f;
        float s = 1.0f;
        float v = value;
        
        float c = v * s;
        float x = c * (1.0f - fabsf(fmodf(h * 6.0f, 2.0f) - 1.0f));
        float m = v - c;
        
        float r = 0, g = 0, b = 0;
        if (h < 1.0f / 6.0f) { r = c; g = x; b = 0; }
        else if (h < 2.0f / 6.0f) { r = x; g = c; b = 0; }
        else if (h < 3.0f / 6.0f) { r = 0; g = c; b = x; }
        else if (h < 4.0f / 6.0f) { r = 0; g = x; b = c; }
        else if (h < 5.0f / 6.0f) { r = x; g = 0; b = c; }
        else { r = c; g = 0; b = x; }
        
        return IM_COL32(
            (ImU32)((r + m) * 255),
            (ImU32)((g + m) * 255),
            (ImU32)((b + m) * 255),
            255
        );
    }

    void compute_spectrum_line(int num_pixels, float sr_mhz, 
                               float disp_start, float disp_end) {
        current_spectrum.assign(num_pixels, -80.0f);
        
        float nyquist = sr_mhz / 2.0f;
        int half_fft = header.fft_size / 2;
        int mem_idx = current_fft_idx % MAX_FFTS_MEMORY;
        
        for (int px = 0; px < num_pixels; px++) {
            float freq_norm = (float)px / num_pixels;
            float freq_display = disp_start + freq_norm * (disp_end - disp_start);
            
            int bin;
            if (freq_display >= 0.0f) {
                bin = (int)((freq_display / nyquist) * half_fft);
            } else {
                bin = FFT_SIZE + (int)((freq_display / nyquist) * half_fft);
            }
            
            if (bin >= 0 && bin < FFT_SIZE) {
                int8_t raw = fft_data[mem_idx * FFT_SIZE + bin];
                float power = (raw / 127.0f) * (header.power_max - header.power_min) + header.power_min;
                current_spectrum[px] = power;
            }
        }
    }

    void draw_spectrum(float w, float h) {
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0, 0));
        ImGui::BeginChild("spectrum_plot", ImVec2(w, h), false, ImGuiWindowFlags_NoScrollbar);
        ImDrawList *draw_list = ImGui::GetWindowDrawList();
        ImVec2 pos = ImGui::GetCursorScreenPos();

        draw_list->AddRectFilled(pos, ImVec2(pos.x + w, pos.y + h), IM_COL32(10, 10, 10, 255));

        float nyquist = header.sample_rate / 2.0f / 1e6f;
        float total_range = 2.0f * nyquist;
        float disp_start = -nyquist + freq_pan * total_range;
        float disp_width = total_range / freq_zoom;
        float disp_end = disp_start + disp_width;
        
        disp_start = std::max(-nyquist, disp_start);
        disp_end = std::min(nyquist, disp_end);

        float sr_mhz = header.sample_rate / 1e6f;

        float graph_x = pos.x + AXIS_LABEL_WIDTH;
        float graph_y = pos.y;
        float graph_w = w - AXIS_LABEL_WIDTH;
        float graph_h = h - BOTTOM_LABEL_HEIGHT;

        bool cache_valid = (cached_spectrum_idx == current_fft_idx &&
                           cached_spectrum_freq_pan == freq_pan &&
                           cached_spectrum_freq_zoom == freq_zoom &&
                           cached_spectrum_pixels == (int)graph_w &&
                           cached_spectrum_power_min == display_power_min &&
                           cached_spectrum_power_max == display_power_max);

        if (!cache_valid) {
            compute_spectrum_line((int)graph_w, sr_mhz, disp_start, disp_end);
            cached_spectrum_idx = current_fft_idx;
            cached_spectrum_freq_pan = freq_pan;
            cached_spectrum_freq_zoom = freq_zoom;
            cached_spectrum_pixels = (int)graph_w;
            cached_spectrum_power_min = display_power_min;
            cached_spectrum_power_max = display_power_max;
        }
        
        float power_range = display_power_max - display_power_min;
        int num_pixels_sp = static_cast<int>(graph_w);
        
        for (int px = 0; px < num_pixels_sp - 1; px++) {
            if (px >= (int)current_spectrum.size() || px + 1 >= (int)current_spectrum.size()) break;
            
            float p1 = (current_spectrum[px] - display_power_min) / power_range;
            float p2 = (current_spectrum[px + 1] - display_power_min) / power_range;
            p1 = std::max(0.0f, std::min(1.0f, p1));
            p2 = std::max(0.0f, std::min(1.0f, p2));
            
            ImVec2 p1_screen(graph_x + px, graph_y + (1.0f - p1) * graph_h);
            ImVec2 p2_screen(graph_x + px + 1, graph_y + (1.0f - p2) * graph_h);
            
            draw_list->AddLine(p1_screen, p2_screen, IM_COL32(0, 255, 0, 255), 1.5f);
        }

        for (int i = 0; i <= 10; i++) {
            float norm_pos = (float)i / 10.0f;
            float y = graph_y + (1.0f - norm_pos) * graph_h;
            draw_list->AddLine(ImVec2(graph_x, y), ImVec2(graph_x + graph_w, y), 
                              IM_COL32(60, 60, 60, 100), 1.0f);
        }

        int num_freq_ticks = std::max(5, static_cast<int>(10 / freq_zoom));
        for (int i = 0; i <= num_freq_ticks; i++) {
            float freq_norm = (float)i / num_freq_ticks;
            float x = graph_x + freq_norm * graph_w;
            draw_list->AddLine(ImVec2(x, graph_y), ImVec2(x, graph_y + graph_h), 
                              IM_COL32(60, 60, 60, 100), 1.0f);
        }

        for (int i = 0; i <= 10; i++) {
            float power_level = display_power_min + (i / 10.0f) * power_range;
            float norm_pos = (float)i / 10.0f;
            float y = graph_y + (1.0f - norm_pos) * graph_h;
            
            draw_list->AddLine(ImVec2(graph_x - 5, y), ImVec2(graph_x, y), 
                              IM_COL32(100, 100, 100, 200), 1.0f);
            char label[16];
            snprintf(label, sizeof(label), "%.0f", power_level);
            ImVec2 text_size = ImGui::CalcTextSize(label);
            ImVec2 text_pos(graph_x - 10 - text_size.x, y - 7);
            draw_list->AddText(text_pos, IM_COL32(200, 200, 200, 255), label);
        }

        int num_ticks = std::max(5, static_cast<int>(10 / freq_zoom));
        int decimals = (freq_zoom > 5.0f) ? 3 : ((freq_zoom > 2.0f) ? 2 : 1);
        
        for (int i = 0; i <= num_ticks; i++) {
            float freq_norm = (float)i / num_ticks;
            float tick = disp_start + freq_norm * (disp_end - disp_start);
            
            float x = graph_x + freq_norm * graph_w;
            draw_list->AddLine(ImVec2(x, graph_y + graph_h), ImVec2(x, graph_y + graph_h + 5), 
                              IM_COL32(100, 100, 100, 200), 1.0f);
            float abs_f = tick + header.center_frequency / 1e6f;
            char label[32];
            snprintf(label, sizeof(label), "%.*f", decimals, abs_f);
            ImVec2 text_size = ImGui::CalcTextSize(label);
            draw_list->AddText(ImVec2(x - text_size.x / 2, graph_y + graph_h + 8), IM_COL32(0, 255, 0, 255), label);
        }

        ImGui::InvisibleButton("spectrum_canvas", ImVec2(w, h));
        
        ImGuiIO& io = ImGui::GetIO();
        if (ImGui::IsItemHovered()) {
            ImVec2 mouse = ImGui::GetMousePos();
            int px = (int)((mouse.x - graph_x) + 0.5f);
            px = std::max(0, std::min((int)graph_w - 1, px));
            
            float nyquist_local = header.sample_rate / 2.0f / 1e6f;
            float total_range_local = 2.0f * nyquist_local;
            float disp_start_local = -nyquist_local + freq_pan * total_range_local;
            float disp_end_local = disp_start_local + total_range_local / freq_zoom;
            
            float freq_norm = (float)px / (float)graph_w;
            float freq_display = disp_start_local + freq_norm * (disp_end_local - disp_start_local);
            float abs_freq = freq_display + header.center_frequency / 1e6f;
            
            float power_db = -80.0f;
            if (px >= 0 && px < (int)current_spectrum.size()) {
                power_db = current_spectrum[px];
            }
            
            char info[64];
            snprintf(info, sizeof(info), "%.3f MHz\n%.1f dB", abs_freq, power_db);
            
            ImVec2 text_size = ImGui::CalcTextSize(info);
            float text_x = graph_x + graph_w - text_size.x;
            float text_y = graph_y;
            
            draw_list->AddRectFilled(ImVec2(text_x, text_y), 
                                    ImVec2(text_x + text_size.x, text_y + text_size.y + 5),
                                    IM_COL32(20, 20, 20, 220));
            draw_list->AddRect(ImVec2(text_x, text_y), 
                              ImVec2(text_x + text_size.x, text_y + text_size.y + 5),
                              IM_COL32(100, 100, 100, 255));
            draw_list->AddText(ImVec2(text_x, text_y + 2), IM_COL32(0, 255, 0, 255), info);
        }
        
        if (ImGui::IsItemHovered() && io.MouseWheel != 0.0f) {
            ImVec2 mouse = ImGui::GetMousePos();
            float mx = (mouse.x - graph_x) / graph_w;
            mx = std::max(0.0f, std::min(1.0f, mx));
            
            float nyquist_local = header.sample_rate / 2.0f / 1e6f;
            float total_range_local = 2.0f * nyquist_local;
            float disp_start_local = -nyquist_local + freq_pan * total_range_local;
            float freq_mouse = disp_start_local + mx * (total_range_local / freq_zoom);
            
            freq_zoom *= (1.0f + io.MouseWheel * 0.1f);
            freq_zoom = std::max(1.0f, std::min(10.0f, freq_zoom));
            
            float new_width = total_range_local / freq_zoom;
            float new_start = freq_mouse - (mx * new_width);
            freq_pan = (new_start + nyquist_local) / total_range_local;
            freq_pan = std::max(0.0f, std::min(1.0f - 1.0f / freq_zoom, freq_pan));
        }

        ImGui::EndChild();
        ImGui::PopStyleVar(2);
    }

    void draw_waterfall_canvas(ImDrawList *draw_list, ImVec2 plot_pos, ImVec2 plot_size) {
        draw_list->AddRectFilled(plot_pos, ImVec2(plot_pos.x + plot_size.x, plot_pos.y + plot_size.y), 
                                 IM_COL32(10, 10, 10, 255));

        float nyquist = header.sample_rate / 2.0f / 1e6f;
        float total_range = 2.0f * nyquist;
        float disp_start = -nyquist + freq_pan * total_range;
        float disp_end = disp_start + total_range / freq_zoom;
        
        disp_start = std::max(-nyquist, disp_start);
        disp_end = std::min(nyquist, disp_end);

        float graph_x = plot_pos.x + AXIS_LABEL_WIDTH;
        float graph_y = plot_pos.y;
        float graph_w = plot_size.x - AXIS_LABEL_WIDTH;
        float graph_h = plot_size.y - BOTTOM_LABEL_HEIGHT;
        
        if (waterfall_texture == 0) {
            create_waterfall_texture();
        }
        
        if (last_waterfall_update_idx != current_fft_idx) {
            update_waterfall_row(current_fft_idx);
            last_waterfall_update_idx = current_fft_idx;
        }
        
        if (waterfall_texture != 0) {
            ImTextureID tex_id = (ImTextureID)(intptr_t)waterfall_texture;
            int display_rows = std::min(static_cast<int>(header.num_ffts), 1000);
            
            // freq_zoom, freq_pan에 따른 U 좌표 범위 계산
            float nyquist = header.sample_rate / 2.0f / 1e6f;
            float total_range = 2.0f * nyquist;
            float disp_start = -nyquist + freq_pan * total_range;
            float disp_end = disp_start + total_range / freq_zoom;
            
            disp_start = std::max(-nyquist, disp_start);
            disp_end = std::min(nyquist, disp_end);
            
            // FFT bin을 U 좌표로 변환
            float u_start = (disp_start + nyquist) / (2.0f * nyquist);
            float u_end = (disp_end + nyquist) / (2.0f * nyquist);
            
            float v_offset = (float)(current_fft_idx % MAX_FFTS_MEMORY) / MAX_FFTS_MEMORY;
            float v_end = v_offset;
            float v_start = v_end + (float)display_rows / MAX_FFTS_MEMORY;
            
            draw_list->AddImage(tex_id, 
                               ImVec2(graph_x, graph_y), 
                               ImVec2(graph_x + graph_w, graph_y + display_rows),
                               ImVec2(u_start, v_start), 
                               ImVec2(u_end, v_end),
                               IM_COL32(255, 255, 255, 255));
        }
        
        ImGui::InvisibleButton("waterfall_canvas", plot_size);
        
        if (ImGui::IsItemHovered()) {
            ImGuiIO& io = ImGui::GetIO();
            ImVec2 mouse = ImGui::GetMousePos();
            int px = (int)((mouse.x - graph_x) + 0.5f);
            px = std::max(0, std::min((int)graph_w - 1, px));
            
            int py = (int)((mouse.y - graph_y) + 0.5f);
            py = std::max(0, std::min((int)graph_h - 1, py));
            
            float nyquist_local = header.sample_rate / 2.0f / 1e6f;
            float total_range_local = 2.0f * nyquist_local;
            float disp_start_local = -nyquist_local + freq_pan * total_range_local;
            float disp_end_local = disp_start_local + total_range_local / freq_zoom;
            
            float freq_norm = (float)px / (float)graph_w;
            float freq_display = disp_start_local + freq_norm * (disp_end_local - disp_start_local);
            float abs_freq = freq_display + header.center_frequency / 1e6f;
            
            // 시간축 인덱스 계산
            int display_rows = std::min(static_cast<int>(header.num_ffts), 1000);
            int time_row = (int)((graph_h - py) / (graph_h / display_rows));
            int fft_idx = current_fft_idx - display_rows + 1 + time_row;
            
            float power_db = -80.0f;
            if (fft_idx >= 0 && fft_idx <= current_fft_idx) {
                int mem_idx = fft_idx % MAX_FFTS_MEMORY;
                int half_fft = header.fft_size / 2;
                
                // px → bin 매핑 (올바른 FFT 정렬)
                int bin;
                if (px < half_fft) {
                    bin = half_fft + 1 + px;  // 음수 주파수
                } else if (px == half_fft) {
                    bin = 0;  // CF
                } else {
                    bin = px - half_fft;  // 양수 주파수
                }
                
                if (bin >= 0 && bin < FFT_SIZE) {
                    int8_t raw = fft_data[mem_idx * FFT_SIZE + bin];
                    power_db = (raw / 127.0f) * (header.power_max - header.power_min) + header.power_min;
                }
            }
            
            char info[64];
            snprintf(info, sizeof(info), "%.3f MHz\n%.1f dB", abs_freq, power_db);
            
            ImVec2 text_size = ImGui::CalcTextSize(info);
            float text_x = graph_x + graph_w - text_size.x;
            float text_y = graph_y;
            
            draw_list->AddRectFilled(ImVec2(text_x, text_y), 
                                    ImVec2(text_x + text_size.x, text_y + text_size.y + 5),
                                    IM_COL32(20, 20, 20, 220));
            draw_list->AddRect(ImVec2(text_x, text_y), 
                              ImVec2(text_x + text_size.x, text_y + text_size.y + 5),
                              IM_COL32(100, 100, 100, 255));
            draw_list->AddText(ImVec2(text_x, text_y + 2), IM_COL32(0, 255, 0, 255), info);
        }
        
        ImGuiIO& io = ImGui::GetIO();
        if (ImGui::IsItemHovered() && io.MouseWheel != 0.0f) {
            ImVec2 mouse = ImGui::GetMousePos();
            float mx = (mouse.x - graph_x) / graph_w;
            mx = std::max(0.0f, std::min(1.0f, mx));
            
            float nyquist_local = header.sample_rate / 2.0f / 1e6f;
            float total_range_local = 2.0f * nyquist_local;
            float disp_start_local = -nyquist_local + freq_pan * total_range_local;
            float freq_mouse = disp_start_local + mx * (total_range_local / freq_zoom);
            
            freq_zoom *= (1.0f + io.MouseWheel * 0.1f);
            freq_zoom = std::max(1.0f, std::min(10.0f, freq_zoom));
            
            float new_width = total_range_local / freq_zoom;
            float new_start = freq_mouse - (mx * new_width);
            freq_pan = (new_start + nyquist_local) / total_range_local;
            freq_pan = std::max(0.0f, std::min(1.0f - 1.0f / freq_zoom, freq_pan));
        }
    }
};

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
    glfwSwapInterval(0);
    glewExperimental = GL_TRUE;
    glewInit();

    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    viewer.create_waterfall_texture();

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize);

        if (ImGui::Begin("##fft_viewer", nullptr, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoTitleBar)) {
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
            ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, 0));
            
            float w = ImGui::GetContentRegionAvail().x;
            float total_h = ImGui::GetIO().DisplaySize.y;
            float divider_h = 15.0f;
            float h1 = (total_h - divider_h) * viewer.spectrum_height_ratio;
            float h2 = (total_h - divider_h) * (1.0f - viewer.spectrum_height_ratio);
            
            // Power Spectrum
            viewer.draw_spectrum(w, h1);
            
            ImVec2 divider_pos = ImGui::GetCursorScreenPos();
            ImGui::InvisibleButton("divider", ImVec2(w, divider_h));
            
            ImDrawList *draw_list = ImGui::GetWindowDrawList();
            draw_list->AddRectFilled(ImVec2(divider_pos.x, divider_pos.y), 
                                    ImVec2(divider_pos.x + w, divider_pos.y + divider_h),
                                    IM_COL32(50, 50, 50, 100));
            draw_list->AddLine(ImVec2(divider_pos.x, divider_pos.y + divider_h/2), 
                              ImVec2(divider_pos.x + w, divider_pos.y + divider_h/2), 
                              IM_COL32(100, 100, 100, 255), 2.0f);
            
            if (ImGui::IsItemHovered()) {
                ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNS);
                draw_list->AddRectFilled(ImVec2(divider_pos.x, divider_pos.y), 
                                        ImVec2(divider_pos.x + w, divider_pos.y + divider_h),
                                        IM_COL32(100, 100, 100, 50));
            }
            
            if (ImGui::IsItemActive()) {
                ImGuiIO& io = ImGui::GetIO();
                float delta = io.MouseDelta.y;
                viewer.spectrum_height_ratio += delta / total_h;
                viewer.spectrum_height_ratio = std::max(0.1f, std::min(0.9f, viewer.spectrum_height_ratio));
            }
            
            // Waterfall
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0, 0));
            ImGui::BeginChild("waterfall_plot", ImVec2(w, h2), false, ImGuiWindowFlags_NoScrollbar);
            ImDrawList *wf_draw = ImGui::GetWindowDrawList();
            ImVec2 canvas_pos = ImGui::GetCursorScreenPos();
            ImVec2 canvas_size(w, h2);
            
            viewer.draw_waterfall_canvas(wf_draw, canvas_pos, canvas_size);
            ImGui::EndChild();
            ImGui::PopStyleVar(2);
            ImGui::PopStyleVar(2);
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