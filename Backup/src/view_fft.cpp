#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
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
#include <atomic>

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
    
    std::vector<std::vector<float>> waterfall_precomputed;
    std::atomic<bool> precompute_done{false};
    std::atomic<int> precompute_progress{0};
    std::atomic<bool> precompute_requested{true};
    int precomputed_num_pixels = 1360;
    
    std::vector<float> spectrum_cache;
    int cached_spectrum_idx = -1;
    float cached_spectrum_freq_pan = -999.0f;
    float cached_spectrum_freq_zoom = -999.0f;
    int cached_spectrum_pixels = -1;
    
    float last_cached_power_min = -999.0f;
    float last_cached_power_max = -999.0f;
    
    std::chrono::steady_clock::time_point last_input_time;
    bool high_fps_mode = true;
    
    std::string window_title;

    bool load_file(const char *filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            fprintf(stderr, "Failed to open file: %s\n", filename);
            return false;
        }

        file.read(reinterpret_cast<char*>(&header), sizeof(FFTHeader));
        if (std::string(header.magic, 4) != "FFTD") {
            fprintf(stderr, "Invalid FFT data file\n");
            return false;
        }

        printf("FFT Header: num_ffts=%u, fft_size=%u\n", header.num_ffts, header.fft_size);
        printf("Power range: %.1f ~ %.1f dB\n", header.power_min, header.power_max);

        size_t expected_size = static_cast<size_t>(header.num_ffts) * header.fft_size;
        fft_data.resize(expected_size);
        file.read(reinterpret_cast<char*>(fft_data.data()), expected_size * sizeof(int8_t));
        file.close();
        
        char title[256];
        snprintf(title, sizeof(title), "FFT Viewer - %.2f MHz", header.center_frequency / 1e6f);
        window_title = title;
        
        display_power_min = header.power_min;
        display_power_max = header.power_max;
        
        total_duration = (static_cast<double>(header.num_ffts) * header.fft_size) / header.sample_rate;
        
        waterfall_precomputed.resize(header.num_ffts);
        last_input_time = std::chrono::steady_clock::now();
        
        return true;
    }

    void precompute_waterfall_lines(int window_width = 1400) {
        float sr_mhz = header.sample_rate / 1e6f;
        float nyquist = header.sample_rate / 2.0f / 1e6f;
        float total_range = 2.0f * nyquist;
        int num_pixels = window_width - 40;
        precomputed_num_pixels = num_pixels;
        
        float visible_bins = header.fft_size / 1.0f;
        int bin_skip = std::max(1, static_cast<int>(visible_bins / num_pixels));
        
        float disp_start = -nyquist;
        float disp_end = nyquist;
        
        precompute_done = false;
        precompute_progress = 0;
        
        for (int fft_idx = 0; fft_idx < static_cast<int>(header.num_ffts); fft_idx++) {
            waterfall_precomputed[fft_idx].assign(num_pixels, -1e10f);
            
            const int8_t *spec = fft_data.data() + fft_idx * header.fft_size;
            
            for (int bin = 0; bin < static_cast<int>(header.fft_size); bin += bin_skip) {
                float freq = get_freq_from_bin(bin, sr_mhz);
                
                if (freq < disp_start || freq > disp_end) continue;
                
                float norm_freq = (freq - disp_start) / (disp_end - disp_start);
                int px = static_cast<int>(norm_freq * num_pixels);
                
                if (px >= 0 && px < num_pixels) {
                    float power = dequantize(spec[bin]);
                    waterfall_precomputed[fft_idx][px] = std::max(waterfall_precomputed[fft_idx][px], power);
                }
            }
            
            precompute_progress = ((fft_idx + 1) * 100) / header.num_ffts;
        }
        
        precompute_done = true;
    }

    void update_playback() {
        if (!is_playing) return;
        
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - play_start_time).count();
        
        if (elapsed >= total_duration) {
            if (is_looping) {
                play_start_time = now;
                elapsed = 0.0;
            } else {
                is_playing = false;
                return;
            }
        }
        
        double progress = elapsed / total_duration;
        current_fft_idx = static_cast<int>(progress * (header.num_ffts - 1));
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

    float get_freq_from_bin(int bin, float sr_mhz) {
        int n = header.fft_size;
        if (bin == 0) return 0.0f;
        if (bin <= n / 2) {
            return bin * sr_mhz / n;
        } else {
            return (bin - n) * sr_mhz / n;
        }
    }

    void draw_spectrum(float canvas_width, float canvas_height) {
        ImDrawList *draw_list = ImGui::GetWindowDrawList();
        ImVec2 canvas_pos = ImGui::GetCursorScreenPos();
        ImVec2 canvas_size(canvas_width, canvas_height);

        ImVec2 plot_pos(canvas_pos.x + 40, canvas_pos.y);
        ImVec2 plot_size(canvas_size.x - 40, canvas_size.y - 25);

        draw_list->AddRectFilled(plot_pos, ImVec2(plot_pos.x + plot_size.x, plot_pos.y + plot_size.y), 
                                 IM_COL32(20, 20, 20, 255));

        if (current_fft_idx >= static_cast<int>(header.num_ffts)) return;

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
                    float power = dequantize(spectrum[bin]);
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
            
            // 호버 시 주파수 표시
            char info[64];
            float abs_freq = freq_mouse + header.center_frequency / 1e6f;
            snprintf(info, sizeof(info), "%.3f MHz", abs_freq);
            ImVec2 text_size = ImGui::CalcTextSize(info);
            float text_x = plot_pos.x + plot_size.x - text_size.x - 5;
            float text_y = plot_pos.y + 5;
            draw_list->AddRectFilled(ImVec2(text_x, text_y), 
                                    ImVec2(text_x + text_size.x + 10, text_y + text_size.y + 5),
                                    IM_COL32(20, 20, 20, 220));
            draw_list->AddText(ImVec2(text_x + 5, text_y + 2), IM_COL32(0, 255, 0, 255), info);
            
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

    void draw_power_grid(ImDrawList *draw_list, ImVec2 pos, ImVec2 size, float range) {
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

    void draw_freq_grid(ImDrawList *draw_list, ImVec2 pos, ImVec2 size, float start, float end) {
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

    void draw_waterfall_canvas(ImDrawList *draw_list, ImVec2 plot_pos, ImVec2 plot_size) {
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

        bool power_changed = (last_cached_power_min != display_power_min || 
                             last_cached_power_max != display_power_max);

        for (int display_row = 0; display_row < display_rows; display_row++) {
            int fft_idx = current_fft_idx - display_rows + 1 + display_row;
            
            if (fft_idx < 0 || fft_idx > current_fft_idx) continue;

            float row_y = plot_pos.y + (display_rows - 1 - display_row) * plot_size.y / display_rows;
            float row_h = plot_size.y / display_rows;

            std::vector<float>& line_data = waterfall_precomputed[fft_idx];

            for (int px = 0; px < num_pixels; px++) {
                float freq_at_px = disp_start + (px / static_cast<float>(num_pixels)) * (disp_end - disp_start);
                
                int precomp_bin = static_cast<int>((freq_at_px + nyquist) / (2.0f * nyquist) * precomputed_num_pixels);
                precomp_bin = std::max(0, std::min(precomputed_num_pixels - 1, precomp_bin));
                
                if (precomp_bin >= 0 && precomp_bin < static_cast<int>(line_data.size())) {
                    float power_range = display_power_max - display_power_min;
                    float np = (line_data[precomp_bin] - display_power_min) / power_range;
                    np = std::max(0.0f, std::min(1.0f, np));

                    float bx = plot_pos.x + px;
                    draw_list->AddRectFilled(ImVec2(bx, row_y), ImVec2(bx + 1, row_y + row_h), get_color(np));
                }
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
                float freq_mouse = disp_start + mx * (disp_end - disp_start);
                
                freq_zoom *= (1.0f + io.MouseWheel * 0.1f);
                freq_zoom = std::max(1.0f, std::min(10.0f, freq_zoom));
                
                float new_width = total_range / freq_zoom;
                float new_start = freq_mouse - (mx * new_width);
                freq_pan = (new_start + nyquist) / total_range;
                freq_pan = std::max(0.0f, std::min(1.0f - 1.0f / freq_zoom, freq_pan));
            }
        }
    }

    ImU32 get_color(float v) {
        v = std::max(0.0f, std::min(1.0f, v));
        // 빨간색 그래디언트: 검정 → 빨강
        float r = v;        // 0 ~ 1
        float g = v * 0.3f; // 0 ~ 0.3 (약간의 주황색)
        float b = 0.0f;
        return IM_COL32(r*255, g*255, b*255, 255);
    }

    float dequantize(int8_t v) {
        return header.power_min + (v / 127.0f) * (header.power_max - header.power_min);
    }
};

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <fftdata_file>\n", argv[0]);
        return 1;
    }

    FFTViewer viewer;
    if (!viewer.load_file(argv[1])) return 1;

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

    std::thread precompute_thread(&FFTViewer::precompute_waterfall_lines, &viewer, 1400);

    bool show_loading = true;
    int last_window_width = 1400;
    std::thread* resize_thread = nullptr;

    while (!glfwWindowShouldClose(window)) {
        viewer.check_adaptive_fps(window);
        
        glfwPollEvents();
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        if (!viewer.precompute_done) {
            ImGui::SetNextWindowPos(ImVec2(ImGui::GetIO().DisplaySize.x * 0.5f - 150, ImGui::GetIO().DisplaySize.y * 0.5f - 50));
            ImGui::SetNextWindowSize(ImVec2(300, 100));
            ImGui::Begin("Loading", &show_loading, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize);
            ImGui::Text("Precomputing waterfall...");
            ImGui::ProgressBar(viewer.precompute_progress / 100.0f, ImVec2(-1, 0));
            ImGui::Text("%d%%", viewer.precompute_progress.load());
            ImGui::End();
        } else {
            viewer.update_playback();

            ImGui::SetNextWindowPos(ImVec2(0, 0));
            ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize);

            if (ImGui::Begin("FFT Viewer", nullptr, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize)) {
                ImGui::BeginGroup();
                
                if (ImGui::Button(viewer.is_playing ? "PAUSE" : "PLAY", ImVec2(60, 0))) {
                    viewer.register_input();
                    if (viewer.is_playing) {
                        viewer.is_playing = false;
                    } else {
                        viewer.is_playing = true;
                        viewer.play_start_time = std::chrono::steady_clock::now();
                    }
                }
                
                ImGui::SameLine();
                ImGui::PushStyleColor(ImGuiCol_Button, viewer.is_looping ? ImVec4(0.0f, 0.7f, 0.0f, 1.0f) : ImVec4(0.3f, 0.3f, 0.3f, 1.0f));
                if (ImGui::Button("LOOP", ImVec2(50, 0))) {
                    viewer.register_input();
                    viewer.is_looping = !viewer.is_looping;
                }
                ImGui::PopStyleColor();
                
                ImGui::SameLine();
                ImGui::SliderInt("FFT Index", &viewer.current_fft_idx, 0, viewer.header.num_ffts - 1);
                
                if (ImGui::IsItemHovered()) {
                    ImGuiIO& io = ImGui::GetIO();
                    if (io.MouseWheel != 0.0f && !viewer.is_playing) {
                        viewer.register_input();
                        int delta = static_cast<int>(io.MouseWheel) * viewer.fft_index_step;
                        viewer.current_fft_idx += delta;
                        viewer.current_fft_idx = std::max(0, std::min(static_cast<int>(viewer.header.num_ffts - 1), viewer.current_fft_idx));
                    }
                }
                
                ImGui::EndGroup();
                ImGui::Separator();

                float w = ImGui::GetContentRegionAvail().x;
                float total_h = ImGui::GetIO().DisplaySize.y - 120;
                float divider_h = 10.0f;
                float h1 = (total_h - divider_h) * viewer.spectrum_height_ratio;
                float h2 = (total_h - divider_h) * (1.0f - viewer.spectrum_height_ratio);

                viewer.draw_spectrum(w, h1);
                
                ImVec2 divider_pos = ImGui::GetCursorScreenPos();
                ImGui::InvisibleButton("divider", ImVec2(w, divider_h));
                
                if (ImGui::IsItemHovered()) ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNS);
                if (ImGui::IsItemActive()) {
                    viewer.register_input();
                    ImGuiIO& io = ImGui::GetIO();
                    viewer.spectrum_height_ratio += io.MouseDelta.y / total_h;
                    viewer.spectrum_height_ratio = std::max(0.2f, std::min(0.8f, viewer.spectrum_height_ratio));
                }
                
                ImDrawList *wf_draw = ImGui::GetWindowDrawList();
                ImVec2 canvas_pos = ImGui::GetCursorScreenPos();
                ImVec2 canvas_size(w, h2);
                
                viewer.draw_waterfall_canvas(wf_draw, canvas_pos, canvas_size);
                ImGui::Dummy(canvas_size);
                
                ImGui::End();
            }
        }

        ImGui::Render();
        int dw, dh;
        glfwGetFramebufferSize(window, &dw, &dh);
        
        if (dw != last_window_width && viewer.precompute_done && dw > 100) {
            last_window_width = dw;
            if (resize_thread != nullptr && resize_thread->joinable()) {
                resize_thread->join();
                delete resize_thread;
            }
            resize_thread = new std::thread(&FFTViewer::precompute_waterfall_lines, &viewer, dw);
            show_loading = true;
        }
        
        show_loading = !viewer.precompute_done;
        
        glViewport(0, 0, dw, dh);
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    precompute_thread.join();
    if (resize_thread != nullptr && resize_thread->joinable()) {
        resize_thread->join();
        delete resize_thread;
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwTerminate();
    return 0;
}