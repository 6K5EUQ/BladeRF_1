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

#define RX_GAIN 30
#define CHANNEL BLADERF_CHANNEL_RX(0)
#define DEFAULT_FFT_SIZE 8192
#define TIME_AVERAGE 50
#define MAX_FFTS_MEMORY 1000
#define FFT_UPDATE_FPS 15
#define FFT_UPDATE_INTERVAL_MS (1000 / FFT_UPDATE_FPS)

// ✅ Hann window power gain compensation (1 / power_sum)
// Hann window: sum(w²) / N ≈ 0.375, 따라서 보정계수 ≈ 2.67
#define HANN_WINDOW_CORRECTION 2.67f

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
    
    int fft_size = DEFAULT_FFT_SIZE;  // 런타임 변경 가능
    int time_average = TIME_AVERAGE;  // fft_size에 따라 조정
    bool fft_size_change_requested = false;
    int pending_fft_size = DEFAULT_FFT_SIZE;
    bool texture_needs_recreate = false;
    
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

    // 자동 스케일링 (1초 누적 heuristic)
    std::vector<float> autoscale_accum;   // 1초치 모든 bin 값 누적
    std::chrono::steady_clock::time_point autoscale_last_update;
    bool autoscale_initialized = false;
    bool autoscale_active = true;  // 주파수 변경 시 true, 1회 갱신 후 false
    
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
    
    // 주파수 변경 관련
    float pending_center_freq = 0.0f;
    bool freq_change_requested = false;
    bool freq_change_in_progress = false;
    
    enum ColorMapType { COLORMAP_JET = 0, COLORMAP_COOL = 1, COLORMAP_HOT = 2, COLORMAP_VIRIDIS = 3 };
    ColorMapType color_map = COLORMAP_COOL;

    bool initialize_bladerf(float center_freq_mhz, float sample_rate_msps) {
        int status = bladerf_open(&dev, nullptr);
        if (status != 0) {
            fprintf(stderr, "Failed to open device: %s\n", bladerf_strerror(status));
            return false;
        }

        status = bladerf_set_frequency(dev, CHANNEL, static_cast<uint64_t>(center_freq_mhz * 1e6));
        if (status != 0) {
            fprintf(stderr, "Failed to set frequency: %s\n", bladerf_strerror(status));
            bladerf_close(dev);
            return false;
        }

        uint32_t actual_rate = 0;
        status = bladerf_set_sample_rate(dev, CHANNEL, static_cast<uint32_t>(sample_rate_msps * 1e6), &actual_rate);
        
        if (status != 0) {
            fprintf(stderr, "Failed to set sample rate: %s\n", bladerf_strerror(status));
            bladerf_close(dev);
            return false;
        }

        if (actual_rate != static_cast<uint32_t>(sample_rate_msps * 1e6)) {
            fprintf(stderr, "Warning: Requested rate %.2f MSPS, got %.2f MSPS\n", 
                    sample_rate_msps, actual_rate / 1e6f);
        }

        // Bandwidth 설정 (sample rate에 맞춰)
        uint32_t bw = static_cast<uint32_t>(sample_rate_msps * 1e6 * 0.8);  // 80% of sample rate
        status = bladerf_set_bandwidth(dev, CHANNEL, bw, nullptr);
        if (status != 0) {
            fprintf(stderr, "Failed to set bandwidth: %s\n", bladerf_strerror(status));
            bladerf_close(dev);
            return false;
        }

        status = bladerf_set_gain(dev, CHANNEL, RX_GAIN);
        if (status != 0) {
            fprintf(stderr, "Failed to set gain: %s\n", bladerf_strerror(status));
            bladerf_close(dev);
            return false;
        }

        status = bladerf_enable_module(dev, CHANNEL, true);
        if (status != 0) {
            fprintf(stderr, "Failed to enable RX: %s\n", bladerf_strerror(status));
            bladerf_close(dev);
            return false;
        }

        unsigned int num_buffers = 512;
        unsigned int buffer_size = 16384;
        
        status = bladerf_sync_config(dev, BLADERF_RX_X1, BLADERF_FORMAT_SC16_Q11,
                                     num_buffers, buffer_size, 128, 10000);
        if (status != 0) {
            fprintf(stderr, "Failed to configure sync: %s\n", bladerf_strerror(status));
            bladerf_close(dev);
            return false;
        }

        printf("BladeRF initialized: %.2f MHz, %.2f MSPS (16-bit mode)\n", 
               center_freq_mhz, sample_rate_msps);
        
        std::memcpy(header.magic, "FFTD", 4);
        header.version = 1;
        header.fft_size = fft_size;
        header.sample_rate = static_cast<uint32_t>(sample_rate_msps * 1e6);
        header.center_frequency = static_cast<uint64_t>(center_freq_mhz * 1e6);
        header.time_average = TIME_AVERAGE;
        header.power_min = -80.0f;  // ✅ 고정값
        header.power_max = -30.0f;    // ✅ 고정값
        header.num_ffts = 0;
        
        fft_data.resize(MAX_FFTS_MEMORY * fft_size);
        waterfall_texture_data.resize(MAX_FFTS_MEMORY * fft_size, 0.0f);
        current_spectrum.resize(fft_size, -80.0f);
        
        char title[256];
        snprintf(title, sizeof(title), "Real-time FFT Viewer - %.2f MHz", center_freq_mhz);
        window_title = title;
        
        // ✅ 고정 dBFS 범위: -80 dB ~ 0 dB
        display_power_min = -80.0f;
        display_power_max = 0.0f;
        
        fft_in = fftwf_alloc_complex(fft_size);
        fft_out = fftwf_alloc_complex(fft_size);
        fft_plan = fftwf_plan_dft_1d(fft_size, fft_in, fft_out, FFTW_FORWARD, FFTW_MEASURE);
        
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
        
        // 초기 색상을 완전 검은색으로 설정 (대비 개선)
        std::vector<uint32_t> init_data(fft_size * MAX_FFTS_MEMORY, IM_COL32(0, 0, 0, 255));
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, fft_size, MAX_FFTS_MEMORY, 
                     0, GL_RGBA, GL_UNSIGNED_BYTE, init_data.data());
        
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glBindTexture(GL_TEXTURE_2D, 0);
    }

    void update_waterfall_row(int fft_idx) {
        if (waterfall_texture == 0) return;
        
        int mem_idx = fft_idx % MAX_FFTS_MEMORY;
        int8_t *fft_row = fft_data.data() + mem_idx * fft_size;
        
        std::vector<uint32_t> row_uint32(fft_size);
        
        auto get_jet_color = [](float v) -> uint32_t {
            // 극적인 Jet colormap (SDR++ 스타일)
            float r, g, b;
            
            if (v < 0.04f) {
                // 검은색 ~ 어두운 파란색
                r = 0.0f;
                g = 0.0f;
                b = v / 0.04f * 0.5f;
            } else if (v < 0.15f) {
                // 어두운 파란색 ~ 밝은 파란색
                float t = (v - 0.04f) / 0.11f;
                r = 0.0f;
                g = 0.0f;
                b = 0.5f + t * 0.5f;
            } else if (v < 0.35f) {
                // 파란색 ~ 초록색
                float t = (v - 0.15f) / 0.2f;
                r = 0.0f;
                g = t;
                b = 1.0f - t * 0.3f;
            } else if (v < 0.55f) {
                // 초록색 ~ 노란색
                float t = (v - 0.35f) / 0.2f;
                r = t;
                g = 1.0f;
                b = 0.0f;
            } else if (v < 0.75f) {
                // 노란색 ~ 주황색/빨강
                float t = (v - 0.55f) / 0.2f;
                r = 1.0f;
                g = 1.0f - t * 0.5f;
                b = 0.0f;
            } else if (v < 0.95f) {
                // 주황색 ~ 빨강
                float t = (v - 0.75f) / 0.2f;
                r = 1.0f;
                g = 0.5f - t * 0.5f;
                b = 0.0f;
            } else {
                // 밝은 빨강 ~ 흰색
                float t = (v - 0.95f) / 0.05f;
                r = 1.0f;
                g = t * 0.3f;
                b = t * 0.3f;
            }
            
            return IM_COL32((uint8_t)(r*255), (uint8_t)(g*255), (uint8_t)(b*255), 255);
        };
        
        int half = fft_size / 2;
        
        // display_power_min/max 를 스냅샷으로 캡처 (스펙트럼과 동일 기준)
        float wf_power_min = display_power_min;
        float wf_power_max = display_power_max;
        float wf_range = wf_power_max - wf_power_min;
        if (wf_range < 1.0f) wf_range = 1.0f;  // 0 나눗셈 방지

        // 음수 주파수
        for (int i = 0; i < half; i++) {
            int bin = half + 1 + i;
            float power_db = (fft_row[bin] / 127.0f) * (header.power_max - header.power_min) + header.power_min;
            float normalized = (power_db - wf_power_min) / wf_range;
            normalized = std::max(0.0f, std::min(1.0f, normalized));
            row_uint32[i] = get_jet_color(normalized);
        }
        
        // CF (중심 주파수)
        float power_db = (fft_row[0] / 127.0f) * (header.power_max - header.power_min) + header.power_min;
        float normalized = (power_db - wf_power_min) / wf_range;
        normalized = std::max(0.0f, std::min(1.0f, normalized));
        row_uint32[half] = get_jet_color(normalized);
        
        // 양수 주파수
        for (int i = 1; i <= half; i++) {
            power_db = (fft_row[i] / 127.0f) * (header.power_max - header.power_min) + header.power_min;
            normalized = (power_db - wf_power_min) / wf_range;
            normalized = std::max(0.0f, std::min(1.0f, normalized));
            row_uint32[half + i] = get_jet_color(normalized);
        }
        
        glBindTexture(GL_TEXTURE_2D, waterfall_texture);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, mem_idx, fft_size, 1, 
                        GL_RGBA, GL_UNSIGNED_BYTE, row_uint32.data());
        glBindTexture(GL_TEXTURE_2D, 0);
    }

    void capture_and_process() {
        int16_t *iq_buffer = new int16_t[fft_size * 2];
        std::vector<float> power_accum(fft_size, 0.0f);
        int fft_count = 0;

        while (is_running) {
            // FFT size 변경 요청 처리
            if (fft_size_change_requested) {
                fft_size_change_requested = false;
                int new_size = pending_fft_size;
                
                // fftw 재할당
                fftwf_destroy_plan(fft_plan);
                fftwf_free(fft_in);
                fftwf_free(fft_out);
                
                fft_size = new_size;
                time_average = TIME_AVERAGE * DEFAULT_FFT_SIZE / fft_size;
                if (time_average < 1) time_average = 1;
                fft_in = fftwf_alloc_complex(fft_size);
                fft_out = fftwf_alloc_complex(fft_size);
                fft_plan = fftwf_plan_dft_1d(fft_size, fft_in, fft_out, FFTW_FORWARD, FFTW_MEASURE);
                
                delete[] iq_buffer;
                iq_buffer = new int16_t[fft_size * 2];
                power_accum.assign(fft_size, 0.0f);
                fft_count = 0;
                
                {
                    std::lock_guard<std::mutex> lock(data_mutex);
                    header.fft_size = fft_size;
                    fft_data.assign(MAX_FFTS_MEMORY * fft_size, 0);
                    waterfall_texture_data.assign(MAX_FFTS_MEMORY * fft_size, 0.0f);
                    current_spectrum.assign(fft_size, -80.0f);
                    total_ffts_captured = 0;
                    current_fft_idx = 0;
                    cached_spectrum_idx = -1;
                    autoscale_accum.clear();
                    autoscale_initialized = false;
                    autoscale_active = true;
                }
                
                printf("FFT size changed to %d\n", fft_size);
                texture_needs_recreate = true;
                continue;
            }
            // 주파수 변경 요청 확인
            if (freq_change_requested && !freq_change_in_progress) {
                freq_change_in_progress = true;
                int status = bladerf_set_frequency(dev, CHANNEL, static_cast<uint64_t>(pending_center_freq * 1e6));
                if (status == 0) {
                    {
                        std::lock_guard<std::mutex> lock(data_mutex);
                        header.center_frequency = static_cast<uint64_t>(pending_center_freq * 1e6);
                    }
                    printf("Frequency changed to: %.2f MHz\n", pending_center_freq);
                    char title[256];
                    snprintf(title, sizeof(title), "Real-time FFT Viewer - %.2f MHz", pending_center_freq);
                    window_title = title;
                    // 주파수 변경 후 autoscale 재활성
                    autoscale_accum.clear();
                    autoscale_initialized = false;
                    autoscale_active = true;
                } else {
                    fprintf(stderr, "Failed to change frequency: %s\n", bladerf_strerror(status));
                }
                freq_change_requested = false;
                freq_change_in_progress = false;
            }
            
            int status = bladerf_sync_rx(dev, iq_buffer, fft_size, nullptr, 10000);
            if (status != 0) {
                fprintf(stderr, "RX error: %s\n", bladerf_strerror(status));
                continue;
            }

            for (int i = 0; i < fft_size; i++) {
                fft_in[i][0] = iq_buffer[i * 2] / 2048.0f;
                fft_in[i][1] = iq_buffer[i * 2 + 1] / 2048.0f;
            }

            apply_hann_window(fft_in, fft_size);
            fftwf_execute(fft_plan);

            for (int i = 0; i < fft_size; i++) {
                float mag_sq = fft_out[i][0] * fft_out[i][0] + fft_out[i][1] * fft_out[i][1];
                
                // ✅ SDR++ 기준 dBFS 계산:
                // 1. FFT normalization: |X|² / N²
                float normalized_power = mag_sq / (fft_size * fft_size);
                
                // 2. Window gain compensation (Hann window)
                normalized_power *= HANN_WINDOW_CORRECTION;
                
                // 3. dBFS 변환 (기준: ±1.0 normalized IQ)
                float power_db = 10.0f * log10(normalized_power + 1e-10f);
                
                power_accum[i] += power_db;
            }

            // DC spike 제거
            power_accum[0] = (power_accum[1] + power_accum[fft_size-1]) / 2.0f;

            fft_count++;

            if (fft_count >= time_average) {
                int fft_idx = total_ffts_captured % MAX_FFTS_MEMORY;
                int8_t *fft_row = fft_data.data() + fft_idx * fft_size;

                {
                    std::lock_guard<std::mutex> lock(data_mutex);
                    
                    for (int i = 0; i < fft_size; i++) {
                        float avg_power = power_accum[i] / fft_count;
                        float normalized = (avg_power - header.power_min) / (header.power_max - header.power_min);
                        normalized = std::max(-1.0f, std::min(1.0f, normalized));
                        fft_row[i] = static_cast<int8_t>(normalized * 127);
                        current_spectrum[i] = avg_power;
                    }

                    // 자동 스케일링: 주파수 변경 후 1초 누적 → 하위 15 percentile → display_power_min
                    if (autoscale_active) {
                        if (!autoscale_initialized) {
                            autoscale_accum.reserve(fft_size * 200);
                            autoscale_last_update = std::chrono::steady_clock::now();
                            autoscale_initialized = true;
                        }
                        for (int i = 1; i < fft_size; i++) {
                            autoscale_accum.push_back(current_spectrum[i]);
                        }
                        auto now = std::chrono::steady_clock::now();
                        float elapsed = std::chrono::duration<float>(now - autoscale_last_update).count();
                        if (elapsed >= 1.0f && !autoscale_accum.empty()) {
                            size_t idx = (size_t)(autoscale_accum.size() * 0.15f);
                            std::nth_element(autoscale_accum.begin(),
                                             autoscale_accum.begin() + idx,
                                             autoscale_accum.end());
                            float p15 = autoscale_accum[idx];
                            display_power_min = p15 - 10.0f;
                            autoscale_accum.clear();
                            autoscale_active = false;  // 1회 갱신 후 비활성
                            cached_spectrum_idx = -1;
                        }
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
        
        float r = 0, g = 0, b = 0;
        
        switch(color_map) {
            case COLORMAP_JET: {
                // 무지개: 파랑→초록→노랑→빨강
                float h = value * 240.0f / 360.0f;
                float s = 1.0f;
                float v = value;
                
                float c = v * s;
                float x = c * (1.0f - fabsf(fmodf(h * 6.0f, 2.0f) - 1.0f));
                float m = v - c;
                
                if (h < 1.0f / 6.0f) { r = c; g = x; b = 0; }
                else if (h < 2.0f / 6.0f) { r = x; g = c; b = 0; }
                else if (h < 3.0f / 6.0f) { r = 0; g = c; b = x; }
                else if (h < 4.0f / 6.0f) { r = 0; g = x; b = c; }
                else if (h < 5.0f / 6.0f) { r = x; g = 0; b = c; }
                else { r = c; g = 0; b = x; }
                
                r += m; g += m; b += m;
                break;
            }
            case COLORMAP_COOL: {
                // 파란 하늘색: 검정→파랑→하늘색
                r = value * 0.3f;
                g = value * 0.7f + 0.2f;
                b = 0.8f + value * 0.2f;
                break;
            }
            case COLORMAP_HOT: {
                // 검정→빨강→노랑→흰색
                if (value < 0.33f) {
                    r = value * 3.0f;
                    g = 0.0f;
                    b = 0.0f;
                } else if (value < 0.67f) {
                    r = 1.0f;
                    g = (value - 0.33f) * 3.0f;
                    b = 0.0f;
                } else {
                    r = 1.0f;
                    g = 1.0f;
                    b = (value - 0.67f) * 3.0f;
                }
                break;
            }
            case COLORMAP_VIRIDIS: {
                // 보라→초록→노랑
                if (value < 0.25f) {
                    r = 0.267 + value * 0.5f;
                    g = 0.004 + value * 0.2f;
                    b = 0.329 + value * 0.8f;
                } else if (value < 0.5f) {
                    r = 0.293 + (value - 0.25f) * 0.4f;
                    g = 0.058 + (value - 0.25f) * 0.8f;
                    b = 0.633 - (value - 0.25f) * 1.0f;
                } else if (value < 0.75f) {
                    r = 0.553 + (value - 0.5f) * 1.0f;
                    g = 0.258 + (value - 0.5f) * 1.5f;
                    b = 0.029 + (value - 0.5f) * 0.2f;
                } else {
                    r = 0.993;
                    g = 0.906 + (value - 0.75f) * 0.2f;
                    b = 0.145;
                }
                break;
            }
        }
        
        return IM_COL32(
            (ImU32)(r * 255),
            (ImU32)(g * 255),
            (ImU32)(b * 255),
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
                bin = fft_size + (int)((freq_display / nyquist) * half_fft);
            }
            
            if (bin >= 0 && bin < fft_size) {
                int8_t raw = fft_data[mem_idx * fft_size + bin];
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
        
        // ✅ 실제 유효 대역폭으로 제한 (양끝 roll-off 제거)
        float effective_nyquist = nyquist * 0.875f;  // AD9361 실제 유효 BW: 87.5%
        float effective_total_range = 2.0f * effective_nyquist;
        float disp_start = -effective_nyquist + freq_pan * effective_total_range;
        float disp_width = effective_total_range / freq_zoom;
        float disp_end = disp_start + disp_width;
        
        disp_start = std::max(-effective_nyquist, disp_start);
        disp_end = std::min(effective_nyquist, disp_end);

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
            
            // ✅ 0dB를 위에, -80dB를 아래에: (1.0f - p1)으로 역전
            ImVec2 p1_screen(graph_x + px, graph_y + (1.0f - p1) * graph_h);
            ImVec2 p2_screen(graph_x + px + 1, graph_y + (1.0f - p2) * graph_h);
            
            draw_list->AddLine(p1_screen, p2_screen, IM_COL32(0, 255, 0, 255), 1.5f);
        }

        for (int i = 0; i <= 10; i++) {
            float norm_pos = (float)i / 10.0f;
            float y = graph_y + (1.0f - norm_pos) * graph_h;  // ✅ 역전: 위부터 아래로
            draw_list->AddLine(ImVec2(graph_x, y), ImVec2(graph_x + graph_w, y), 
                              IM_COL32(60, 60, 60, 100), 1.0f);
        }

        // dB 수평 그리드 (변경 없음)
        for (int i = 1; i <= 9; i++) {
            float power_level = 0.0f - (i / 10.0f) * 80.0f;
            float norm_pos = (float)i / 10.0f;
            float y = graph_y + norm_pos * graph_h;
            
            draw_list->AddLine(ImVec2(graph_x - 5, y), ImVec2(graph_x, y), 
                              IM_COL32(100, 100, 100, 200), 1.0f);
            char label[16];
            snprintf(label, sizeof(label), "%.0f", power_level);
            ImVec2 text_size = ImGui::CalcTextSize(label);
            ImVec2 text_pos(graph_x - 10 - text_size.x, y - 7);
            draw_list->AddText(text_pos, IM_COL32(200, 200, 200, 255), label);
        }

        // 주파수 그리드: zoom==1이면 5MHz 단위 정수, zoom>1이면 10등분 소수점3자리
        {
            float cf_mhz = header.center_frequency / 1e6f;
            float disp_range = disp_end - disp_start;  // MHz (relative)

            if (freq_zoom <= 1.0f) {
                // 5MHz 단위 절대 주파수 그리드
                float abs_start = disp_start + cf_mhz;
                float abs_end   = disp_end   + cf_mhz;
                float step = 5.0f;
                float first_tick = ceilf(abs_start / step) * step;
                for (float abs_f = first_tick; abs_f <= abs_end + 1e-4f; abs_f += step) {
                    float rel = abs_f - cf_mhz;
                    float x = graph_x + (rel - disp_start) / disp_range * graph_w;
                    if (x < graph_x || x > graph_x + graph_w) continue;
                    draw_list->AddLine(ImVec2(x, graph_y), ImVec2(x, graph_y + graph_h),
                                      IM_COL32(60, 60, 60, 100), 1.0f);
                    draw_list->AddLine(ImVec2(x, graph_y + graph_h), ImVec2(x, graph_y + graph_h + 5),
                                      IM_COL32(100, 100, 100, 200), 1.0f);
                    char label[32];
                    snprintf(label, sizeof(label), "%.0f", abs_f);
                    ImVec2 ts = ImGui::CalcTextSize(label);
                    draw_list->AddText(ImVec2(x - ts.x / 2, graph_y + graph_h + 8),
                                      IM_COL32(0, 255, 0, 255), label);
                }
            } else {
                // 줌 상태: 10등분 고정, 소수점 3자리
                for (int i = 0; i <= 10; i++) {
                    float freq_norm = (float)i / 10.0f;
                    float rel = disp_start + freq_norm * disp_range;
                    float x = graph_x + freq_norm * graph_w;
                    draw_list->AddLine(ImVec2(x, graph_y), ImVec2(x, graph_y + graph_h),
                                      IM_COL32(60, 60, 60, 100), 1.0f);
                    draw_list->AddLine(ImVec2(x, graph_y + graph_h), ImVec2(x, graph_y + graph_h + 5),
                                      IM_COL32(100, 100, 100, 200), 1.0f);
                    float abs_f = rel + cf_mhz;
                    char label[32];
                    snprintf(label, sizeof(label), "%.3f", abs_f);
                    ImVec2 ts = ImGui::CalcTextSize(label);
                    draw_list->AddText(ImVec2(x - ts.x / 2, graph_y + graph_h + 8),
                                      IM_COL32(0, 255, 0, 255), label);
                }
            }
        }

        ImGui::SetCursorScreenPos(ImVec2(graph_x, graph_y));
        ImGui::InvisibleButton("spectrum_canvas", ImVec2(graph_w, graph_h));
        
        ImGuiIO& io = ImGui::GetIO();
        
        // spectrum_canvas 호버 & 마우스휠 처리
        if (ImGui::IsItemHovered()) {
            ImVec2 mouse = ImGui::GetMousePos();
            int px = (int)((mouse.x - graph_x) + 0.5f);
            px = std::max(0, std::min((int)graph_w - 1, px));
            
            float nyquist_local = header.sample_rate / 2.0f / 1e6f;
            float effective_nyquist_local = nyquist_local * 0.875f;  // ✅ 87.5% 유효 대역폭
            float total_range_local = 2.0f * effective_nyquist_local;
            float disp_start_local = -effective_nyquist_local + freq_pan * total_range_local;
            float disp_end_local = disp_start_local + total_range_local / freq_zoom;
            
            float freq_norm = (float)px / (float)graph_w;
            float freq_display = disp_start_local + freq_norm * (disp_end_local - disp_start_local);
            float abs_freq = freq_display + header.center_frequency / 1e6f;
            
            float power_db = -80.0f;
            if (px >= 0 && px < (int)current_spectrum.size()) {
                power_db = current_spectrum[px];
            }
            
            char info[64];
            snprintf(info, sizeof(info), "%.3f MHz", abs_freq);
            
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
            float effective_nyquist_local = nyquist_local * 0.875f;  // ✅ 87.5% 유효 대역폭
            float total_range_local = 2.0f * effective_nyquist_local;
            float disp_start_local = -effective_nyquist_local + freq_pan * total_range_local;
            float freq_mouse = disp_start_local + mx * (total_range_local / freq_zoom);
            
            freq_zoom *= (1.0f + io.MouseWheel * 0.1f);
            freq_zoom = std::max(1.0f, std::min(10.0f, freq_zoom));
            
            float new_width = total_range_local / freq_zoom;
            float new_start = freq_mouse - (mx * new_width);
            freq_pan = (new_start + effective_nyquist_local) / total_range_local;
            freq_pan = std::max(0.0f, std::min(1.0f - 1.0f / freq_zoom, freq_pan));
        }
        
        // Y축 드래그로 power_min/max 조절
        ImGui::SetCursorScreenPos(ImVec2(pos.x, graph_y));
        ImGui::InvisibleButton("power_axis_drag", ImVec2(AXIS_LABEL_WIDTH, graph_h));
        
        static float drag_start_y = 0.0f;
        static float drag_start_min = 0.0f;
        static float drag_start_max = 0.0f;
        static bool is_dragging_lower = false;
        
        if (ImGui::IsItemActive()) {
            if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                // 드래그 시작 - 시작점 저장
                ImVec2 mouse = ImGui::GetMousePos();
                float mid_power = (display_power_min + display_power_max) / 2.0f;
                float mid_y = graph_y + graph_h * (1.0f - (mid_power - display_power_min) / (display_power_max - display_power_min));
                
                drag_start_y = mouse.y;
                drag_start_min = display_power_min;
                drag_start_max = display_power_max;
                is_dragging_lower = (mouse.y > mid_y);
            }
            
            if (ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
                ImVec2 mouse = ImGui::GetMousePos();
                float delta_y = mouse.y - drag_start_y;
                float mid_power = (drag_start_min + drag_start_max) / 2.0f;
                
                if (is_dragging_lower) {
                    // 아래쪽(min 영역): delta_y > 0이면 아래로(min 감소)
                    float norm = delta_y / (graph_y + graph_h - (graph_y + graph_h * (1.0f - (mid_power - drag_start_min) / (drag_start_max - drag_start_min))));
                    norm = std::max(-1.0f, std::min(1.0f, norm));
                    float db_change = norm * 50.0f;
                    display_power_min = mid_power - db_change;
                } else {
                    // 위쪽(max 영역): delta_y < 0이면 위로(max 증가)
                    float norm = -delta_y / (graph_y + graph_h * (1.0f - (mid_power - drag_start_min) / (drag_start_max - drag_start_min)));
                    norm = std::max(-1.0f, std::min(1.0f, norm));
                    float db_change = norm * 50.0f;
                    display_power_max = mid_power + db_change;
                }
                
                // 최소 범위 보장
                if (display_power_max - display_power_min < 5.0f) {
                    float mid = (display_power_min + display_power_max) / 2.0f;
                    display_power_min = mid - 2.5f;
                    display_power_max = mid + 2.5f;
                }
                
                cached_spectrum_idx = -1;
            }
            
        }

        ImGui::EndChild();
        ImGui::PopStyleVar(2);
    }

    void draw_waterfall_canvas(ImDrawList *draw_list, ImVec2 plot_pos, ImVec2 plot_size) {
        draw_list->AddRectFilled(plot_pos, ImVec2(plot_pos.x + plot_size.x, plot_pos.y + plot_size.y), 
                                 IM_COL32(10, 10, 10, 255));

        float nyquist = header.sample_rate / 2.0f / 1e6f;
        float total_range = 2.0f * nyquist;
        
        // ✅ 실제 유효 대역폭으로 제한 (양끝 roll-off 제거)
        float effective_nyquist = nyquist * 0.875f;  // AD9361 실제 유효 BW: 87.5%
        float effective_total_range = 2.0f * effective_nyquist;
        float disp_start = -effective_nyquist + freq_pan * effective_total_range;
        float disp_end = disp_start + effective_total_range / freq_zoom;
        
        disp_start = std::max(-effective_nyquist, disp_start);
        disp_end = std::min(effective_nyquist, disp_end);

        float graph_x = plot_pos.x + AXIS_LABEL_WIDTH;
        float graph_y = plot_pos.y;
        float graph_w = plot_size.x - AXIS_LABEL_WIDTH;
        float graph_h = plot_size.y - BOTTOM_LABEL_HEIGHT;
        
        if (waterfall_texture == 0) {
            create_waterfall_texture();
        }
        
        // 새 FFT row만 현재 display_power_min/max 기준으로 색상 계산
        if (total_ffts_captured > 0 && last_waterfall_update_idx != current_fft_idx) {
            update_waterfall_row(current_fft_idx);
            last_waterfall_update_idx = current_fft_idx;
        }
        
        if (waterfall_texture != 0) {
            ImTextureID tex_id = (ImTextureID)(intptr_t)waterfall_texture;
            int display_rows = std::min(static_cast<int>(total_ffts_captured), MAX_FFTS_MEMORY);
            
            float nyquist = header.sample_rate / 2.0f / 1e6f;
            float effective_nyquist = nyquist * 0.875f;
            float total_range = 2.0f * effective_nyquist;
            float disp_start = -effective_nyquist + freq_pan * total_range;
            float disp_end = disp_start + total_range / freq_zoom;
            
            disp_start = std::max(-effective_nyquist, disp_start);
            disp_end = std::min(effective_nyquist, disp_end);
            
            float u_start = (disp_start + nyquist) / (2.0f * nyquist);
            float u_end = (disp_end + nyquist) / (2.0f * nyquist);
            
            float v_newest = (float)(current_fft_idx % MAX_FFTS_MEMORY) / MAX_FFTS_MEMORY;
            // 화면 상단 = 최신, 화면 하단 = 오래된 데이터
            // v축: 최신 row부터 display_rows만큼 거슬러 올라감
            float v_top    = v_newest + 1.0f / MAX_FFTS_MEMORY;  // 최신 row 바로 다음(상단)
            float v_bottom = v_top - (float)display_rows / MAX_FFTS_MEMORY;
            // GL_TEXTURE_WRAP_T = REPEAT 이므로 음수 UV도 wrap됨
            
            // 실제 데이터 높이만큼만 그림 (초기 검은창 방지)
            float draw_h = (display_rows >= (int)graph_h)
                ? graph_h
                : (float)display_rows;
            // draw_h에 맞게 v_bottom 재조정
            float v_draw_bottom = v_top - (float)display_rows / MAX_FFTS_MEMORY;
            
            draw_list->AddImage(tex_id,
                               ImVec2(graph_x, graph_y),
                               ImVec2(graph_x + graph_w, graph_y + draw_h),
                               ImVec2(u_start, v_top),
                               ImVec2(u_end,   v_draw_bottom),
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
            float effective_nyquist_local = nyquist_local * 0.875f;  // ✅ 87.5% 유효 대역폭
            float total_range_local = 2.0f * effective_nyquist_local;
            float disp_start_local = -effective_nyquist_local + freq_pan * total_range_local;
            float disp_end_local = disp_start_local + total_range_local / freq_zoom;
            
            float freq_norm = (float)px / (float)graph_w;
            float freq_display = disp_start_local + freq_norm * (disp_end_local - disp_start_local);
            float abs_freq = freq_display + header.center_frequency / 1e6f;
            
            int display_rows = std::min(static_cast<int>(total_ffts_captured), 1000);
            int time_row = (int)((graph_h - py) / (graph_h / display_rows));
            int fft_idx = current_fft_idx - display_rows + 1 + time_row;
            
            float power_db = -80.0f;
            if (fft_idx >= 0 && fft_idx <= current_fft_idx) {
                int mem_idx = fft_idx % MAX_FFTS_MEMORY;
                int half_fft = header.fft_size / 2;
                
                int bin;
                if (px < half_fft) {
                    bin = half_fft + 1 + px;
                } else if (px == half_fft) {
                    bin = 0;
                } else {
                    bin = px - half_fft;
                }
                
                if (bin >= 0 && bin < fft_size) {
                    int8_t raw = fft_data[mem_idx * fft_size + bin];
                    power_db = (raw / 127.0f) * (header.power_max - header.power_min) + header.power_min;
                }
            }
            
            char info[64];
            snprintf(info, sizeof(info), "%.3f MHz", abs_freq);
            
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
            float effective_nyquist_local = nyquist_local * 0.875f;  // ✅ 87.5% 유효 대역폭
            float total_range_local = 2.0f * effective_nyquist_local;
            float disp_start_local = -effective_nyquist_local + freq_pan * total_range_local;
            float freq_mouse = disp_start_local + mx * (total_range_local / freq_zoom);
            
            freq_zoom *= (1.0f + io.MouseWheel * 0.1f);
            freq_zoom = std::max(1.0f, std::min(10.0f, freq_zoom));
            
            float new_width = total_range_local / freq_zoom;
            float new_start = freq_mouse - (mx * new_width);
            freq_pan = (new_start + effective_nyquist_local) / total_range_local;
            freq_pan = std::max(0.0f, std::min(1.0f - 1.0f / freq_zoom, freq_pan));
        }
    }
};

void run_streaming_viewer() {
    float center_freq = 450.0f;
    float sample_rate = 61.44f;
    
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

        if (viewer.texture_needs_recreate) {
            viewer.texture_needs_recreate = false;
            viewer.create_waterfall_texture();
        }

        ImGui::SetNextWindowPos(ImVec2(0, 0));
        ImGui::SetNextWindowSize(ImGui::GetIO().DisplaySize);

        if (ImGui::Begin("##fft_viewer", nullptr, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoTitleBar)) {
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
            ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, 0));
            
            // 상단바 UI
            ImGui::PopStyleVar(2);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(5, 5));
            ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(8, 5));
            
            // 주파수 입력 (Enter로 활성화, 숫자 입력 후 Enter로 적용 및 비활성화)
            static float new_freq = 450.0f;
            static bool freq_deactivate = false;
            
            if (!ImGui::IsAnyItemActive()) {
                if (ImGui::IsKeyPressed(ImGuiKey_Enter) || ImGui::IsKeyPressed(ImGuiKey_KeypadEnter)) {
                    ImGui::SetKeyboardFocusHere();
                }
            }
            
            if (freq_deactivate) {
                freq_deactivate = false;
                ImGui::SetWindowFocus(nullptr);
            }
            
            ImGui::SetNextItemWidth(120);
            ImGui::InputFloat("Freq (MHz)", &new_freq, 0.0f, 0.0f, "%.3f");
            // IsItemDeactivatedAfterEdit: Enter 또는 포커스 이탈 시 호출됨
            if (ImGui::IsItemDeactivatedAfterEdit()) {
                viewer.pending_center_freq = new_freq;
                viewer.freq_change_requested = true;
                freq_deactivate = true;
            }
            
            ImGui::SameLine();
            ImGui::Spacing(); ImGui::SameLine();
            
            // FFT Size 드롭다운
            static const int fft_sizes[] = {512, 1024, 2048, 4096, 8192, 16384, 32768};
            static const char* fft_size_labels[] = {"512", "1024", "2048", "4096", "8192", "16384", "32768"};
            static int fft_size_idx = 4; // 기본값 8192
            ImGui::Text("FFT Size:"); ImGui::SameLine();
            ImGui::SetNextItemWidth(90);
            if (ImGui::BeginCombo("##fftsize", fft_size_labels[fft_size_idx])) {
                for (int i = 0; i < 7; i++) {
                    bool selected = (fft_size_idx == i);
                    if (ImGui::Selectable(fft_size_labels[i], selected)) {
                        fft_size_idx = i;
                        viewer.pending_fft_size = fft_sizes[i];
                        viewer.fft_size_change_requested = true;
                    }
                    if (selected) ImGui::SetItemDefaultFocus();
                }
                ImGui::EndCombo();
            }
            
            ImGui::PopStyleVar(2);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
            ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, 0));
            
            float w = ImGui::GetContentRegionAvail().x;
            float total_h = ImGui::GetIO().DisplaySize.y - 35;
            float divider_h = 15.0f;
            float h1 = (total_h - divider_h) * viewer.spectrum_height_ratio;
            
            viewer.draw_spectrum(w, h1);
            
            ImVec2 divider_pos = ImGui::GetCursorScreenPos();
            ImGui::InvisibleButton("divider", ImVec2(w, divider_h));
            
            if (ImGui::IsItemActive()) {
                ImGuiIO& io = ImGui::GetIO();
                float delta = io.MouseDelta.y;
                viewer.spectrum_height_ratio += delta / total_h;
                viewer.spectrum_height_ratio = std::max(0.1f, std::min(0.9f, viewer.spectrum_height_ratio));
            }
            
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0, 0));
            float remaining_h = ImGui::GetContentRegionAvail().y;
            ImGui::BeginChild("waterfall_plot", ImVec2(w, remaining_h), false, ImGuiWindowFlags_NoScrollbar);
            ImDrawList *wf_draw = ImGui::GetWindowDrawList();
            ImVec2 canvas_pos = ImGui::GetCursorScreenPos();
            ImVec2 canvas_size(w, remaining_h);
            
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

int main() {
    run_streaming_viewer();
    return 0;
}
