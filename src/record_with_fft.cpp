#include <libbladeRF.h>
#include <fftw3.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <ctime>

#define RX_GAIN 10
#define CHANNEL BLADERF_CHANNEL_RX(0)
#define FFT_SIZE 8192
#define TIME_AVERAGE 10

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

void run_record_fft() {
    float center_freq, sample_rate;
    int duration_seconds;
    
    printf("Enter center frequency (MHz): ");
    scanf("%f", &center_freq);
    
    printf("Enter sample rate (MSPS): ");
    scanf("%f", &sample_rate);
    
    printf("Enter duration (seconds): ");
    scanf("%d", &duration_seconds);
    
    // 자동 파일명 생성 (날짜 기반)
    time_t now = time(nullptr);
    struct tm* timeinfo = localtime(&now);
    char filename[256];
    strftime(filename, sizeof(filename), "%b_%d_%Y_%H_%M_%S.fftdata", timeinfo);
    
    uint64_t samples_to_collect = static_cast<uint64_t>(duration_seconds) * static_cast<uint64_t>(sample_rate * 1e6);
    
    struct bladerf *dev = nullptr;
    int status = bladerf_open(&dev, nullptr);
    if (status != 0) {
        fprintf(stderr, "Failed to open device: %s\n", bladerf_strerror(status));
        return;
    }

    status = bladerf_set_frequency(dev, CHANNEL, static_cast<uint64_t>(center_freq * 1e6));
    if (status != 0) {
        fprintf(stderr, "Failed to set frequency: %s\n", bladerf_strerror(status));
        bladerf_close(dev);
        return;
    }

    status = bladerf_set_sample_rate(dev, CHANNEL, static_cast<uint32_t>(sample_rate * 1e6), nullptr);
    if (status != 0) {
        fprintf(stderr, "Failed to set sample rate: %s\n", bladerf_strerror(status));
        bladerf_close(dev);
        return;
    }

    status = bladerf_set_gain(dev, CHANNEL, RX_GAIN);
    if (status != 0) {
        fprintf(stderr, "Failed to set gain: %s\n", bladerf_strerror(status));
        bladerf_close(dev);
        return;
    }

    status = bladerf_enable_module(dev, CHANNEL, true);
    if (status != 0) {
        fprintf(stderr, "Failed to enable RX: %s\n", bladerf_strerror(status));
        bladerf_close(dev);
        return;
    }

    status = bladerf_sync_config(dev, BLADERF_RX_X1, BLADERF_FORMAT_SC16_Q11, 
                                 512, 16384, 32, 5000);
    if (status != 0) {
        fprintf(stderr, "Failed to configure sync: %s\n", bladerf_strerror(status));
        bladerf_close(dev);
        return;
    }

    printf("Recording: %.2f MHz, %.2f MSPS, %d seconds\n", center_freq, sample_rate, duration_seconds);
    printf("Output file: %s\n", filename);

    FFTHeader header;
    std::memcpy(header.magic, "FFTD", 4);
    header.version = 1;
    header.fft_size = FFT_SIZE;
    header.sample_rate = static_cast<uint32_t>(sample_rate * 1e6);
    header.center_frequency = static_cast<uint64_t>(center_freq * 1e6);
    header.time_average = TIME_AVERAGE;
    header.power_min = -80.0f;
    header.power_max = 30.0f;

    int16_t *iq_buffer = new int16_t[FFT_SIZE * 2];
    uint64_t received = 0;
    uint32_t num_ffts = 0;

    fftwf_complex *fft_in = fftwf_alloc_complex(FFT_SIZE);
    fftwf_complex *fft_out = fftwf_alloc_complex(FFT_SIZE);
    fftwf_plan plan = fftwf_plan_dft_1d(FFT_SIZE, fft_in, fft_out, FFTW_FORWARD, FFTW_MEASURE);

    std::vector<float> power_accum(FFT_SIZE, 0.0f);
    std::vector<int8_t> fft_data;
    int fft_count = 0;

    while (received < samples_to_collect) {
        uint64_t samples_needed = samples_to_collect - received;
        uint64_t samples_to_read = std::min((uint64_t)FFT_SIZE, samples_needed);

        status = bladerf_sync_rx(dev, iq_buffer, samples_to_read, nullptr, 5000);
        if (status != 0) {
            fprintf(stderr, "RX error: %s\n", bladerf_strerror(status));
            break;
        }

        for (uint64_t i = 0; i < samples_to_read; i++) {
            fft_in[i][0] = iq_buffer[i * 2] / 2048.0f;
            fft_in[i][1] = iq_buffer[i * 2 + 1] / 2048.0f;
        }

        for (uint64_t i = samples_to_read; i < FFT_SIZE; i++) {
            fft_in[i][0] = 0.0f;
            fft_in[i][1] = 0.0f;
        }

        fftwf_execute(plan);

        for (int i = 0; i < FFT_SIZE; i++) {
            float mag_sq = fft_out[i][0] * fft_out[i][0] + fft_out[i][1] * fft_out[i][1];
            float power_db = 10.0f * std::log10(mag_sq + 1e-10f);
            power_accum[i] += power_db;
        }

        fft_count++;

        if (fft_count >= TIME_AVERAGE) {
            for (int i = 0; i < FFT_SIZE; i++) {
                float avg_power = power_accum[i] / fft_count;
                float normalized = (avg_power - header.power_min) / (header.power_max - header.power_min);
                normalized = std::max(-1.0f, std::min(1.0f, normalized));
                fft_data.push_back(static_cast<int8_t>(normalized * 127));
            }
            num_ffts++;
            fft_count = 0;
            std::fill(power_accum.begin(), power_accum.end(), 0.0f);

            printf("Progress: %u FFTs captured\r", num_ffts);
            fflush(stdout);
        }

        received += samples_to_read;
    }

    printf("\n");

    header.num_ffts = num_ffts;

    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        fprintf(stderr, "Failed to open output file\n");
    } else {
        file.write(reinterpret_cast<char*>(&header), sizeof(FFTHeader));
        file.write(reinterpret_cast<char*>(fft_data.data()), fft_data.size());
        file.close();
        printf("Saved: %s (%u FFTs)\n", filename, num_ffts);
    }

    bladerf_enable_module(dev, CHANNEL, false);
    bladerf_close(dev);
    
    fftwf_destroy_plan(plan);
    fftwf_free(fft_in);
    fftwf_free(fft_out);
    delete[] iq_buffer;
}