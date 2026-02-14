#include <libbladeRF.h>
#include <fftw3.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <vector>
#include <ctime>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <unistd.h>
#include <chrono>
#include <cmath>

#define RX_GAIN       10
#define CHANNEL       BLADERF_CHANNEL_RX(0)
#define FFT_SIZE      8192
#define TIME_AVERAGE  10

struct WavHeader {
    char riff[4] = {'R','I','F','F'};
    uint32_t file_size;
    char wave[4] = {'W','A','V','E'};
    char fmt[4]  = {'f','m','t',' '};
    uint32_t fmt_size = 16;
    uint16_t audio_format = 1;
    uint16_t num_channels = 2;
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_sample = 16;
    char data[4] = {'d','a','t','a'};
    uint32_t data_size;
};

struct FFTHeader {
    char magic[4] = {'F','F','T','D'};
    uint32_t version = 1;
    uint32_t fft_size;
    uint32_t sample_rate;
    uint64_t center_frequency;
    uint32_t num_ffts;
    uint32_t time_average;
    float power_min;
    float power_max;
    float reserved[8] = {0};
};

int main() {
    double freq_mhz;
    double rate_msps;
    int duration_sec;

    std::cout << "Enter center frequency (MHz): ";
    std::cin >> freq_mhz;
    std::cout << "Enter sample rate (MSPS): ";
    std::cin >> rate_msps;
    std::cout << "Enter duration (seconds): ";
    std::cin >> duration_sec;

    uint64_t center_freq = static_cast<uint64_t>(freq_mhz * 1e6);
    uint32_t sample_rate = static_cast<uint32_t>(rate_msps * 1e6);

    std::time_t t = std::time(nullptr);
    std::tm *tm = std::localtime(&t);

    uint64_t mhz = center_freq / 1000000;
    uint64_t khz = (center_freq / 1000) % 1000;
    uint64_t hz  = center_freq % 1000;

    std::ostringstream fname;
    fname << std::put_time(tm, "%b_%d_%Y_")
        << mhz << "-"
        << std::setw(3) << std::setfill('0') << khz << "-"
        << std::setw(3) << std::setfill('0') << hz
        << "MHz";

    std::string base = fname.str();

    struct bladerf *dev = nullptr;
    int status;
    
    status = bladerf_open(&dev, nullptr);
    if (status != 0) {
        fprintf(stderr, "Failed to open bladeRF: %s\n", bladerf_strerror(status));
        return 1;
    }

    status = bladerf_set_frequency(dev, CHANNEL, center_freq);
    if (status != 0) {
        fprintf(stderr, "Failed to set frequency: %s\n", bladerf_strerror(status));
        bladerf_close(dev);
        return 1;
    }

    uint32_t actual_rate;
    status = bladerf_set_sample_rate(dev, CHANNEL, sample_rate, &actual_rate);
    if (status != 0) {
        fprintf(stderr, "Failed to set sample rate: %s\n", bladerf_strerror(status));
        bladerf_close(dev);
        return 1;
    }

    uint32_t actual_bw;
    status = bladerf_set_bandwidth(dev, CHANNEL, sample_rate, &actual_bw);
    if (status != 0) {
        fprintf(stderr, "Failed to set bandwidth: %s\n", bladerf_strerror(status));
        bladerf_close(dev);
        return 1;
    }

    status = bladerf_set_gain_mode(dev, CHANNEL, BLADERF_GAIN_MANUAL);
    if (status != 0) {
        fprintf(stderr, "Failed to set gain mode: %s\n", bladerf_strerror(status));
        bladerf_close(dev);
        return 1;
    }

    status = bladerf_set_gain(dev, CHANNEL, RX_GAIN);
    if (status != 0) {
        fprintf(stderr, "Failed to set gain: %s\n", bladerf_strerror(status));
        bladerf_close(dev);
        return 1;
    }

    status = bladerf_sync_config(
        dev,
        BLADERF_RX_X1,
        BLADERF_FORMAT_SC16_Q11,
        512,
        16384,
        128,
        3000
    );
    if (status != 0) {
        fprintf(stderr, "Failed to configure sync: %s\n", bladerf_strerror(status));
        bladerf_close(dev);
        return 1;
    }

    status = bladerf_enable_module(dev, CHANNEL, true);
    if (status != 0) {
        fprintf(stderr, "Failed to enable RX: %s\n", bladerf_strerror(status));
        bladerf_close(dev);
        return 1;
    }
    
    usleep(200000);

    printf("\n");
    printf("START RECORDING\n\n");

    size_t total_samples = actual_rate * duration_sec;
    std::vector<int16_t> iq(total_samples * 2);

    size_t received = 0;
    auto start_time = std::chrono::steady_clock::now();
    
    while (received < total_samples) {
        size_t to_read = std::min<size_t>(16384, total_samples - received);
        status = bladerf_sync_rx(
            dev,
            iq.data() + received * 2,
            to_read,
            nullptr,
            3000
        );
        if (status != 0) {
            fprintf(stderr, "\nRX error: %s\n", bladerf_strerror(status));
            break;
        }
        received += to_read;
        
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - start_time).count();
        
        static double last_update = 0.0;
        if (elapsed - last_update >= 0.1 || received >= total_samples) {
            last_update = elapsed;
            double progress = 100.0 * received / total_samples;
            printf("\r[%5.1fs / %ds] Progress: %.1f%% ", 
                   elapsed, duration_sec, progress);
            fflush(stdout);
        }
    }
    
    printf("\n\n");

    bladerf_enable_module(dev, CHANNEL, false);
    bladerf_close(dev);

    WavHeader wav;
    wav.sample_rate = actual_rate;
    wav.byte_rate = actual_rate * 2 * 2;
    wav.block_align = 4;
    wav.data_size = received * 2 * sizeof(int16_t);
    wav.file_size = wav.data_size + sizeof(WavHeader) - 8;

    std::ofstream wav_file(base + ".wav", std::ios::binary);
    wav_file.write(reinterpret_cast<char*>(&wav), sizeof(wav));
    wav_file.write(reinterpret_cast<char*>(iq.data()), wav.data_size);
    wav_file.close();

    std::ofstream data_file(base + ".sigmf-data", std::ios::binary);
    data_file.write(reinterpret_cast<char*>(iq.data()), wav.data_size);
    data_file.close();

    std::ofstream meta_file(base + ".sigmf-meta");
    meta_file << "{\n";
    meta_file << "  \"global\": {\n";
    meta_file << "    \"core:datatype\": \"ci16_le\",\n";
    meta_file << "    \"core:sample_rate\": " << actual_rate << ",\n";
    meta_file << "    \"core:version\": \"1.0.0\"\n";
    meta_file << "  },\n";
    meta_file << "  \"captures\": [\n";
    meta_file << "    {\n";
    meta_file << "      \"core:frequency\": " << center_freq << ",\n";
    meta_file << "      \"core:datetime\": \"" << std::put_time(tm, "%FT%T") << "\"\n";
    meta_file << "    }\n";
    meta_file << "  ]\n";
    meta_file << "}\n";
    meta_file.close();

    printf("Computing FFT with compression...\n");
    
    fftwf_complex *in = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * FFT_SIZE);
    fftwf_complex *out = (fftwf_complex *)fftwf_malloc(sizeof(fftwf_complex) * FFT_SIZE);
    fftwf_plan plan = fftwf_plan_dft_1d(FFT_SIZE, in, out, FFTW_FORWARD, FFTW_MEASURE);

    std::vector<int8_t> fft_results;
    std::vector<float> power_accum(FFT_SIZE, 0.0f);
    uint32_t num_averaged_ffts = 0;
    uint32_t total_ffts_computed = 0;
    
    float min_power_db = 1e10f;
    float max_power_db = -1e10f;

    size_t num_complete_ffts = received / FFT_SIZE;
    printf("Total samples: %zu, FFT size: %d, Number of FFTs: %zu\n", received, FFT_SIZE, num_complete_ffts);
    
    for (size_t fft_idx = 0; fft_idx < num_complete_ffts; fft_idx++) {
        size_t sample_start = fft_idx * FFT_SIZE;
        
        if (sample_start + FFT_SIZE > received) break;
        
        for (int j = 0; j < FFT_SIZE; j++) {
            float re = iq[sample_start * 2 + j * 2] / 2048.0f;
            float im = iq[sample_start * 2 + j * 2 + 1] / 2048.0f;
            in[j][0] = re;
            in[j][1] = im;
        }

        fftwf_execute(plan);

        for (int j = 0; j < FFT_SIZE; j++) {
            float mag_sq = out[j][0] * out[j][0] + out[j][1] * out[j][1];
            float power_db = 10.0f * std::log10(mag_sq + 1e-10f);
            power_accum[j] += power_db;
            
            min_power_db = std::min(min_power_db, power_db);
            max_power_db = std::max(max_power_db, power_db);
        }
        
        num_averaged_ffts++;
        total_ffts_computed++;

        if (num_averaged_ffts == TIME_AVERAGE) {
            for (int j = 0; j < FFT_SIZE; j++) {
                power_accum[j] /= TIME_AVERAGE;
                float normalized = (power_accum[j] - min_power_db) / (max_power_db - min_power_db + 1e-6f);
                normalized = std::max(0.0f, std::min(1.0f, normalized));
                int8_t quantized = static_cast<int8_t>(normalized * 127.0f);
                fft_results.push_back(quantized);
            }
            
            std::fill(power_accum.begin(), power_accum.end(), 0.0f);
            num_averaged_ffts = 0;
        }
    }

    if (num_averaged_ffts > 0) {
        for (int j = 0; j < FFT_SIZE; j++) {
            power_accum[j] /= num_averaged_ffts;
            float normalized = (power_accum[j] - min_power_db) / (max_power_db - min_power_db + 1e-6f);
            normalized = std::max(0.0f, std::min(1.0f, normalized));
            int8_t quantized = static_cast<int8_t>(normalized * 127.0f);
            fft_results.push_back(quantized);
        }
    }

    fftwf_destroy_plan(plan);
    fftwf_free(in);
    fftwf_free(out);

    uint32_t final_num_ffts = fft_results.size() / FFT_SIZE;
    printf("FFT completed: %u computed, %u averaged (TIME_AVERAGE=%d)\n", total_ffts_computed, final_num_ffts, TIME_AVERAGE);
    printf("Power actual range: %.1f ~ %.1f dB\n", min_power_db, max_power_db);

    FFTHeader fft_header;
    fft_header.fft_size = FFT_SIZE;
    fft_header.sample_rate = actual_rate;
    fft_header.center_frequency = center_freq;
    fft_header.num_ffts = final_num_ffts;
    fft_header.time_average = TIME_AVERAGE;
    fft_header.power_min = min_power_db;
    fft_header.power_max = max_power_db;

    std::ofstream fft_file(base + ".fftdata", std::ios::binary);
    fft_file.write(reinterpret_cast<char*>(&fft_header), sizeof(FFTHeader));
    fft_file.write(reinterpret_cast<char*>(fft_results.data()), fft_results.size() * sizeof(int8_t));
    fft_file.close();

    double file_size_mb = wav.data_size / 1024.0 / 1024.0;
    double fft_file_size_mb = (sizeof(FFTHeader) + fft_results.size() * sizeof(int8_t)) / 1024.0 / 1024.0;

    printf("RECORDING SUCCESS\n");
    printf("\n");
    printf("Frequency:    %.1f MHz\n", freq_mhz);
    printf("Sample Rate:  %.2f MSPS\n", actual_rate / 1e6);
    printf("Bandwidth:    %.2f MHz\n", actual_bw / 1e6);
    printf("IQ File Size: %.2f MB\n", file_size_mb);
    printf("FFT File Size: %.2f MB (%.2f%% of IQ)\n", fft_file_size_mb, 100.0 * fft_file_size_mb / file_size_mb);
    printf("Samples:      %zu\n", received);
    printf("Raw FFTs:     %u\n", total_ffts_computed);
    printf("Averaged FFTs: %u (TIME_AVERAGE=%d)\n", final_num_ffts, TIME_AVERAGE);
    printf("Power Range: %.1f ~ %.1f dB\n", min_power_db, max_power_db);
    printf("\n");
    printf("Files saved:\n");
    printf("  - %s.wav\n", base.c_str());
    printf("  - %s.sigmf-data\n", base.c_str());
    printf("  - %s.sigmf-meta\n", base.c_str());
    printf("  - %s.fftdata\n", base.c_str());
    printf("\n");

    return 0;
}