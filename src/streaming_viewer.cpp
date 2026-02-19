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
#include <atomic>
#include <deque>
#include <complex>
#include <alsa/asoundlib.h>

#define RX_GAIN 10
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

// ─────────────────────────────────────────────────────────────────────────
// IQ 복조기  WFM / NFM / AM
//
// 파이프라인:
//   61.44MHz IQ → [NCO 회전] → [IIR 채널 LPF] → [dec256] → 240kHz
//   → [FM 복조 / AM] → [de-emphasis] → [audio LPF] → [dec5] → 48kHz
//   → ring buffer (2초) → PA write thread (pa_simple, blocking)
//
// SDR++ 방식: audio clock이 소비를 주도
//   demod는 IQ 도착 즉시 처리 → ring에 push
//   PA thread는 ring에서 CHUNK 모이면 즉시 write (blocking)
//   → pa_simple_write가 PA 내부 버퍼 꽉 차면 자동 block = rate control
// ─────────────────────────────────────────────────────────────────────────
class IQDemodulator {
public:
    enum class Mode { NONE, AM, WFM, NFM };

    std::atomic<bool>  active{false};
    std::atomic<float> sel_rel_mhz{0.0f};
    std::atomic<float> sel_bw_mhz{0.2f};
    std::atomic<int>   mode{(int)Mode::NONE};
    std::atomic<float> volume{1.0f};

    std::mutex  iq_mutex;
    std::deque<std::vector<std::complex<float>>> iq_queue;
    static constexpr int MAX_QUEUE = 512;

    std::atomic<bool> running{false};
    std::thread       demod_thread;
    float             sample_rate = 61.44e6f;
    static constexpr int AUDIO_RATE = 48000;
    std::atomic<int>  dbg_processed{0};

    void start(float sr) {
        sample_rate = sr;
        running     = true;
        demod_thread = std::thread(&IQDemodulator::demod_loop, this);
    }
    void stop() {
        running = false;
        if (demod_thread.joinable()) demod_thread.join();
    }
    void push_iq(const int16_t* buf, int n) {
        if (!active.load()) return;
        std::vector<std::complex<float>> blk(n);
        for (int i = 0; i < n; i++)
            blk[i] = { buf[i*2] / 32768.f, buf[i*2+1] / 32768.f };
        std::lock_guard<std::mutex> lk(iq_mutex);
        if ((int)iq_queue.size() < MAX_QUEUE)
            iq_queue.push_back(std::move(blk));
    }

private:
    using cf = std::complex<float>;

    // ── 1-pole IIR (real) ─────────────────────────────────────────────────
    // y[n] = (1-a)*x[n] + a*y[n-1]   (simple leaky integrator / lowpass)
    struct Pole1 {
        float a=0.f, y=0.f;
        void  set(float alpha){ a=alpha; }
        float run(float x){ y=(1.f-a)*x+a*y; return y; }
        void  reset(){ y=0.f; }
    };

    // ── 8-stage 1-pole complex LPF  (채널 필터) ───────────────────────────
    // 각 스테이지: y_i[n] = (1-a)*x[n] + a*y_i[n-1]
    struct ChanFilter {
        float a=0.f;
        cf    y[8]={{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0},{0,0}};
        void set(float alpha){ a=alpha; }
        cf   run(cf x){
            float b=1.f-a;
            for(int k=0;k<8;k++){ y[k]=x*b+y[k]*a; x=y[k]; }
            return x;
        }
        void reset(){ for(auto&v:y) v={0,0}; }
    };

    // ── 4-stage 1-pole real LPF  (오디오 필터) ────────────────────────────
    struct AudioFilter {
        float a=0.f, y[4]={0,0,0,0};
        void  set(float alpha){ a=alpha; }
        float run(float x){
            float b=1.f-a;
            for(int k=0;k<4;k++){ y[k]=x*b+y[k]*a; x=y[k]; }
            return x;
        }
        void reset(){ for(auto&v:y) v=0.f; }
    };

    // ── ALSA 헬퍼 (non-blocking) ──────────────────────────────────────────
    static void alsa_safe_write(snd_pcm_t* pcm,
                                const float* data, int n) {
        while (n > 0) {
            int r = snd_pcm_writei(pcm, data, n);
            if      (r == -EAGAIN) { break; }  // 버퍼 가득 참 - 나중에 다시
            else if (r == -EPIPE)  { snd_pcm_prepare(pcm); }
            else if (r < 0)        { snd_pcm_recover(pcm, r, 1); }
            else                   { data += r; n -= r; }
        }
    }

    // ── demod_loop ────────────────────────────────────────────────────────
    void demod_loop() {
        fprintf(stderr,"[DEMOD] started sr=%.0f\n", sample_rate);

        // ALSA open
        snd_pcm_t *pcm = nullptr;
        {
            int err = snd_pcm_open(&pcm,"default",SND_PCM_STREAM_PLAYBACK,SND_PCM_NONBLOCK);
            if (err<0){ fprintf(stderr,"[ALSA] open: %s\n",snd_strerror(err)); return; }
            snd_pcm_hw_params_t *hp=nullptr;
            snd_pcm_hw_params_alloca(&hp);
            snd_pcm_hw_params_any(pcm,hp);
            snd_pcm_hw_params_set_access(pcm,hp,SND_PCM_ACCESS_RW_INTERLEAVED);
            snd_pcm_hw_params_set_format(pcm,hp,SND_PCM_FORMAT_FLOAT_LE);
            snd_pcm_hw_params_set_channels(pcm,hp,1);
            unsigned int r=AUDIO_RATE;
            snd_pcm_hw_params_set_rate_near(pcm,hp,&r,0);
            snd_pcm_uframes_t buf=4096, per=512; // 작은 버퍼, 작은 period
            snd_pcm_hw_params_set_buffer_size_near(pcm,hp,&buf);
            snd_pcm_hw_params_set_period_size_near(pcm,hp,&per,0);
            snd_pcm_hw_params(pcm,hp);
            snd_pcm_prepare(pcm);
            snd_pcm_uframes_t got_buf=buf, got_per=per;
            snd_pcm_get_params(pcm,&got_buf,&got_per);
            fprintf(stderr,"[ALSA] buf=%lu per=%lu (non-blocking mode)\n",got_buf,got_per);
        }

        // DSP state
        ChanFilter  chan_lpf;
        AudioFilter audio_lpf;
        Pole1       deemph;
        float prev_i=0.f, prev_q=0.f, am_dc=0.f;
        int   mode_prev=-1, dec1_prev=0;

        // NCO
        cf    nco(1.f,0.f), nco_step(1.f,0.f);
        float last_rel_hz = 1e30f;

        // IQ accumulator
        std::vector<cf> iq_buf;
        iq_buf.reserve(1<<21);
        int rd=0;

        // audio output buffer (즉시 write, 누적 없음)
        std::vector<float> out_buf;
        out_buf.reserve(256);

        while (running.load()) {
            // drain IQ queue
            {
                std::lock_guard<std::mutex> lk(iq_mutex);
                while (!iq_queue.empty()){
                    auto &b=iq_queue.front();
                    iq_buf.insert(iq_buf.end(),b.begin(),b.end());
                    iq_queue.pop_front();
                }
            }

            int cur_mode = mode.load();
            if (!active.load() || cur_mode==(int)Mode::NONE){
                iq_buf.clear(); rd=0;
                std::this_thread::sleep_for(std::chrono::milliseconds(2));
                continue;
            }

            // ── 파라미터 ─────────────────────────────────────────────────
            float rel_hz = sel_rel_mhz.load()*1e6f;
            float sr     = sample_rate;

            // ── BW 기반 dynamic decimation ────────────────────────────────
            // dec1 후보: sr/(AUDIO_RATE*k) 형태로 qr이 정확히 AUDIO_RATE의 배수
            // 선택 기준: qr >= bw_hz*2.5 (nyquist 마진) 중 최소 qr
            // → 채널 필터 부담 최소화, 정확한 48kHz 출력 보장
            float bw_hz = sel_bw_mhz.load() * 1e6f;

            int   dec1, dec2;
            float qr;
            float chan_alpha, audio_alpha, deemph_alpha, fm_gain;

            if (cur_mode==(int)Mode::AM){
                dec1 = (int)(sr/(float)AUDIO_RATE);  // 1280
                dec2 = 1;
                qr   = sr / dec1;
                bw_hz= std::max(bw_hz, 10e3f);
                chan_alpha   = expf(-2.f*(float)M_PI * (bw_hz*0.5f) / qr);
                audio_alpha  = expf(-2.f*(float)M_PI * 4e3f / (float)AUDIO_RATE);
                deemph_alpha = 0.f;
                fm_gain      = 1.f;
            } else {
                // WFM: 최소 BW=200kHz, NFM: 최소 BW=25kHz
                float min_bw = (cur_mode==(int)Mode::WFM) ? 200e3f : 25e3f;
                bw_hz = std::max(bw_hz, min_bw);
                float need_qr = bw_hz * 2.5f;  // nyquist 마진

                // k=1,2,4,5,8,10,16,20 → dec1=1280,640,320,256,160,128,80,64
                // qr이 need_qr 이상인 최소 qr 선택 (= 최대 dec1)
                int best_dec1=256, best_dec2=5; // 기본값
                const int ks[] = {1,2,4,5,8,10,16,20};
                for (int k : ks){
                    int d1 = (int)(sr/(float)AUDIO_RATE/k);
                    if(d1<1) continue;
                    float q = sr/d1;
                    if(q >= need_qr){
                        best_dec1=d1; best_dec2=k;
                        // k 오름차순이므로 첫 번째 만족이 최대 dec1
                        break;
                    }
                }
                dec1=best_dec1; dec2=best_dec2;
                qr = sr/dec1;

                float dev_hz = (cur_mode==(int)Mode::WFM) ? 75e3f : 5e3f;
                // WFM: 200kHz 고정, NFM: 가변
                float chan_bw = (cur_mode==(int)Mode::WFM) ? 200e3f : std::min(bw_hz*0.5f, 12.5e3f);
                // ✅ Audio filter BW 증가: 15kHz → 17kHz (음성 선명도 개선)
                float aud_bw = (cur_mode==(int)Mode::WFM) ? 17e3f : 5e3f;

                // ✅ 필터 계산: sr(61.44MHz) 기준 (원래 구조)
                chan_alpha   = expf(-2.f*(float)M_PI * chan_bw / sr);
                audio_alpha  = expf(-2.f*(float)M_PI * aud_bw / (float)AUDIO_RATE);
                deemph_alpha = (cur_mode==(int)Mode::WFM)
                               ? (1.f - expf(-2.f * (float)M_PI / (75e-6f * qr))) : 0.f;
                // ✅ FM gain 조정: 음성 선명도 개선
                fm_gain      = (qr / (2.f*(float)M_PI * dev_hz)) * 2.0f;
            }

            // ── 모드 변경 시 리셋 ─────────────────────────────────────────
            if (cur_mode!=mode_prev || dec1!=dec1_prev){
                fprintf(stderr,"[DEMOD] reset mode=%d dec1=%d qr=%.0f\n",cur_mode,dec1,qr);
                chan_lpf.reset(); audio_lpf.reset(); deemph.reset();
                prev_i=prev_q=am_dc=0.f;
                nco=cf(1.f,0.f); last_rel_hz=1e30f;
                iq_buf.clear(); rd=0; out_buf.clear();
                mode_prev=cur_mode; dec1_prev=dec1;
            }

            // 필터 계수 갱신
            chan_lpf.set(chan_alpha);
            audio_lpf.set(audio_alpha);
            deemph.set(deemph_alpha);

            // NCO step (주파수 변경 시만 trig 계산)
            if (fabsf(rel_hz-last_rel_hz)>0.5f){
                float phi = -2.f*(float)M_PI * rel_hz / sr;
                nco_step  = cf(cosf(phi), sinf(phi));
                last_rel_hz = rel_hz;
            }

            // ── 처리량 결정 ───────────────────────────────────────────────
            int avail = (int)iq_buf.size() - rd;
            if (avail < dec1){
                // compact
                if (rd>0){
                    int rem=(int)iq_buf.size()-rd;
                    if(rem>0) memmove(iq_buf.data(),iq_buf.data()+rd,rem*sizeof(cf));
                    iq_buf.resize(rem); rd=0;
                }
                std::this_thread::sleep_for(std::chrono::microseconds(200));
                continue;
            }
            int n_in = std::min(avail,(int)(sr*0.02f)); // 최대 20ms
            n_in = (n_in/dec1)*dec1;
            if (n_in<=0){ std::this_thread::sleep_for(std::chrono::microseconds(200)); continue; }

            int n_q = n_in/dec1; // quadrature 샘플 수

            int proc=dbg_processed.fetch_add(1)+1;
            if(proc%500==1)
                fprintf(stderr,"[DEMOD] blk#%d n_in=%d n_q=%d q=%d\n",
                        proc,n_in,n_q,(int)iq_queue.size());

            // ── Stage 1: NCO shift + Chan LPF + dec1 ────────────────────
            std::vector<cf> ch(n_q);
            for (int i=0;i<n_in;i++){
                // NCO 회전 (incremental complex multiply)
                cf x = iq_buf[rd+i];
                cf s = { x.real()*nco.real() - x.imag()*nco.imag(),
                         x.real()*nco.imag() + x.imag()*nco.real() };
                // NCO 전진
                float nr = nco.real()*nco_step.real() - nco.imag()*nco_step.imag();
                float ni = nco.real()*nco_step.imag() + nco.imag()*nco_step.real();
                nco = {nr,ni};
                // 1024샘플마다 크기 정규화 (float drift 방지)
                if((i&1023)==0){
                    float m=sqrtf(nr*nr+ni*ni);
                    if(m>1e-9f) nco*=(1.f/m);
                }
                // 채널 LPF
                s = chan_lpf.run(s);
                // 정수 dec
                if(i%dec1==0) ch[i/dec1]=s;
            }
            rd += n_in;

            // ── Stage 2: FM Quadrature Demodulation ──────────────────────
            // y[n] = arg( x[n]*conj(x[n-1]) )
            // re = x.r*p.r + x.i*p.i
            // im = x.i*p.r - x.r*p.i
            std::vector<float> mpx(n_q);
            if (cur_mode==(int)Mode::AM){
                for(int i=0;i<n_q;i++){
                    float e=std::abs(ch[i]);
                    am_dc=0.9999f*am_dc+0.0001f*e;
                    mpx[i]=e-am_dc;
                }
            } else {
                float pi_=prev_i, pq_=prev_q;
                for(int i=0;i<n_q;i++){
                    float cr=ch[i].real(), ci=ch[i].imag();
                    float re= cr*pi_ + ci*pq_;   // conj multiply real
                    float im= ci*pi_ - cr*pq_;   // conj multiply imag
                    // atan2 출력: -π ~ +π, FM gain 적용
                    mpx[i] = atan2f(im, re) * fm_gain;
                    pi_=cr; pq_=ci;
                }
                prev_i=ch[n_q-1].real();
                prev_q=ch[n_q-1].imag();
            }

            // ── Stage 3: De-emphasis (WFM) ───────────────────────────────
            // 비활성화: WFM pre-emphasis 보정 필요 없음
            if (deemph_alpha > 0.f && false){
                for(int i=0;i<n_q;i++)
                    mpx[i] = deemph.run(mpx[i]);
            }

            // ── Stage 4: Audio LPF + dec2 → 48kHz ───────────────────────
            float vol = volume.load();
            for(int i=0;i<n_q;i++){
                float v = audio_lpf.run(mpx[i]);
                if(i%dec2==0){
                    v *= vol;
                    if(v > 1.f) v= 1.f;
                    if(v <-1.f) v=-1.f;
                    out_buf.push_back(v);
                }
            }

            // ── Stage 5: ALSA write (실시간 non-blocking) ──────────────────
            // 소량이라도 계속 write - non-blocking이므로 적체되지 않음
            if (!out_buf.empty()){
                alsa_safe_write(pcm, out_buf.data(), (int)out_buf.size());
                out_buf.clear();
            }

            // IQ buf compact
            if(rd>(int)iq_buf.size()/2){
                int rem=(int)iq_buf.size()-rd;
                if(rem>0) memmove(iq_buf.data(),iq_buf.data()+rd,rem*sizeof(cf));
                iq_buf.resize(rem); rd=0;
            }
        }

        snd_pcm_drain(pcm);
        snd_pcm_close(pcm);
        fprintf(stderr,"[ALSA] closed\n");
    }
};

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

    IQDemodulator *demodulator = nullptr;  // run_streaming_viewer에서 설정

    // 복조 채널 선택 UI 상태
    bool   demod_dragging = false;
    float  demod_drag_start_x = 0.0f;   // 화면 픽셀
    float  demod_sel_x0 = -1.0f;        // 화면 픽셀 (좌)
    float  demod_sel_x1 = -1.0f;        // 화면 픽셀 (우)
    float  demod_sel_freq0 = 0.0f;      // 절대 MHz
    float  demod_sel_freq1 = 0.0f;      // 절대 MHz
    
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
        int16_t *demod_buffer = new int16_t[4096 * 2];  // 작은 청크 (약 67µs @ 61.44MHz)
        int demod_buf_idx = 0;
        
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
                    if (demodulator) demodulator->sel_rel_mhz.store(0.0f);
                    printf("Frequency changed to: %.2f MHz\n", pending_center_freq);
                    char title[256];
                    snprintf(title, sizeof(title), "Real-time FFT Viewer - %.2f MHz", pending_center_freq);
                    window_title = title;
                    // 주파수 변경 후 autoscale 재활성
                    autoscale_accum.clear();
                    autoscale_initialized = false;
                    autoscale_active = true;
                    // 주파수 변경 후 demod IQ 큐 플러시 (이전 주파수 오염 데이터 제거)
                    if (demodulator) {
                        std::lock_guard<std::mutex> lk(demodulator->iq_mutex);
                        demodulator->iq_queue.clear();
                    }
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

            // ✅ 복조기에 작은 청크 단위로 전달 (지연 최소화)
            for (int i = 0; i < fft_size; i++) {
                demod_buffer[demod_buf_idx * 2] = iq_buffer[i * 2];
                demod_buffer[demod_buf_idx * 2 + 1] = iq_buffer[i * 2 + 1];
                demod_buf_idx++;
                
                if (demod_buf_idx >= 4096) {
                    if (demodulator) {
                        demodulator->push_iq(demod_buffer, 4096);
                    }
                    demod_buf_idx = 0;
                }
            }
            
            // FFT 계산용 (지연 무관)
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
        delete[] demod_buffer;
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

        // ── 복조 범위 드래그 선택 (오른쪽 버튼) ──────────────────
        {
            float nyquist_l = header.sample_rate / 2.0f / 1e6f;
            float eff_nyq   = nyquist_l * 0.875f;
            float tot_range = 2.0f * eff_nyq;
            float ds        = -eff_nyq + freq_pan * tot_range;
            float de        = ds + tot_range / freq_zoom;

            auto px_to_abs = [&](float px_) -> float {
                float norm = (px_ - graph_x) / graph_w;
                norm = std::max(0.0f, std::min(1.0f, norm));
                return ds + norm * (de - ds) + header.center_frequency / 1e6f;
            };

            ImVec2 mouse = ImGui::GetMousePos();

            if (ImGui::IsItemHovered() && ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
                demod_dragging     = true;
                demod_drag_start_x = mouse.x;
                demod_sel_x0 = mouse.x;
                demod_sel_x1 = mouse.x;
            }
            if (demod_dragging) {
                demod_sel_x1 = mouse.x;
                if (ImGui::IsMouseReleased(ImGuiMouseButton_Right)) {
                    demod_dragging = false;
                    float f0 = px_to_abs(std::min(demod_sel_x0, demod_sel_x1));
                    float f1 = px_to_abs(std::max(demod_sel_x0, demod_sel_x1));
                    demod_sel_freq0 = f0;
                    demod_sel_freq1 = f1;
                    if (demodulator) {
                        float cf = header.center_frequency / 1e6f;
                        demodulator->sel_rel_mhz.store((f0 + f1) / 2.0f - cf);
                        demodulator->sel_bw_mhz.store(std::max(0.01f, f1 - f0));
                    }
                }
            }

            // 선택 영역 반투명 표시
            if (demod_sel_x0 >= 0 && demod_sel_x1 >= 0) {
                float sx0 = std::max(std::min(demod_sel_x0, demod_sel_x1), graph_x);
                float sx1 = std::min(std::max(demod_sel_x0, demod_sel_x1), graph_x + graph_w);
                if (sx1 > sx0) {
                    draw_list->AddRectFilled(ImVec2(sx0, graph_y), ImVec2(sx1, graph_y + graph_h),
                                             IM_COL32(255, 200, 0, 40));
                    draw_list->AddRect(ImVec2(sx0, graph_y), ImVec2(sx1, graph_y + graph_h),
                                       IM_COL32(255, 200, 0, 180));
                    if (demod_sel_freq1 > demod_sel_freq0) {
                        char bw_label[32];
                        float bw = demod_sel_freq1 - demod_sel_freq0;
                        if (bw < 1.0f) snprintf(bw_label, sizeof(bw_label), "%.0f kHz", bw * 1000.0f);
                        else           snprintf(bw_label, sizeof(bw_label), "%.3f MHz", bw);
                        ImVec2 ts = ImGui::CalcTextSize(bw_label);
                        draw_list->AddText(ImVec2((sx0+sx1)/2.0f - ts.x/2.0f, graph_y + 4),
                                           IM_COL32(255, 220, 0, 255), bw_label);
                    }
                }
            }
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

    IQDemodulator demodulator;
    demodulator.start(sample_rate * 1e6f);
    viewer.demodulator = &demodulator;

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

            // ── 복조 패널 ──────────────────────────────────────────
            if (viewer.demodulator) {
                ImGui::SameLine(); ImGui::Spacing(); ImGui::SameLine();
                ImGui::Separator(); ImGui::SameLine();

                IQDemodulator *dm = viewer.demodulator;
                int cur_mode = dm->mode.load();
                bool is_active = dm->active.load();

                // AM 버튼
                // AM 버튼
                bool am_on = is_active && cur_mode == (int)IQDemodulator::Mode::AM;
                if (am_on) ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.7f, 0.2f, 1.0f));
                if (ImGui::Button("AM")) {
                    if (am_on) { dm->active.store(false); dm->mode.store((int)IQDemodulator::Mode::NONE); }
                    else        { dm->mode.store((int)IQDemodulator::Mode::AM); dm->active.store(true); }
                }
                if (am_on) ImGui::PopStyleColor();

                ImGui::SameLine();

                // WFM 버튼 (광대역 FM, 최대 200kHz)
                bool wfm_on = is_active && cur_mode == (int)IQDemodulator::Mode::WFM;
                if (wfm_on) ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.2f, 0.5f, 0.9f, 1.0f));
                if (ImGui::Button("WFM")) {
                    if (wfm_on) { dm->active.store(false); dm->mode.store((int)IQDemodulator::Mode::NONE); }
                    else         { dm->mode.store((int)IQDemodulator::Mode::WFM); dm->active.store(true); }
                }
                if (wfm_on) ImGui::PopStyleColor();

                ImGui::SameLine();

                // NFM 버튼 (협대역 FM, 최대 25kHz)
                bool nfm_on = is_active && cur_mode == (int)IQDemodulator::Mode::NFM;
                if (nfm_on) ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.7f, 0.4f, 0.9f, 1.0f));
                if (ImGui::Button("NFM")) {
                    if (nfm_on) { dm->active.store(false); dm->mode.store((int)IQDemodulator::Mode::NONE); }
                    else         { dm->mode.store((int)IQDemodulator::Mode::NFM); dm->active.store(true); }
                }
                if (nfm_on) ImGui::PopStyleColor();

                ImGui::SameLine(); ImGui::Spacing(); ImGui::SameLine();

                // 볼륨 슬라이더
                ImGui::Text("Vol:"); ImGui::SameLine();
                ImGui::SetNextItemWidth(80);
                float vol = dm->volume.load();
                if (ImGui::SliderFloat("##vol", &vol, 0.0f, 4.0f, "%.1f"))
                    dm->volume.store(vol);

                // 현재 선택 주파수 표시
                if (is_active && viewer.demod_sel_freq1 > viewer.demod_sel_freq0) {
                    ImGui::SameLine(); ImGui::Spacing(); ImGui::SameLine();
                    char sel_info[64];
                    float bw = viewer.demod_sel_freq1 - viewer.demod_sel_freq0;
                    float cf = (viewer.demod_sel_freq0 + viewer.demod_sel_freq1) / 2.0f;
                    if (bw < 1.0f)
                        snprintf(sel_info, sizeof(sel_info), "%.3f MHz / BW %.0f kHz", cf, bw*1000.0f);
                    else
                        snprintf(sel_info, sizeof(sel_info), "%.3f MHz / BW %.3f MHz", cf, bw);
                    ImGui::TextColored(ImVec4(1.0f, 0.85f, 0.0f, 1.0f), "%s", sel_info);
                } else if (!is_active) {
                    ImGui::SameLine(); ImGui::Spacing(); ImGui::SameLine();
                    ImGui::TextDisabled("Right-drag spectrum to select");
                }
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
    demodulator.stop();

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

// BUILD VERSION
// v5.6
static const char* BUILD_VERSION = "v5.6";
static void __attribute__((constructor)) print_version() {
    fprintf(stderr, "[VERSION] streaming_viewer %s\n", BUILD_VERSION);
}