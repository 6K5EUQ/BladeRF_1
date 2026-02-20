#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <libbladeRF.h>
#include <fftw3.h>
#include <alsa/asoundlib.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <string>
#include <chrono>
#include <thread>
#include <mutex>
#include <atomic>
#include <time.h>

// ─────────────────────────────────────────────────────────────────────────────
#define RX_GAIN                10
#define CHANNEL                BLADERF_CHANNEL_RX(0)
#define DEFAULT_FFT_SIZE       8192
#define TIME_AVERAGE           50
#define MAX_FFTS_MEMORY        1000
#define HANN_WINDOW_CORRECTION 2.67f
#define AXIS_LABEL_WIDTH       50
#define BOTTOM_LABEL_HEIGHT    30
#define TOPBAR_H               32.0f
#define IQ_RING_CAPACITY       (1 << 22)
#define IQ_RING_MASK           (IQ_RING_CAPACITY - 1)
#define AUDIO_SR               48000u
#define AUDIO_DEVICE           "default"
#define MAX_CHANNELS           5

// Per-channel colors: border, fill, selected-fill
static const ImU32 CH_BORD[MAX_CHANNELS]={
    IM_COL32(255,220, 50,220), IM_COL32( 50,200,255,220),
    IM_COL32(255, 90, 50,220), IM_COL32(180, 60,255,220),
    IM_COL32( 50,255,110,220)};
static const ImU32 CH_FILL[MAX_CHANNELS]={
    IM_COL32(255,220, 50, 30), IM_COL32( 50,200,255, 30),
    IM_COL32(255, 90, 50, 30), IM_COL32(180, 60,255, 30),
    IM_COL32( 50,255,110, 30)};
static const ImU32 CH_SFIL[MAX_CHANNELS]={
    IM_COL32(255,220, 50, 75), IM_COL32( 50,200,255, 75),
    IM_COL32(255, 90, 50, 75), IM_COL32(180, 60,255, 75),
    IM_COL32( 50,255,110, 75)};

// ─────────────────────────────────────────────────────────────────────────────
struct FFTHeader {
    char magic[4]; uint32_t version,fft_size,sample_rate;
    uint64_t center_frequency; uint32_t num_ffts,time_average;
    float power_min,power_max,reserved[8];
};

// ─────────────────────────────────────────────────────────────────────────────
struct WAVWriter {
    FILE* fp=nullptr; uint32_t sample_rate=0; uint64_t num_samples=0;
    std::vector<int16_t> buf;
    static constexpr size_t BUF_FRAMES=65536;
    bool open(const std::string& fn,uint32_t sr){
        fp=fopen(fn.c_str(),"wb"); if(!fp) return false;
        sample_rate=sr; num_samples=0; buf.reserve(BUF_FRAMES*2); write_hdr(); return true;
    }
    void push(int16_t i,int16_t q){
        buf.push_back(i); buf.push_back(q); ++num_samples;
        if(buf.size()>=BUF_FRAMES*2) flush();
    }
    void flush(){ if(!fp||buf.empty()) return; fwrite(buf.data(),2,buf.size(),fp); buf.clear(); }
    void close(){ flush(); if(!fp) return; fseek(fp,0,SEEK_SET); write_hdr(); fclose(fp); fp=nullptr; }
private:
    void write_hdr(){
        auto w32=[&](uint32_t v){fwrite(&v,4,1,fp);}; auto w16=[&](uint16_t v){fwrite(&v,2,1,fp);};
        uint32_t db=(uint32_t)(num_samples*4);
        fwrite("RIFF",1,4,fp); w32(36+db); fwrite("WAVE",1,4,fp);
        fwrite("fmt ",1,4,fp); w32(16); w16(1); w16(2); w32(sample_rate); w32(sample_rate*4); w16(4); w16(16);
        fwrite("data",1,4,fp); w32(db);
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Stereo ALSA output
struct AlsaOut {
    snd_pcm_t* pcm=nullptr;
    bool open(uint32_t sr=AUDIO_SR){
        int err=snd_pcm_open(&pcm,AUDIO_DEVICE,SND_PCM_STREAM_PLAYBACK,0);
        if(err<0){fprintf(stderr,"ALSA open: %s\n",snd_strerror(err));return false;}
        snd_pcm_hw_params_t* hw; snd_pcm_hw_params_alloca(&hw);
        snd_pcm_hw_params_any(pcm,hw);
        snd_pcm_hw_params_set_access(pcm,hw,SND_PCM_ACCESS_RW_INTERLEAVED);
        snd_pcm_hw_params_set_format(pcm,hw,SND_PCM_FORMAT_S16_LE);
        snd_pcm_hw_params_set_channels(pcm,hw,2);  // stereo
        unsigned rsr=sr; snd_pcm_hw_params_set_rate_near(pcm,hw,&rsr,0);
        snd_pcm_uframes_t buf_sz=8192,per_sz=256;
        snd_pcm_hw_params_set_buffer_size_near(pcm,hw,&buf_sz);
        snd_pcm_hw_params_set_period_size_near(pcm,hw,&per_sz,0);
        err=snd_pcm_hw_params(pcm,hw);
        if(err<0){fprintf(stderr,"ALSA hw: %s\n",snd_strerror(err));snd_pcm_close(pcm);pcm=nullptr;return false;}
        snd_pcm_sw_params_t* sw; snd_pcm_sw_params_alloca(&sw);
        snd_pcm_sw_params_current(pcm,sw);
        snd_pcm_sw_params_set_start_threshold(pcm,sw,256);
        snd_pcm_sw_params_set_avail_min(pcm,sw,256);
        snd_pcm_sw_params(pcm,sw);
        printf("ALSA: %u Hz stereo\n",rsr); return true;
    }
    // buf = interleaved L,R,L,R,... int16 pairs; frames = number of stereo frames
    void write(const int16_t* buf,int frames){
        if(!pcm) return;
        while(frames>0){
            snd_pcm_sframes_t r=snd_pcm_writei(pcm,buf,frames);
            if(r<0){r=snd_pcm_recover(pcm,(int)r,0);if(r<0){fprintf(stderr,"ALSA wr: %s\n",snd_strerror((int)r));return;}continue;}
            buf+=r*2; frames-=(int)r;
        }
    }
    void close(){if(pcm){snd_pcm_drain(pcm);snd_pcm_close(pcm);pcm=nullptr;}}
};

// ─────────────────────────────────────────────────────────────────────────────
struct Oscillator {
    float re=1,im=0,dre=1,dim=0; int cnt=0;
    static constexpr int NORM=4096;
    void set_freq(double freq_hz,double sr){
        double w=-2.0*M_PI*freq_hz/sr; dre=(float)cos(w); dim=(float)sin(w); re=1; im=0; cnt=0;
    }
    inline void mix(float si,float sq,float& mi,float& mq){
        mi=si*re-sq*im; mq=si*im+sq*re;
        float nr=re*dre-im*dim,ni=re*dim+im*dre; re=nr; im=ni;
        if(++cnt>=NORM){float m=1.0f/sqrtf(re*re+im*im+1e-30f);re*=m;im*=m;cnt=0;}
    }
};

struct IIR1 {
    float a=0,b=1,s=0;
    void set(double cn){a=(float)exp(-2.0*M_PI*cn);b=1-a;}
    inline float p(float x){s=a*s+b*x;return s;}
};

static void apply_hann(fftwf_complex* in,int n){
    for(int i=0;i<n;i++){float w=0.5f*(1-cosf(2*M_PI*i/(n-1)));in[i][0]*=w;in[i][1]*=w;}
}

static uint32_t optimal_iq_sr(uint32_t main_sr,float bw_hz){
    float target=bw_hz*2.8f; if(target<10000) target=10000;
    uint32_t decim=(uint32_t)(main_sr/target); if(decim<1) decim=1;
    return main_sr/decim;
}

static void demod_rates(uint32_t main_sr,float bw_hz,
                        uint32_t& inter_sr,uint32_t& audio_decim,uint32_t& cap_decim){
    float min_inter=bw_hz*3.0f;
    if(min_inter<(float)AUDIO_SR) min_inter=(float)AUDIO_SR;
    uint32_t ad=(uint32_t)ceilf(min_inter/AUDIO_SR); if(ad<1) ad=1;
    uint32_t isr=AUDIO_SR*ad; uint32_t cd=main_sr/isr; if(cd<1) cd=1;
    inter_sr=isr; audio_decim=ad; cap_decim=cd;
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-channel state
struct Channel {
    // Filter geometry (absolute MHz)
    float s=0,e=0;
    bool  filter_active=false;
    bool  selected=false;

    // Demod
    enum DemodMode{DM_NONE=0,DM_AM,DM_FM} mode=DM_NONE;
    int  pan=0;   // -1=L  0=both  1=R

    // Demod thread
    std::atomic<bool>   dem_run{false};
    std::atomic<bool>   dem_stop_req{false};
    std::thread         dem_thr;
    std::atomic<size_t> dem_rp{0};

    // Per-channel audio ring (float mono, written by dem worker, read by mix thread)
    static constexpr size_t AR_SZ  = 16384;
    static constexpr size_t AR_MASK= AR_SZ-1;
    float ar[AR_SZ]={};
    std::atomic<size_t> ar_wp{0};
    std::atomic<size_t> ar_rp{0};

    void push_audio(float v){
        size_t w=ar_wp.load(std::memory_order_relaxed);
        ar[w&AR_MASK]=v;
        ar_wp.store(w+1,std::memory_order_release);
    }
    bool pop_audio(float& v){
        size_t r=ar_rp.load(std::memory_order_relaxed);
        size_t w=ar_wp.load(std::memory_order_acquire);
        if(r==w) return false;
        v=ar[r&AR_MASK];
        ar_rp.store(r+1,std::memory_order_release);
        return true;
    }
    size_t audio_avail(){
        return ar_wp.load(std::memory_order_acquire)-ar_rp.load(std::memory_order_relaxed);
    }

    // Squelch: 0=off, 1-10 (1=auto Lv1 default)
    std::atomic<float> sq_threshold{-50.0f}; // dBFS threshold, default -50
    // sq_sig = current signal dBFS (for UI bar), sq_gate = gate state
    std::atomic<float> sq_sig{-120.0f}, sq_nf{0.0f}; // sq_nf unused now
    std::atomic<bool>  sq_gate{false};

    // Filter move-drag state
    bool  move_drag=false;
    float move_anchor=0;   // abs MHz at drag start
    float move_s0=0,move_e0=0;

    // Non-copyable/movable (has atomic + thread)
    Channel()=default;
    Channel(const Channel&)=delete;
    Channel& operator=(const Channel&)=delete;
};

// ─────────────────────────────────────────────────────────────────────────────
class FFTViewer {
public:
    FFTHeader  header;
    std::vector<int8_t>  fft_data;
    GLuint               waterfall_texture=0;
    std::vector<uint32_t> wf_row_buf;   // reused per update_wf_row call

    int   fft_size=DEFAULT_FFT_SIZE, time_average=TIME_AVERAGE;
    bool  fft_size_change_req=false; int pending_fft_size=DEFAULT_FFT_SIZE;
    bool  texture_needs_recreate=false;
    int   current_fft_idx=0, last_wf_update_idx=-1;
    float freq_zoom=1,freq_pan=0;
    float display_power_min=-80,display_power_max=0;
    float spectrum_height_ratio=0.2f;
    std::vector<float> current_spectrum;
    int   cached_sp_idx=-1; float cached_pan=-999,cached_zoom=-999;
    int   cached_px=-1; float cached_pmin=-999,cached_pmax=-999;
    std::vector<float> autoscale_accum;
    std::chrono::steady_clock::time_point autoscale_last;
    bool autoscale_init=false,autoscale_active=true;

    struct bladerf* dev=nullptr;
    fftwf_plan fft_plan=nullptr;
    fftwf_complex *fft_in=nullptr,*fft_out=nullptr;
    bool is_running=true; int total_ffts=0;
    std::string window_title;
    std::mutex  data_mtx;

    float pending_cf=0; bool freq_req=false,freq_prog=false;

    // Shared IQ ring
    std::vector<int16_t> ring;
    std::atomic<size_t>  ring_wp{0};

    // Channels
    Channel channels[MAX_CHANNELS];
    int     selected_ch=-1;
    bool    topbar_sel_this_frame=false; // prevent spectrum from overriding topbar selection

    // New-channel drag (right-drag creates channel)
    struct NewDrag{ bool active=false; float anch=0,s=0,e=0; } new_drag;

    // IQ Recording (single-channel only)
    std::atomic<bool>     rec_on{false},rec_stop{false};
    std::thread           rec_thr;
    std::atomic<size_t>   rec_rp{0};
    float                 rec_cf_mhz=0; uint32_t rec_sr=0;
    int                   rec_ch=-1; // channel index being recorded
    std::string           rec_filename;
    std::atomic<uint64_t> rec_frames{0};
    std::chrono::steady_clock::time_point rec_t0;

    // Stereo mix thread
    std::atomic<bool> mix_stop{false};
    std::thread       mix_thr;

    // ── initialize_bladerf ────────────────────────────────────────────────
    bool initialize_bladerf(float cf_mhz,float sr_msps){
        int s=bladerf_open(&dev,nullptr);
        if(s){fprintf(stderr,"bladerf_open: %s\n",bladerf_strerror(s));return false;}
        s=bladerf_set_frequency(dev,CHANNEL,(uint64_t)(cf_mhz*1e6));
        if(s){fprintf(stderr,"set_freq: %s\n",bladerf_strerror(s));bladerf_close(dev);return false;}
        uint32_t actual=0;
        s=bladerf_set_sample_rate(dev,CHANNEL,(uint32_t)(sr_msps*1e6),&actual);
        if(s){fprintf(stderr,"set_sr: %s\n",bladerf_strerror(s));bladerf_close(dev);return false;}
        s=bladerf_set_bandwidth(dev,CHANNEL,(uint32_t)(sr_msps*1e6*0.8f),nullptr);
        if(s){fprintf(stderr,"set_bw: %s\n",bladerf_strerror(s));bladerf_close(dev);return false;}
        s=bladerf_set_gain(dev,CHANNEL,RX_GAIN);
        if(s){fprintf(stderr,"set_gain: %s\n",bladerf_strerror(s));bladerf_close(dev);return false;}
        s=bladerf_enable_module(dev,CHANNEL,true);
        if(s){fprintf(stderr,"enable: %s\n",bladerf_strerror(s));bladerf_close(dev);return false;}
        s=bladerf_sync_config(dev,BLADERF_RX_X1,BLADERF_FORMAT_SC16_Q11,512,16384,128,10000);
        if(s){fprintf(stderr,"sync: %s\n",bladerf_strerror(s));bladerf_close(dev);return false;}
        printf("BladeRF: %.2f MHz  %.2f MSPS\n",cf_mhz,actual/1e6f);
        std::memcpy(header.magic,"FFTD",4);
        header.version=1; header.fft_size=fft_size; header.sample_rate=actual;
        header.center_frequency=(uint64_t)(cf_mhz*1e6);
        header.time_average=TIME_AVERAGE; header.power_min=-80; header.power_max=-30; header.num_ffts=0;
        fft_data.resize(MAX_FFTS_MEMORY*fft_size);
        current_spectrum.resize(fft_size,-80.0f);
        char title[256]; snprintf(title,256,"Real-time FFT Viewer - %.2f MHz",cf_mhz);
        window_title=title; display_power_min=-80; display_power_max=0;
        fft_in=fftwf_alloc_complex(fft_size); fft_out=fftwf_alloc_complex(fft_size);
        fft_plan=fftwf_plan_dft_1d(fft_size,fft_in,fft_out,FFTW_FORWARD,FFTW_MEASURE);
        ring.resize(IQ_RING_CAPACITY*2,0);
        return true;
    }

    // ── Waterfall ─────────────────────────────────────────────────────────
    void create_waterfall_texture(){
        if(waterfall_texture) glDeleteTextures(1,&waterfall_texture);
        glGenTextures(1,&waterfall_texture);
        glBindTexture(GL_TEXTURE_2D,waterfall_texture);
        std::vector<uint32_t> init(fft_size*MAX_FFTS_MEMORY,IM_COL32(0,0,0,255));
        glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,fft_size,MAX_FFTS_MEMORY,0,GL_RGBA,GL_UNSIGNED_BYTE,init.data());
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_REPEAT);
        glBindTexture(GL_TEXTURE_2D,0);
    }
    void update_wf_row(int fi){
        if(!waterfall_texture) return;
        int mi=fi%MAX_FFTS_MEMORY;
        int8_t* row=fft_data.data()+mi*fft_size;
        if((int)wf_row_buf.size()!=fft_size) wf_row_buf.resize(fft_size);
        auto jet=[](float v)->uint32_t{
            float r,g,b;
            if(v<0.04f){r=0;g=0;b=v*12.5f;}
            else if(v<0.15f){float t=(v-0.04f)/0.11f;r=0;g=0;b=0.5f+t*0.5f;}
            else if(v<0.35f){float t=(v-0.15f)*5.0f;r=0;g=t;b=1-t*0.3f;}
            else if(v<0.55f){float t=(v-0.35f)*5.0f;r=t;g=1;b=0;}
            else if(v<0.75f){float t=(v-0.55f)*5.0f;r=1;g=1-t*0.5f;b=0;}
            else if(v<0.95f){float t=(v-0.75f)*5.0f;r=1;g=0.5f-t*0.5f;b=0;}
            else{float t=(v-0.95f)*20.0f;r=1;g=t*0.3f;b=t*0.3f;}
            return IM_COL32((uint8_t)(r*255),(uint8_t)(g*255),(uint8_t)(b*255),255);
        };
        float wmin=display_power_min;
        float wrng_inv=1.0f/std::max(1.0f,display_power_max-wmin);
        float pscale=(header.power_max-header.power_min)/127.0f;
        float pbase=header.power_min;
        int half=fft_size/2;
        auto norm=[&](int bin)->float{
            float p=row[bin]*pscale+pbase;
            float v=(p-wmin)*wrng_inv;
            return v<0.0f?0.0f:v>1.0f?1.0f:v;
        };
        for(int i=0;i<half;i++) wf_row_buf[i]=jet(norm(half+1+i));
        wf_row_buf[half]=jet(norm(0));
        for(int i=1;i<=half;i++) wf_row_buf[half+i]=jet(norm(i));
        glBindTexture(GL_TEXTURE_2D,waterfall_texture);
        glTexSubImage2D(GL_TEXTURE_2D,0,0,mi,fft_size,1,GL_RGBA,GL_UNSIGNED_BYTE,wf_row_buf.data());
        glBindTexture(GL_TEXTURE_2D,0);
    }

    // ── capture_and_process ───────────────────────────────────────────────
    void capture_and_process(){
        int16_t* iq=new int16_t[fft_size*2];
        std::vector<float> pacc(fft_size,0.0f); int fcnt=0;
        while(is_running){
            if(fft_size_change_req){
                fft_size_change_req=false; int ns=pending_fft_size;
                fftwf_destroy_plan(fft_plan); fftwf_free(fft_in); fftwf_free(fft_out);
                fft_size=ns; time_average=std::max(1,TIME_AVERAGE*DEFAULT_FFT_SIZE/ns);
                fft_in=fftwf_alloc_complex(fft_size); fft_out=fftwf_alloc_complex(fft_size);
                fft_plan=fftwf_plan_dft_1d(fft_size,fft_in,fft_out,FFTW_FORWARD,FFTW_MEASURE);
                delete[] iq; iq=new int16_t[fft_size*2];
                pacc.assign(fft_size,0.0f); fcnt=0;
                {std::lock_guard<std::mutex> lk(data_mtx);
                 header.fft_size=fft_size; fft_data.assign(MAX_FFTS_MEMORY*fft_size,0);
                 current_spectrum.assign(fft_size,-80.0f);
                 total_ffts=0; current_fft_idx=0; cached_sp_idx=-1;
                 autoscale_accum.clear(); autoscale_init=false; autoscale_active=true;}
                texture_needs_recreate=true; continue;
            }
            if(freq_req&&!freq_prog){
                freq_prog=true;
                int s=bladerf_set_frequency(dev,CHANNEL,(uint64_t)(pending_cf*1e6));
                if(!s){
                    {std::lock_guard<std::mutex> lk(data_mtx);header.center_frequency=(uint64_t)(pending_cf*1e6);}
                    printf("Freq → %.2f MHz\n",pending_cf);
                    char t[256]; snprintf(t,256,"Real-time FFT Viewer - %.2f MHz",pending_cf); window_title=t;
                    autoscale_accum.clear(); autoscale_init=false; autoscale_active=true;
                }
                freq_req=false; freq_prog=false;
            }
            int status=bladerf_sync_rx(dev,iq,fft_size,nullptr,10000);
            if(status){fprintf(stderr,"RX: %s\n",bladerf_strerror(status));continue;}

            // Ring needed if any demod or recorder running
            bool need_ring=rec_on.load(std::memory_order_relaxed);
            if(!need_ring) for(int i=0;i<MAX_CHANNELS;i++) if(channels[i].dem_run.load()){need_ring=true;break;}
            if(need_ring){
                size_t wp=ring_wp.load(std::memory_order_relaxed);
                size_t n=(size_t)fft_size,cap=IQ_RING_CAPACITY;
                if(wp+n<=cap) memcpy(&ring[wp*2],iq,n*2*sizeof(int16_t));
                else{size_t p1=cap-wp,p2=n-p1;memcpy(&ring[wp*2],iq,p1*2*sizeof(int16_t));memcpy(&ring[0],iq+p1*2,p2*2*sizeof(int16_t));}
                ring_wp.store((wp+n)&IQ_RING_MASK,std::memory_order_release);
            }
            for(int i=0;i<fft_size;i++){fft_in[i][0]=iq[i*2]/2048.0f;fft_in[i][1]=iq[i*2+1]/2048.0f;}
            apply_hann(fft_in,fft_size); fftwf_execute(fft_plan);
            {
                const float scale=HANN_WINDOW_CORRECTION/((float)fft_size*(float)fft_size);
                for(int i=0;i<fft_size;i++){
                    float ms=(fft_out[i][0]*fft_out[i][0]+fft_out[i][1]*fft_out[i][1])*scale+1e-10f;
                    pacc[i]+=10.0f*log10f(ms);
                }
            }
            pacc[0]=(pacc[1]+pacc[fft_size-1])*0.5f; fcnt++;
            if(fcnt>=time_average){
                int fi=total_ffts%MAX_FFTS_MEMORY; int8_t* rowp=fft_data.data()+fi*fft_size;
                {std::lock_guard<std::mutex> lk(data_mtx);
                 for(int i=0;i<fft_size;i++){
                     float avg=pacc[i]/fcnt;
                     float nn=(avg-header.power_min)/(header.power_max-header.power_min);
                     rowp[i]=(int8_t)(std::max(-1.0f,std::min(1.0f,nn))*127);
                     current_spectrum[i]=avg;
                 }
                 if(autoscale_active){
                     if(!autoscale_init){autoscale_accum.reserve(fft_size*200);autoscale_last=std::chrono::steady_clock::now();autoscale_init=true;}
                     for(int i=1;i<fft_size;i++) autoscale_accum.push_back(current_spectrum[i]);
                     float el=std::chrono::duration<float>(std::chrono::steady_clock::now()-autoscale_last).count();
                     if(el>=1.0f&&!autoscale_accum.empty()){
                         size_t idx=(size_t)(autoscale_accum.size()*0.15f);
                         std::nth_element(autoscale_accum.begin(),autoscale_accum.begin()+idx,autoscale_accum.end());
                         display_power_min=autoscale_accum[idx]-10.0f;
                         autoscale_accum.clear(); autoscale_active=false; cached_sp_idx=-1;
                     }
                 }
                 total_ffts++; current_fft_idx=total_ffts-1;
                 header.num_ffts=std::min(total_ffts,MAX_FFTS_MEMORY); cached_sp_idx=-1;}
                std::fill(pacc.begin(),pacc.end(),0.0f); fcnt=0;
            }
        }
        delete[] iq;
    }

    // ── IQ Recording (single channel only) ───────────────────────────────
    void rec_worker(){
        uint32_t msr=header.sample_rate;
        float off=(rec_cf_mhz-(float)(header.center_frequency/1e6f))*1e6f;
        uint32_t decim=std::max(1u,msr/rec_sr), actual_sr=msr/decim;
        WAVWriter wav;
        if(!wav.open(rec_filename,actual_sr)){rec_on.store(false);return;}
        printf("REC: %.4f MHz  off=%.0fHz  decim=%u  SR=%u\n",rec_cf_mhz,off,decim,actual_sr);
        Oscillator osc; osc.set_freq((double)off,(double)msr);
        double ai=0,aq=0; int cnt=0;
        auto c16=[](float v)->int16_t{return (int16_t)(std::max(-1.0f,std::min(1.0f,v))*32767.0f);};
        while(!rec_stop.load(std::memory_order_relaxed)){
            size_t wp=ring_wp.load(std::memory_order_acquire);
            size_t rp=rec_rp.load(std::memory_order_relaxed);
            if(rp==wp){std::this_thread::sleep_for(std::chrono::microseconds(100));continue;}
            size_t avail=std::min((wp-rp)&IQ_RING_MASK,(size_t)65536);
            for(size_t s=0;s<avail;s++){
                size_t pos=(rp+s)&IQ_RING_MASK;
                float si=ring[pos*2]/2048.0f,sq=ring[pos*2+1]/2048.0f;
                float mi,mq; osc.mix(si,sq,mi,mq);
                ai+=mi; aq+=mq; cnt++;
                if(cnt>=(int)decim){wav.push(c16((float)(ai/cnt)),c16((float)(aq/cnt)));rec_frames.fetch_add(1);ai=aq=0;cnt=0;}
            }
            rec_rp.store((rp+avail)&IQ_RING_MASK,std::memory_order_release);
        }
        wav.close(); printf("REC done: %llu frames → %s\n",(unsigned long long)rec_frames.load(),rec_filename.c_str());
    }

    void start_rec(){
        if(rec_on.load()) return;
        // Record the currently selected channel
        int fi=selected_ch;
        if(fi<0||!channels[fi].filter_active){
            printf("REC: no active channel selected\n"); return;
        }
        Channel& ch=channels[fi];
        float ss=std::min(ch.s,ch.e),se=std::max(ch.s,ch.e);
        rec_cf_mhz=(ss+se)/2.0f; float bw_hz=(se-ss)*1e6f;
        rec_sr=optimal_iq_sr(header.sample_rate,bw_hz);
        time_t t=time(nullptr); struct tm tm2; localtime_r(&t,&tm2);
        char fn[256]; snprintf(fn,256,"iq_%.4fMHz_BW%.0fkHz_%04d%02d%02d_%02d%02d%02d.wav",
                 rec_cf_mhz,bw_hz/1000.0f,tm2.tm_year+1900,tm2.tm_mon+1,tm2.tm_mday,tm2.tm_hour,tm2.tm_min,tm2.tm_sec);
        rec_filename=fn; rec_frames.store(0); rec_rp.store(ring_wp.load());
        rec_ch=fi;
        rec_stop.store(false); rec_on.store(true);
        rec_t0=std::chrono::steady_clock::now();
        rec_thr=std::thread(&FFTViewer::rec_worker,this);
        printf("REC start ch%d → %s  SR=%u\n",fi,fn,rec_sr);
    }
    void stop_rec(){
        if(!rec_on.load()) return;
        rec_stop.store(true); if(rec_thr.joinable()) rec_thr.join(); rec_on.store(false);
    }

    // ── Demod worker (per channel) ────────────────────────────────────────
    void dem_worker(int ch_idx){
        Channel& ch=channels[ch_idx];
        Channel::DemodMode mode=ch.mode;
        uint32_t msr=header.sample_rate;
        float off_hz=(((ch.s+ch.e)/2.0f)-(float)(header.center_frequency/1e6f))*1e6f;
        float bw_hz=fabsf(ch.e-ch.s)*1e6f;
        uint32_t inter_sr,audio_decim,cap_decim;
        demod_rates(msr,bw_hz,inter_sr,audio_decim,cap_decim);
        uint32_t actual_inter=msr/cap_decim;
        uint32_t actual_ad=std::max(1u,(uint32_t)round((double)actual_inter/AUDIO_SR));
        uint32_t actual_asr=actual_inter/actual_ad;
        printf("DEM[%d]: mode=%d  cf=%.4fMHz  off=%.0fHz  cap_dec=%u  asr=%u\n",
               ch_idx,(int)mode,(ch.s+ch.e)/2.0f,off_hz,cap_decim,actual_asr);

        Oscillator osc; osc.set_freq((double)off_hz,(double)msr);
        double cap_i=0,cap_q=0; int cap_cnt=0;
        IIR1 lpi,lpq;
        { float cn=(bw_hz*0.5f)/(float)actual_inter; if(cn>0.45f)cn=0.45f; lpi.set(cn); lpq.set(cn); }
        float prev_i=0,prev_q=0,am_dc=0;
        IIR1 alf; alf.set(8000.0/actual_inter);
        double aac=0; int acnt=0;

        // ── Squelch ───────────────────────────────────────────────────────
        // sql_avg: EMA-smoothed signal level (dBFS)
        // Auto-calibration: measure noise floor for 500ms at start,
        // then set threshold = noise_floor + 10dB (user can adjust after)
        const float SQL_ALPHA     = 0.05f; // EMA ~20 samples
        const int   SQL_HOLD_SAMP = 0;     // 0ms hold: instant close
        const int   CALIB_SAMP    = (int)(actual_inter * 0.500f);
        float sql_avg    = -120.0f;
        std::vector<float> calib_buf;
        calib_buf.reserve(CALIB_SAMP);
        bool  calibrated = false;
        bool  gate_open  = false;
        int   gate_hold  = 0;
        int   sq_ui_tick = 0;

        const size_t MAX_LAG=(size_t)(msr*0.08);
        const size_t BATCH=(size_t)cap_decim*actual_asr/50;

        while(!ch.dem_stop_req.load(std::memory_order_relaxed)){
            size_t wp=ring_wp.load(std::memory_order_acquire);
            size_t rp=ch.dem_rp.load(std::memory_order_relaxed);
            size_t lag=(wp-rp)&IQ_RING_MASK;
            if(lag>MAX_LAG){
                size_t keep=(size_t)(msr*0.02);
                rp=(wp-keep)&IQ_RING_MASK;
                ch.dem_rp.store(rp,std::memory_order_release);
                lpi.s=lpq.s=alf.s=0; prev_i=prev_q=0; am_dc=0;
                aac=0; acnt=0; cap_i=cap_q=0; cap_cnt=0;
                lag=(wp-rp)&IQ_RING_MASK;
            }
            if(lag==0){std::this_thread::sleep_for(std::chrono::microseconds(50));continue;}

            size_t avail=std::min(lag,BATCH);
            for(size_t s=0;s<avail;s++){
                size_t pos=(rp+s)&IQ_RING_MASK;
                float si=ring[pos*2]/2048.0f,sq=ring[pos*2+1]/2048.0f;
                float mi,mq; osc.mix(si,sq,mi,mq);
                cap_i+=mi; cap_q+=mq; cap_cnt++;
                if(cap_cnt<(int)cap_decim) continue;
                float fi=(float)(cap_i/cap_cnt),fq=(float)(cap_q/cap_cnt);
                cap_i=cap_q=0; cap_cnt=0;
                fi=lpi.p(fi); fq=lpq.p(fq);

                // ── Squelch ───────────────────────────────────────────────
                float p_inst = fi*fi + fq*fq;

                // 1) EMA smoothed dBFS
                float db_inst = (p_inst > 1e-12f) ? 10.0f*log10f(p_inst) : -120.0f;
                sql_avg = SQL_ALPHA*db_inst + (1.0f-SQL_ALPHA)*sql_avg;

                // 2) Auto-calibrate threshold on first 500ms (20th percentile of dB)
                if(!calibrated){
                    if((int)calib_buf.size() < CALIB_SAMP)
                        calib_buf.push_back(db_inst);
                    if((int)calib_buf.size() >= CALIB_SAMP){
                        // 20th percentile: robust noise floor, ignores rare deep dips
                        std::vector<float> tmp=calib_buf;
                        size_t p20=tmp.size()/5;
                        std::nth_element(tmp.begin(),tmp.begin()+p20,tmp.end());
                        float noise_floor=tmp[p20];
                        ch.sq_threshold.store(noise_floor + 10.0f, std::memory_order_relaxed);
                        calibrated = true;
                        calib_buf.clear(); calib_buf.shrink_to_fit();
                    }
                }

                // 3) Compare to threshold + hold (locked closed until calibrated)
                float thr = ch.sq_threshold.load(std::memory_order_relaxed);
                const float HYS = 3.0f;
                if(calibrated){
                    if(!gate_open && sql_avg >= thr)
                        { gate_open = true;  gate_hold = SQL_HOLD_SAMP; }
                    if( gate_open){
                        if(sql_avg >= thr - HYS) gate_hold = SQL_HOLD_SAMP;
                        else if(--gate_hold <= 0){ gate_open = false; gate_hold = 0; }
                    }
                }

                // 3) UI update every 256 samples
                if(++sq_ui_tick >= 256){ sq_ui_tick = 0;
                    ch.sq_sig .store(sql_avg,   std::memory_order_relaxed);
                    ch.sq_gate.store(gate_open,  std::memory_order_relaxed);
                }

                // Demodulate — AM reuses p_inst (no extra sqrtf argument)
                float samp=0;
                if(mode==Channel::DM_AM){
                    float env2=sqrtf(p_inst);
                    am_dc=am_dc*0.9995f+env2*0.0005f;
                    samp=alf.p(env2-am_dc)*8.0f;
                } else {
                    float cross=fi*prev_q-fq*prev_i,dot=fi*prev_i+fq*prev_q;
                    float d=atan2f(cross,dot+1e-12f); prev_i=fi; prev_q=fq;
                    samp=alf.p(d)*4.0f;
                }
                aac+=samp; acnt++;
                if(acnt>=(int)actual_ad){
                    float out=gate_open
                              ? std::max(-1.0f,std::min(1.0f,(float)(aac/acnt)))
                              : 0.0f;
                    aac=0; acnt=0;
                    ch.push_audio(out);
                }
            }
            ch.dem_rp.store((rp+avail)&IQ_RING_MASK,std::memory_order_release);
        }
        printf("DEM[%d] worker exited\n",ch_idx);
    }

    void start_dem(int ch_idx, Channel::DemodMode mode){
        Channel& ch=channels[ch_idx];
        if(ch.dem_run.load()||!ch.filter_active) return;
        ch.mode=mode;
        ch.dem_rp.store(ring_wp.load());
        ch.dem_stop_req.store(false);
        ch.dem_run.store(true);
        ch.dem_thr=std::thread(&FFTViewer::dem_worker,this,ch_idx);
        const char* n[]={"NONE","AM","FM"};
        printf("DEM[%d] start: %s  %.4f-%.4f MHz\n",ch_idx,n[(int)mode],ch.s,ch.e);
    }
    void stop_dem(int ch_idx){
        Channel& ch=channels[ch_idx];
        if(!ch.dem_run.load()) return;
        ch.dem_stop_req.store(true);
        if(ch.dem_thr.joinable()) ch.dem_thr.join();
        ch.dem_run.store(false);
        ch.mode=Channel::DM_NONE;
    }
    void stop_all_dem(){
        for(int i=0;i<MAX_CHANNELS;i++) stop_dem(i);
    }

    // ── Stereo mix thread ─────────────────────────────────────────────────
    void mix_worker(){
        AlsaOut alsa; alsa.open(AUDIO_SR);
        static constexpr int PERIOD=256;
        std::vector<int16_t> sbuf(PERIOD*2,0);
        while(!mix_stop.load(std::memory_order_relaxed)){
            // Cache channel state once per period (not per sample)
            bool  ch_active[MAX_CHANNELS];
            int   ch_pan[MAX_CHANNELS];
            for(int c=0;c<MAX_CHANNELS;c++){
                ch_active[c]=channels[c].dem_run.load(std::memory_order_relaxed);
                ch_pan[c]   =channels[c].pan;
            }
            for(int i=0;i<PERIOD;i++){
                float L=0,R=0;
                for(int c=0;c<MAX_CHANNELS;c++){
                    if(!ch_active[c]) continue;
                    float smp=0; channels[c].pop_audio(smp);
                    if(ch_pan[c]<=0) L+=smp;
                    if(ch_pan[c]>=0) R+=smp;
                }
                L=L< -1.0f?-1.0f:L>1.0f?1.0f:L;
                R=R< -1.0f?-1.0f:R>1.0f?1.0f:R;
                sbuf[i*2  ]=(int16_t)(L*32767.0f);
                sbuf[i*2+1]=(int16_t)(R*32767.0f);
            }
            alsa.write(sbuf.data(),PERIOD);
        }
        alsa.close(); printf("Mix worker exited\n");
    }

    // ─────────────────────────────────────────────────────────────────────
    // Display helpers
    // ─────────────────────────────────────────────────────────────────────
    void get_disp(float& ds,float& de) const {
        float nyq=header.sample_rate/2.0f/1e6f,eff=nyq*0.875f,rng=2*eff;
        ds=-eff+freq_pan*rng; de=ds+rng/freq_zoom;
        ds=std::max(-eff,ds); de=std::min(eff,de);
    }
    float x_to_abs(float x,float gx,float gw) const {
        float ds,de; get_disp(ds,de);
        float nm=std::max(0.0f,std::min(1.0f,(x-gx)/gw));
        return (float)(header.center_frequency/1e6f)+ds+nm*(de-ds);
    }
    float abs_to_x(float abs_mhz,float gx,float gw) const {
        float cf=header.center_frequency/1e6f; float ds,de; get_disp(ds,de);
        return gx+(abs_mhz-cf-ds)/(de-ds)*gw;
    }

    // Find channel index whose filter contains mouse x; -1 if none
    int channel_at_x(float mx,float gx,float gw) const {
        for(int i=0;i<MAX_CHANNELS;i++){
            if(!channels[i].filter_active) continue;
            float x0=abs_to_x(std::min(channels[i].s,channels[i].e),gx,gw);
            float x1=abs_to_x(std::max(channels[i].s,channels[i].e),gx,gw);
            x0=std::max(x0,gx); x1=std::min(x1,gx+gw);
            if(mx>=x0&&mx<=x1) return i;
        }
        return -1;
    }

    // ── New-channel drag (right mouse) ────────────────────────────────────
    void handle_new_channel_drag(float gx,float gw){
        ImVec2 m=ImGui::GetIO().MousePos;
        bool in_graph=(m.x>=gx&&m.x<=gx+gw);

        if(in_graph&&ImGui::IsMouseClicked(ImGuiMouseButton_Right)){
            float af=x_to_abs(m.x,gx,gw);
            new_drag.active=true; new_drag.anch=af; new_drag.s=af; new_drag.e=af;
        }
        if(new_drag.active){
            if(ImGui::IsMouseDown(ImGuiMouseButton_Right)){
                float f=x_to_abs(m.x,gx,gw);
                new_drag.s=std::min(new_drag.anch,f); new_drag.e=std::max(new_drag.anch,f);
            }
            if(ImGui::IsMouseReleased(ImGuiMouseButton_Right)){
                new_drag.active=false;
                float bw=fabsf(new_drag.e-new_drag.s);
                if(bw>0.001f){ // min 1kHz wide
                    // Find free slot
                    int slot=-1;
                    for(int i=0;i<MAX_CHANNELS;i++) if(!channels[i].filter_active){slot=i;break;}
                    if(slot>=0){
                        channels[slot].s=new_drag.s; channels[slot].e=new_drag.e;
                        channels[slot].filter_active=true; channels[slot].mode=Channel::DM_NONE;
                        channels[slot].pan=0; channels[slot].selected=false;
                        // Reset audio ring
                        channels[slot].ar_wp.store(0); channels[slot].ar_rp.store(0);
                        // Select this new channel
                        if(selected_ch>=0) channels[selected_ch].selected=false;
                        selected_ch=slot; channels[slot].selected=true;
                    }
                }
            }
        }
    }

    // ── Channel click / double-click / drag-move ──────────────────────────
    // Call this after the InvisibleButton on the graph
    void handle_channel_interactions(float gx,float gw,float gy,float gh){
        ImVec2 m=ImGui::GetIO().MousePos;
        if(m.x<gx||m.x>gx+gw) return;
        bool in_graph=(m.y>=gy&&m.y<=gy+gh);

        // Check if any channel move-drag active (left button held)
        bool any_move=false;
        for(int i=0;i<MAX_CHANNELS;i++) if(channels[i].move_drag){any_move=true;break;}

        if(any_move){
            if(ImGui::IsMouseDown(ImGuiMouseButton_Left)){
                float cur_abs=x_to_abs(m.x,gx,gw);
                for(int i=0;i<MAX_CHANNELS;i++){
                    if(!channels[i].move_drag) continue;
                    float delta_abs=cur_abs-channels[i].move_anchor;
                    // Snap to 1kHz
                    float snapped=roundf(delta_abs*1000.0f)/1000.0f;
                    float half_bw=(channels[i].move_e0-channels[i].move_s0)/2.0f;
                    float new_cf=(channels[i].move_s0+channels[i].move_e0)/2.0f+snapped;
                    channels[i].s=new_cf-half_bw; channels[i].e=new_cf+half_bw;
                }
            } else {
                // Drag ended: restart demod with new center frequency
                for(int i=0;i<MAX_CHANNELS;i++){
                    if(!channels[i].move_drag) continue;
                    channels[i].move_drag=false;
                    if(channels[i].dem_run.load()){
                        Channel::DemodMode m=channels[i].mode;
                        stop_dem(i);
                        start_dem(i,m);
                    }
                }
            }
            return;
        }

        // Double-click: delete channel
        if(in_graph && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)){
            int ci=channel_at_x(m.x,gx,gw);
            if(ci>=0){
                stop_dem(ci);
                channels[ci].filter_active=false;
                channels[ci].selected=false;
                channels[ci].mode=Channel::DM_NONE;
                if(selected_ch==ci) selected_ch=-1;
            }
            return;
        }

        // Single left-click
        if(in_graph && ImGui::IsMouseClicked(ImGuiMouseButton_Left)){
            int ci=channel_at_x(m.x,gx,gw);
            if(selected_ch>=0) channels[selected_ch].selected=false;
            if(ci>=0){
                selected_ch=ci; channels[ci].selected=true;
                channels[ci].move_drag=true;
                channels[ci].move_anchor=x_to_abs(m.x,gx,gw);
                channels[ci].move_s0=std::min(channels[ci].s,channels[ci].e);
                channels[ci].move_e0=std::max(channels[ci].s,channels[ci].e);
            } else {
                selected_ch=-1;
            }
        }
    }

    // ── Draw all channel overlays ─────────────────────────────────────────
    void draw_all_channels(ImDrawList* dl,float gx,float gw,float gy,float gh,bool show_label){
        float cf=header.center_frequency/1e6f;
        float ds,de; get_disp(ds,de); float dw=de-ds;

        // Draw new-drag preview
        if(new_drag.active){
            float x0=gx+(new_drag.s-cf-ds)/dw*gw, x1=gx+(new_drag.e-cf-ds)/dw*gw;
            float c0=std::max(gx,x0), c1=std::min(gx+gw,x1);
            if(c1>c0){
                dl->AddRectFilled(ImVec2(c0,gy),ImVec2(c1,gy+gh),IM_COL32(255,255,255,20));
                dl->AddLine(ImVec2(x0,gy),ImVec2(x0,gy+gh),IM_COL32(200,200,200,160),1.5f);
                dl->AddLine(ImVec2(x1,gy),ImVec2(x1,gy+gh),IM_COL32(200,200,200,160),1.5f);
            }
        }

        for(int i=0;i<MAX_CHANNELS;i++){
            Channel& ch=channels[i];
            if(!ch.filter_active) continue;
            float ss=std::min(ch.s,ch.e),se=std::max(ch.s,ch.e);
            float x0=gx+(ss-cf-ds)/dw*gw, x1=gx+(se-cf-ds)/dw*gw;
            float c0=std::max(gx,x0), c1=std::min(gx+gw,x1);
            if(c1<=c0) continue;

            ImU32 fill=ch.selected?CH_SFIL[i]:CH_FILL[i];
            ImU32 bord=CH_BORD[i];
            if(rec_on.load()&&i==rec_ch) fill=IM_COL32(255,60,60,60);

            dl->AddRectFilled(ImVec2(c0,gy),ImVec2(c1,gy+gh),fill);
            // Dashed border
            auto dash=[&](float x){
                if(x<gx-1||x>gx+gw+1) return;
                for(float y=gy;y<gy+gh;y+=10){float ye=std::min(y+5.0f,gy+gh);dl->AddLine(ImVec2(x,y),ImVec2(x,ye),bord,1.5f);}
            };
            dash(x0); dash(x1);

            if(ch.selected){
                // Bright solid border for selected channel
                dl->AddLine(ImVec2(std::max(gx,x0),gy),ImVec2(std::max(gx,x0),gy+gh),bord,2.0f);
                dl->AddLine(ImVec2(std::min(gx+gw,x1),gy),ImVec2(std::min(gx+gw,x1),gy+gh),bord,2.0f);
            }

            if(!show_label) continue;

            // Label: channel index + mode + freq + BW
            const char* mname[3]={"","AM","FM"};
            const char* pname[]={" L"," L+R"," R"}; // pan -1,0,1
            int pi=ch.pan+1; if(pi<0)pi=0; if(pi>2)pi=2;
            char lb[128];
            if(rec_on.load()&&i==rec_ch)
                snprintf(lb,sizeof(lb),"[%d] REC %.3fMHz / %.1fkHz",i+1,(ss+se)/2.0f,(se-ss)*1000.0f);
            else
                snprintf(lb,sizeof(lb),"[%d]%s%s %.3fMHz / %.1fkHz",
                         i+1,mname[(int)ch.mode],pname[pi],(ss+se)/2.0f,(se-ss)*1000.0f);

            ImVec2 ts=ImGui::CalcTextSize(lb);
            float cx=std::max(gx,std::min(gx+gw-ts.x,(c0+c1)/2-ts.x/2));
            float ly=gy+4;
            dl->AddRectFilled(ImVec2(cx-2,ly),ImVec2(cx+ts.x+2,ly+ts.y+2),IM_COL32(0,0,0,190));
            dl->AddText(ImVec2(cx,ly+1),ch.sq_gate.load()?bord:IM_COL32(160,160,160,200),lb);
        }
    }

    // ── Freq grid ─────────────────────────────────────────────────────────
    void draw_freq_axis(ImDrawList* dl,float gx,float gw,float gy,float gh,bool ticks_only=false){
        float cf=header.center_frequency/1e6f;
        float ds,de; get_disp(ds,de); float dr=de-ds;
        if(freq_zoom<=1.0f){
            float step=5,first=ceilf((ds+cf)/step)*step;
            for(float af=first;af<=de+cf+1e-4f;af+=step){
                float x=gx+(af-cf-ds)/dr*gw; if(x<gx||x>gx+gw) continue;
                if(!ticks_only) dl->AddLine(ImVec2(x,gy),ImVec2(x,gy+gh),IM_COL32(60,60,60,100),1);
                dl->AddLine(ImVec2(x,gy+gh-5),ImVec2(x,gy+gh),IM_COL32(100,100,100,200),1);
                if(!ticks_only){dl->AddLine(ImVec2(x,gy+gh),ImVec2(x,gy+gh+5),IM_COL32(100,100,100,200),1);
                    char lb[32]; snprintf(lb,32,"%.0f",af); ImVec2 ts=ImGui::CalcTextSize(lb);
                    dl->AddText(ImVec2(x-ts.x/2,gy+gh+8),IM_COL32(0,255,0,255),lb);}
            }
        } else {
            for(int i=0;i<=10;i++){
                float fn=(float)i/10,x=gx+fn*gw,af=cf+ds+fn*dr;
                if(!ticks_only) dl->AddLine(ImVec2(x,gy),ImVec2(x,gy+gh),IM_COL32(60,60,60,100),1);
                dl->AddLine(ImVec2(x,gy+gh-5),ImVec2(x,gy+gh),IM_COL32(100,100,100,200),1);
                if(!ticks_only){dl->AddLine(ImVec2(x,gy+gh),ImVec2(x,gy+gh+5),IM_COL32(100,100,100,200),1);
                    char lb[32]; snprintf(lb,32,"%.3f",af); ImVec2 ts=ImGui::CalcTextSize(lb);
                    dl->AddText(ImVec2(x-ts.x/2,gy+gh+8),IM_COL32(0,255,0,255),lb);}
            }
        }
    }

    // ── Zoom scroll ───────────────────────────────────────────────────────
    void handle_zoom_scroll(float gx,float gw,float mouse_x){
        float wheel=ImGui::GetIO().MouseWheel;
        if(wheel==0) return;
        float nyq=header.sample_rate/2.0f/1e6f,eff=nyq*0.875f,rng=2*eff;
        float mx=(mouse_x-gx)/gw; mx=std::max(0.0f,std::min(1.0f,mx));
        float fmx=-eff+freq_pan*rng+mx*(rng/freq_zoom);
        freq_zoom*=(1+wheel*0.15f); freq_zoom=std::max(1.0f,std::min(200.0f,freq_zoom));
        float nw=rng/freq_zoom,ns=fmx-mx*nw;
        freq_pan=(ns+eff)/rng; freq_pan=std::max(0.0f,std::min(1-1/freq_zoom,freq_pan));
    }

    // ── draw_spectrum_area ────────────────────────────────────────────────
    void draw_spectrum_area(ImDrawList* dl,float full_x,float full_y,float total_w,float total_h){
        float gx=full_x+AXIS_LABEL_WIDTH, gy=full_y;
        float gw=total_w-AXIS_LABEL_WIDTH, gh=total_h-BOTTOM_LABEL_HEIGHT;
        dl->AddRectFilled(ImVec2(full_x,full_y),ImVec2(full_x+total_w,full_y+total_h),IM_COL32(10,10,10,255));

        float ds,de; get_disp(ds,de);
        float sr_mhz=header.sample_rate/1e6f; int np=(int)gw;
        bool cv=(cached_sp_idx==current_fft_idx&&cached_pan==freq_pan&&cached_zoom==freq_zoom&&
                 cached_px==np&&cached_pmin==display_power_min&&cached_pmax==display_power_max);
        if(!cv){
            current_spectrum.assign(np,-80.0f);
            float nyq=sr_mhz/2.0f; int hf=header.fft_size/2;
            int mi=current_fft_idx%MAX_FFTS_MEMORY;
            for(int px=0;px<np;px++){
                float fd=ds+(float)px/np*(de-ds);
                int bin=(fd>=0)?(int)((fd/nyq)*hf):fft_size+(int)((fd/nyq)*hf);
                if(bin>=0&&bin<fft_size)
                    current_spectrum[px]=(fft_data[mi*fft_size+bin]/127.0f)*(header.power_max-header.power_min)+header.power_min;
            }
            cached_sp_idx=current_fft_idx;cached_pan=freq_pan;cached_zoom=freq_zoom;
            cached_px=np;cached_pmin=display_power_min;cached_pmax=display_power_max;
        }
        float pr=display_power_max-display_power_min;
        for(int px=0;px<np-1;px++){
            if(px>=(int)current_spectrum.size()||px+1>=(int)current_spectrum.size()) break;
            float p1=std::max(0.0f,std::min(1.0f,(current_spectrum[px]-display_power_min)/pr));
            float p2=std::max(0.0f,std::min(1.0f,(current_spectrum[px+1]-display_power_min)/pr));
            dl->AddLine(ImVec2(gx+px,gy+(1-p1)*gh),ImVec2(gx+px+1,gy+(1-p2)*gh),IM_COL32(0,255,0,255),1.5f);
        }
        for(int i=1;i<=9;i++){
            float y=gy+(float)i/10*gh;
            dl->AddLine(ImVec2(gx,y),ImVec2(gx+gw,y),IM_COL32(60,60,60,100),1);
            dl->AddLine(ImVec2(gx-5,y),ImVec2(gx,y),IM_COL32(100,100,100,200),1);
            char lb[16]; snprintf(lb,16,"%.0f",-8.0f*i); ImVec2 ts=ImGui::CalcTextSize(lb);
            dl->AddText(ImVec2(gx-10-ts.x,y-7),IM_COL32(200,200,200,255),lb);
        }
        draw_freq_axis(dl,gx,gw,gy,gh,false);
        draw_all_channels(dl,gx,gw,gy,gh,true);

        ImGui::SetCursorScreenPos(ImVec2(gx,gy));
        ImGui::InvisibleButton("sp_graph",ImVec2(gw,gh));
        bool hov=ImGui::IsItemHovered();
        handle_new_channel_drag(gx,gw);
        handle_channel_interactions(gx,gw,gy,gh);
        if(hov){
            ImVec2 mm=ImGui::GetIO().MousePos;
            float af=x_to_abs(mm.x,gx,gw);
            char info[48]; snprintf(info,48,"%.3f MHz",af);
            ImVec2 ts=ImGui::CalcTextSize(info);
            float tx=gx+gw-ts.x-4,ty=gy+2;
            dl->AddRectFilled(ImVec2(tx-2,ty),ImVec2(tx+ts.x+2,ty+ts.y+4),IM_COL32(20,20,20,220));
            dl->AddRect(ImVec2(tx-2,ty),ImVec2(tx+ts.x+2,ty+ts.y+4),IM_COL32(100,100,100,255));
            dl->AddText(ImVec2(tx,ty+2),IM_COL32(0,255,0,255),info);
            handle_zoom_scroll(gx,gw,mm.x);
        }
        // Power axis drag
        ImGui::SetCursorScreenPos(ImVec2(full_x,gy));
        ImGui::InvisibleButton("pax",ImVec2(AXIS_LABEL_WIDTH,gh));
        static float dsy=0,dsmin=0,dsmax=0; static bool dl_lo=false;
        if(ImGui::IsItemActive()){
            if(ImGui::IsMouseClicked(ImGuiMouseButton_Left)){
                ImVec2 m2=ImGui::GetMousePos();
                float mid=(display_power_min+display_power_max)/2;
                float midy=gy+gh*(1-(mid-display_power_min)/(display_power_max-display_power_min));
                dsy=m2.y;dsmin=display_power_min;dsmax=display_power_max;dl_lo=(m2.y>midy);
            }
            if(ImGui::IsMouseDragging(ImGuiMouseButton_Left)){
                ImVec2 m2=ImGui::GetMousePos(); float dy=m2.y-dsy;
                float midp=(dsmin+dsmax)/2,midyy=gy+gh*(1-(midp-dsmin)/(dsmax-dsmin));
                if(dl_lo){float n=dy/(gy+gh-midyy);n=std::max(-1.0f,std::min(1.0f,n));display_power_min=midp-n*50;}
                else     {float n=-dy/midyy;n=std::max(-1.0f,std::min(1.0f,n));display_power_max=midp+n*50;}
                if(display_power_max-display_power_min<5){float md=(display_power_min+display_power_max)/2;display_power_min=md-2.5f;display_power_max=md+2.5f;}
                cached_sp_idx=-1;
            }
        }
    }

    // ── draw_waterfall_area ───────────────────────────────────────────────
    void draw_waterfall_area(ImDrawList* dl,float full_x,float full_y,float total_w,float total_h){
        float gx=full_x+AXIS_LABEL_WIDTH, gy=full_y;
        float gw=total_w-AXIS_LABEL_WIDTH, gh=total_h;
        dl->AddRectFilled(ImVec2(full_x,full_y),ImVec2(full_x+total_w,full_y+total_h),IM_COL32(10,10,10,255));
        if(!waterfall_texture) create_waterfall_texture();
        if(total_ffts>0&&last_wf_update_idx!=current_fft_idx){update_wf_row(current_fft_idx);last_wf_update_idx=current_fft_idx;}
        if(waterfall_texture){
            float ds,de; get_disp(ds,de);
            float nyq=header.sample_rate/2.0f/1e6f;
            int dr2=std::min(total_ffts,MAX_FFTS_MEMORY);
            float us=(ds+nyq)/(2*nyq),ue=(de+nyq)/(2*nyq);
            float vn=(float)(current_fft_idx%MAX_FFTS_MEMORY)/MAX_FFTS_MEMORY;
            float vt=vn+1.0f/MAX_FFTS_MEMORY;
            float dh=(dr2>=(int)gh)?gh:(float)dr2;
            float vb=vt-(float)dr2/MAX_FFTS_MEMORY;
            ImTextureID tid=(ImTextureID)(intptr_t)waterfall_texture;
            dl->AddImage(tid,ImVec2(gx,gy),ImVec2(gx+gw,gy+dh),ImVec2(us,vt),ImVec2(ue,vb),IM_COL32(255,255,255,255));
        }
        draw_freq_axis(dl,gx,gw,gy,gh,true);
        draw_all_channels(dl,gx,gw,gy,gh,false);
        ImGui::SetCursorScreenPos(ImVec2(gx,gy));
        ImGui::InvisibleButton("wf_graph",ImVec2(gw,gh));
        bool hov=ImGui::IsItemHovered();
        handle_new_channel_drag(gx,gw);
        handle_channel_interactions(gx,gw,gy,gh);
        if(hov){
            ImVec2 mm=ImGui::GetIO().MousePos;
            float af=x_to_abs(mm.x,gx,gw);
            char info[64]; snprintf(info,64,"%.3f MHz",af);
            ImVec2 ts=ImGui::CalcTextSize(info);
            float tx=gx+gw-ts.x,ty=gy;
            dl->AddRectFilled(ImVec2(tx,ty),ImVec2(tx+ts.x,ty+ts.y+5),IM_COL32(20,20,20,220));
            dl->AddRect(ImVec2(tx,ty),ImVec2(tx+ts.x,ty+ts.y+5),IM_COL32(100,100,100,255));
            dl->AddText(ImVec2(tx,ty+2),IM_COL32(0,255,0,255),info);
            handle_zoom_scroll(gx,gw,mm.x);
        }
    }
};  // FFTViewer

// ─────────────────────────────────────────────────────────────────────────────
void run_streaming_viewer(){
    float cf=450.0f,sr=61.44f;
    FFTViewer v;
    if(!v.initialize_bladerf(cf,sr)){printf("BladeRF init failed\n");return;}

    std::thread cap(&FFTViewer::capture_and_process,&v);
    v.mix_stop.store(false);
    v.mix_thr=std::thread(&FFTViewer::mix_worker,&v);

    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR,3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR,3);
    glfwWindowHint(GLFW_OPENGL_PROFILE,GLFW_OPENGL_CORE_PROFILE);
    GLFWwindow* win=glfwCreateWindow(1400,900,v.window_title.c_str(),nullptr,nullptr);
    glfwMakeContextCurrent(win); glfwSwapInterval(0);
    glewExperimental=GL_TRUE; glewInit();
    ImGui::CreateContext(); ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(win,true); ImGui_ImplOpenGL3_Init("#version 330");
    v.create_waterfall_texture();

    while(!glfwWindowShouldClose(win)){
        glfwPollEvents();
        ImGui_ImplOpenGL3_NewFrame(); ImGui_ImplGlfw_NewFrame(); ImGui::NewFrame();
        v.topbar_sel_this_frame=false;
        if(v.texture_needs_recreate){v.texture_needs_recreate=false;v.create_waterfall_texture();}

        ImGuiIO& io=ImGui::GetIO();
        bool editing=ImGui::IsAnyItemActive();
        int sci=v.selected_ch; // shorthand

        // ── Keyboard shortcuts ────────────────────────────────────────────
        if(!editing){
            // R: record (single channel only)
            if(ImGui::IsKeyPressed(ImGuiKey_R,false)){
                if(v.rec_on.load()) v.stop_rec(); else v.start_rec();
            }
            // A/F/W: set demod on selected channel
            if(sci>=0 && v.channels[sci].filter_active){
                auto set_mode=[&](Channel::DemodMode m){
                    Channel& ch=v.channels[sci];
                    if(ch.dem_run.load()&&ch.mode==m){ v.stop_dem(sci); }
                    else { v.stop_dem(sci); v.start_dem(sci,m); }
                };
                if(ImGui::IsKeyPressed(ImGuiKey_A,false)) set_mode(Channel::DM_AM);
                if(ImGui::IsKeyPressed(ImGuiKey_F,false)) set_mode(Channel::DM_FM);
                // Arrow keys: pan direction
                if(ImGui::IsKeyPressed(ImGuiKey_LeftArrow,false))  v.channels[sci].pan=-1;
                if(ImGui::IsKeyPressed(ImGuiKey_RightArrow,false)) v.channels[sci].pan= 1;
                if(ImGui::IsKeyPressed(ImGuiKey_UpArrow,false))    v.channels[sci].pan= 0;
                // Arrow keys: move filter by 1kHz
                if(ImGui::IsKeyPressed(ImGuiKey_LeftArrow,true)||ImGui::IsKeyPressed(ImGuiKey_RightArrow,true)){
                    // handled by pan already; for move use Ctrl+Arrow
                }
            }
            // ESC: deselect / Delete: remove selected channel
            if(ImGui::IsKeyPressed(ImGuiKey_Escape,false)){
                if(sci>=0){ v.channels[sci].selected=false; v.selected_ch=-1; }
            }
            if(ImGui::IsKeyPressed(ImGuiKey_Delete,false)){
                if(sci>=0 && v.channels[sci].filter_active){
                    v.stop_dem(sci);
                    v.channels[sci].filter_active=false;
                    v.channels[sci].selected=false;
                    v.channels[sci].mode=Channel::DM_NONE;
                    v.selected_ch=-1;
                }
            }
            // Enter: focus freq
            if(ImGui::IsKeyPressed(ImGuiKey_Enter,false)||ImGui::IsKeyPressed(ImGuiKey_KeypadEnter,false)){
                // handled below via focus_freq flag
            }
        }

        // ── Main window ───────────────────────────────────────────────────
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding,ImVec2(0,0));
        ImGui::SetNextWindowPos(ImVec2(0,0));
        ImGui::SetNextWindowSize(io.DisplaySize);
        ImGui::Begin("##main",nullptr,
                     ImGuiWindowFlags_NoMove|ImGuiWindowFlags_NoResize|
                     ImGuiWindowFlags_NoScrollbar|ImGuiWindowFlags_NoTitleBar|
                     ImGuiWindowFlags_NoBringToFrontOnFocus);
        ImGui::PopStyleVar();

        ImDrawList* dl=ImGui::GetWindowDrawList();
        float disp_w=io.DisplaySize.x, disp_h=io.DisplaySize.y;

        dl->AddRectFilled(ImVec2(0,0),ImVec2(disp_w,TOPBAR_H),IM_COL32(30,30,30,255));
        ImGui::SetCursorPos(ImVec2(6,6));
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing,ImVec2(8,4));

        // Freq input
        static float new_freq=450.0f; static bool fdeact=false,focus_freq=false;
        if(!editing&&(ImGui::IsKeyPressed(ImGuiKey_Enter,false)||ImGui::IsKeyPressed(ImGuiKey_KeypadEnter,false))) focus_freq=true;
        if(fdeact){fdeact=false;ImGui::SetWindowFocus(nullptr);}
        if(focus_freq){ImGui::SetKeyboardFocusHere();focus_freq=false;}
        ImGui::SetNextItemWidth(120);
        ImGui::InputFloat("##freq",&new_freq,0,0,"%.3f MHz");
        if(ImGui::IsItemDeactivatedAfterEdit()){v.pending_cf=new_freq;v.freq_req=true;fdeact=true;}
        if(ImGui::IsItemHovered()) ImGui::SetTooltip("Center Frequency  [Enter] to edit");
        ImGui::SameLine();

        // FFT size
        static const int fft_sizes[]={512,1024,2048,4096,8192,16384,32768};
        static const char* fft_lbls[]={"512","1024","2048","4096","8192","16384","32768"};
        static int fft_si=4;
        ImGui::Text("FFT:"); ImGui::SameLine(); ImGui::SetNextItemWidth(72);
        if(ImGui::BeginCombo("##fftsize",fft_lbls[fft_si],ImGuiComboFlags_HeightSmall)){
            for(int i=0;i<7;i++){
                bool sel2=(fft_si==i);
                if(ImGui::Selectable(fft_lbls[i],sel2)){fft_si=i;v.pending_fft_size=fft_sizes[i];v.fft_size_change_req=true;}
                if(sel2)ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }
        ImGui::SameLine();

        // ── Squelch slider (선택 채널 있을 때만 표시) ────────────────────
        if(sci>=0 && v.channels[sci].filter_active){
            Channel& sch=v.channels[sci];
            float sig  =sch.sq_sig .load(std::memory_order_relaxed);
            bool  gopen=sch.sq_gate.load(std::memory_order_relaxed);

            ImGui::Text("SQL:"); ImGui::SameLine();

            const float SLIDER_W=160.0f, SLIDER_H=14.0f;
            ImVec2 sp=ImGui::GetCursorScreenPos();
            sp.y=ImGui::GetWindowPos().y+(TOPBAR_H-SLIDER_H)/2.0f;
            ImDrawList* bdl=ImGui::GetWindowDrawList();

            // dBFS range: sync with spectrum autoscale
            const float DB_MIN = v.display_power_min;
            const float DB_MAX = v.display_power_max;
            const float DB_RNG = std::max(1.0f, DB_MAX - DB_MIN);
            float thr_db = sch.sq_threshold.load(std::memory_order_relaxed);
            auto to_x=[&](float db)->float{
                float t=(db-DB_MIN)/DB_RNG;
                if(t<0)t=0; if(t>1)t=1;
                return sp.x+t*SLIDER_W;
            };

            // Track background
            bdl->AddRectFilled(ImVec2(sp.x,sp.y),ImVec2(sp.x+SLIDER_W,sp.y+SLIDER_H),IM_COL32(40,40,40,255),3);

            // Green bar: current signal level
            float sw=to_x(sig)-sp.x;
            if(sw>0){
                ImU32 sc=gopen?IM_COL32(60,220,60,200):IM_COL32(40,110,40,150);
                bdl->AddRectFilled(ImVec2(sp.x,sp.y),ImVec2(sp.x+sw,sp.y+SLIDER_H),sc,3);
            }

            // Yellow line: threshold
            float tx=to_x(thr_db);
            bdl->AddLine(ImVec2(tx,sp.y),ImVec2(tx,sp.y+SLIDER_H),IM_COL32(255,220,0,230),2.5f);

            // Label: dBFS value
            char lbl[16]; snprintf(lbl,sizeof(lbl),"%.0fdB",thr_db);
            ImVec2 lsz=ImGui::CalcTextSize(lbl);
            bdl->AddText(ImVec2(sp.x+SLIDER_W/2-lsz.x/2,sp.y+(SLIDER_H-lsz.y)/2),IM_COL32(230,230,230,255),lbl);

            // Interaction: mouse wheel moves threshold 1dB per step
            ImGui::SetCursorScreenPos(sp);
            ImGui::InvisibleButton("##sql",ImVec2(SLIDER_W,SLIDER_H));
            if(ImGui::IsItemHovered()){
                float wheel=ImGui::GetIO().MouseWheel;
                if(wheel!=0.0f){
                    float nthr=thr_db+(wheel>0?5.0f:-5.0f);
                    if(nthr<DB_MIN) nthr=DB_MIN;
                    if(nthr>DB_MAX) nthr=DB_MAX;
                    sch.sq_threshold.store(nthr,std::memory_order_relaxed);
                }
                ImGui::SetTooltip("[%d] SQL thr=%.1fdBFS  sig=%.1fdBFS  gate=%s",
                                  sci+1, thr_db, sig, gopen?"OPEN":"CLOSED");
            }
            // Drag to set threshold directly
            if(ImGui::IsItemActive()){
                float mx=ImGui::GetIO().MousePos.x;
                float nthr=DB_MIN+((mx-sp.x)/SLIDER_W)*DB_RNG;
                if(nthr<DB_MIN) nthr=DB_MIN;
                if(nthr>DB_MAX) nthr=DB_MAX;
                sch.sq_threshold.store(nthr,std::memory_order_relaxed);
            }
            ImGui::SetCursorScreenPos(ImVec2(sp.x+SLIDER_W+6,ImGui::GetCursorScreenPos().y));
        }

        // ── Right side: channel status + rec ─────────────────────────────
        {
            float rx=disp_w-8.0f;
            float ty2=(TOPBAR_H-ImGui::GetFontSize())/2;

            if(v.rec_on.load()){
                float el=std::chrono::duration<float>(std::chrono::steady_clock::now()-v.rec_t0).count();
                uint64_t fr=v.rec_frames.load(); float mb=(float)(fr*4)/1048576.0f;
                int mm2=(int)(el/60),ss2=(int)(el)%60;
                char rbuf[80]; snprintf(rbuf,sizeof(rbuf),"REC %d:%02d %.1fMB  ",mm2,ss2,mb);
                ImVec2 rs=ImGui::CalcTextSize(rbuf); rx-=rs.x;
                dl->AddText(ImVec2(rx,ty2),IM_COL32(255,80,80,255),rbuf);
            }

            // Per-channel: right to left
            for(int i=MAX_CHANNELS-1;i>=0;i--){
                Channel& ch=v.channels[i]; if(!ch.filter_active) continue;
                float cf_mhz=(std::min(ch.s,ch.e)+std::max(ch.s,ch.e))/2.0f;
                const char* mn3[3]={"","AM ","FM "};
                bool dem_active=ch.dem_run.load();
                char cb[64];
                snprintf(cb,sizeof(cb),"[%d] %s%.3f MHz  ",
                         i+1,
                         dem_active?mn3[(int)ch.mode]:"",
                         cf_mhz);
                ImVec2 cs2=ImGui::CalcTextSize(cb); rx-=cs2.x;
                ImU32 tc=ch.sq_gate.load()?CH_BORD[i]:IM_COL32(160,160,160,255);

                // Manual hit test (right-to-left layout breaks InvisibleButton)
                ImVec2 mpos=ImGui::GetIO().MousePos;
                bool hovered=(mpos.x>=rx && mpos.x<rx+cs2.x && mpos.y>=0 && mpos.y<TOPBAR_H);
                if(hovered){
                    tc=IM_COL32(255,255,255,255);
                    ImGui::SetTooltip("click to select  double-click to remove");
                    if(ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)){
                        v.stop_dem(i);
                        v.channels[i].filter_active=false;
                        v.channels[i].selected=false;
                        v.channels[i].mode=Channel::DM_NONE;
                        if(v.selected_ch==i) v.selected_ch=-1;
                    } else if(ImGui::IsMouseClicked(ImGuiMouseButton_Left)){
                        if(v.selected_ch>=0) v.channels[v.selected_ch].selected=false;
                        v.selected_ch=i; v.channels[i].selected=true;
                        v.topbar_sel_this_frame=true;
                        printf("TOPBAR click ch%d selected_ch=%d\n",i,v.selected_ch);
                    }
                }
                dl->AddText(ImVec2(rx,ty2),tc,cb);
            }
        }

        ImGui::PopStyleVar(); // ItemSpacing

        // ── Spectrum + Waterfall ──────────────────────────────────────────
        float content_y=TOPBAR_H,content_h=disp_h-content_y,div_h=14.0f;
        float sp_h=(content_h-div_h)*v.spectrum_height_ratio;
        float wf_h=content_h-div_h-sp_h;
        v.draw_spectrum_area(dl,0,content_y,disp_w,sp_h);

        float div_y=content_y+sp_h;
        dl->AddRectFilled(ImVec2(0,div_y),ImVec2(disp_w,div_y+div_h),IM_COL32(50,50,50,255));
        dl->AddLine(ImVec2(0,div_y+div_h/2),ImVec2(disp_w,div_y+div_h/2),IM_COL32(80,80,80,255),1);
        ImGui::SetCursorScreenPos(ImVec2(0,div_y));
        ImGui::InvisibleButton("div",ImVec2(disp_w,div_h));
        if(ImGui::IsItemActive()&&ImGui::IsMouseDragging(ImGuiMouseButton_Left)){
            float d=io.MouseDelta.y; v.spectrum_height_ratio+=d/content_h;
            v.spectrum_height_ratio=std::max(0.1f,std::min(0.9f,v.spectrum_height_ratio));
        }
        if(ImGui::IsItemHovered()) ImGui::SetMouseCursor(ImGuiMouseCursor_ResizeNS);

        v.draw_waterfall_area(dl,0,div_y+div_h,disp_w,wf_h);
        ImGui::End();

        ImGui::Render();
        int dw2,dh2; glfwGetFramebufferSize(win,&dw2,&dh2);
        glViewport(0,0,dw2,dh2);
        glClearColor(0.1f,0.1f,0.1f,1); glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(win);
    }

    v.is_running=false;
    v.stop_all_dem();
    if(v.rec_on.load()) v.stop_rec();
    v.mix_stop.store(true); if(v.mix_thr.joinable()) v.mix_thr.join();
    cap.join();
    if(v.waterfall_texture) glDeleteTextures(1,&v.waterfall_texture);
    ImGui_ImplOpenGL3_Shutdown(); ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext(); glfwTerminate();
    printf("Closed\n");
}

int main(){
    setenv("GTK_IM_MODULE","none",1); setenv("QT_IM_MODULE","none",1);
    setenv("XMODIFIERS","@im=none",1); setenv("GLFW_IM_MODULE","none",1);
    run_streaming_viewer(); return 0;
}