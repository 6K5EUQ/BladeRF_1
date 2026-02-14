#include <cstdio>
#include <cstring>

void run_streaming_viewer();
void run_record_fft();

int main() {
    printf("\n");
    printf("========================================\n");
    printf("  FFT Spectrum Analyzer\n");
    printf("========================================\n");
    printf("1. Real-time Streaming Mode (ImGui Viewer)\n");
    printf("2. Record to File Mode\n");
    printf("========================================\n");
    printf("Select mode (1 or 2): ");
    
    int choice;
    if (scanf("%d", &choice) != 1) {
        printf("Invalid input\n");
        return 1;
    }
    
    printf("\n");
    
    if (choice == 1) {
        printf("Starting Real-time Streaming Mode...\n");
        run_streaming_viewer();
    } else if (choice == 2) {
        printf("Starting Record to File Mode...\n");
        run_record_fft();
    } else {
        printf("Invalid choice. Please enter 1 or 2.\n");
        return 1;
    }
    
    return 0;
}