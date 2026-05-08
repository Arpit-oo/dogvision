"""Generate architecture and flowchart diagrams as PNG images for the Word report."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path
import numpy as np

OUT = Path("out/diagrams")
OUT.mkdir(parents=True, exist_ok=True)

# Color palette
C_BG = '#f8f9fa'
C_GPU = '#1a73e8'
C_GPU_LIGHT = '#d2e3fc'
C_CPU = '#5f6368'
C_CPU_LIGHT = '#e8eaed'
C_DOG = '#0d652d'
C_DOG_LIGHT = '#ceead6'
C_PERSON = '#0d47a1'
C_PERSON_LIGHT = '#bbdefb'
C_ALERT = '#c62828'
C_ALERT_LIGHT = '#ffcdd2'
C_WARN = '#e65100'
C_WARN_LIGHT = '#ffe0b2'
C_ARROW = '#424242'
C_ANALYTICS = '#6a1b9a'
C_ANALYTICS_LIGHT = '#e1bee7'
C_TRT = '#00695c'
C_TRT_LIGHT = '#b2dfdb'


def box(ax, x, y, w, h, text, color, text_color='white', fontsize=9, bold=True):
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle="round,pad=0.1", facecolor=color,
                          edgecolor='#333333', linewidth=1.2)
    ax.add_patch(rect)
    weight = 'bold' if bold else 'normal'
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            color=text_color, fontweight=weight, wrap=True,
            fontfamily='sans-serif')


def arrow(ax, x1, y1, x2, y2, color=C_ARROW):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.8))


def arrow_label(ax, x1, y1, x2, y2, label, color=C_ARROW):
    arrow(ax, x1, y1, x2, y2, color)
    mx, my = (x1+x2)/2, (y1+y2)/2
    ax.text(mx + 0.3, my, label, fontsize=7, color='#555555',
            fontfamily='sans-serif', style='italic')


def generate_pipeline_architecture():
    """Fig 1: Complete pipeline architecture diagram."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis('off')
    fig.patch.set_facecolor(C_BG)

    ax.text(5, 13.6, 'DogVision - Complete Pipeline Architecture', fontsize=14,
            ha='center', fontweight='bold', fontfamily='sans-serif', color='#1a1a2e')

    # Video Input
    box(ax, 5, 12.8, 6, 0.6, 'VIDEO INPUT\n(MP4 / Webcam / RTSP)', C_CPU, fontsize=8)

    arrow_label(ax, 5, 12.5, 5, 11.9, 'BGR frames', C_ARROW)

    # YOLO Stage
    box(ax, 5, 11.4, 7, 1.0,
        'STAGE 1: YOLOv8 + ByteTrack (GPU)\n'
        'TensorRT FP16 | cuDNN Convolutions | CUDA NMS\n'
        'Single pass: detect dog (cls=16) + person (cls=0)',
        C_GPU, fontsize=8, bold=True)

    # Split arrows
    arrow_label(ax, 3.5, 10.9, 3, 10.2, 'dogs[]', C_DOG)
    arrow_label(ax, 6.5, 10.9, 7, 10.2, 'persons[]', C_PERSON)

    # Dog pipeline
    box(ax, 3, 9.6, 3.5, 1.0,
        'STAGE 2a: DOG PIPELINE\n'
        'Bite Risk Analyzer\n'
        'Proximity(30%) + Overlap(25%)\n'
        'Lunge(25%) + Sustained(20%)',
        C_DOG, fontsize=7.5)

    # Person pipeline
    box(ax, 7, 9.6, 3.5, 1.0,
        'STAGE 2b: PERSON PIPELINE\n'
        'Access Controller\n'
        'Per-camera YAML time rules\n'
        'Overnight window support',
        C_PERSON, fontsize=7.5)

    # Outputs from pipelines
    arrow_label(ax, 3, 9.1, 3, 8.3, 'BiteEvents', C_ALERT)
    arrow_label(ax, 7, 9.1, 7, 8.3, 'AccessViolations', C_WARN)

    # Event Log
    box(ax, 5, 7.8, 7, 0.7,
        'EVENT LOG - Unified JSON sink (bite_risk + access_violation events)',
        C_ALERT, fontsize=8)

    arrow(ax, 5, 7.45, 5, 6.9)

    # GPU Analytics
    box(ax, 5, 6.3, 7, 1.0,
        'STAGE 3: GPU ANALYTICS\n'
        'CuPy Ring Buffer (O(1) append, 54K rows on GPU VRAM)\n'
        'cuDF Rolling Window (groupby/agg every 30 frames)\n'
        'ROI Color Histograms (CuPy BGR->HSV + bincount)',
        C_ANALYTICS, fontsize=7.5)

    arrow(ax, 5, 5.8, 5, 5.2)

    # Annotation
    box(ax, 5, 4.7, 7, 0.8,
        'STAGE 4: ANNOTATION RENDERING\n'
        'Green boxes (dogs) | Teal boxes (persons)\n'
        'Red alert lines (bite risk) | Orange labels (unauthorized) | HUD overlay',
        C_CPU, fontsize=7.5)

    arrow(ax, 5, 4.3, 5, 3.7)

    # Outputs
    box(ax, 2.5, 3.2, 2.8, 0.7, 'dogvision_output.mp4\nAnnotated Video', '#37474f', fontsize=7.5)
    box(ax, 5, 3.2, 2.2, 0.7, 'events.json\nBite + Access', '#37474f', fontsize=7.5)
    box(ax, 7.5, 3.2, 2.2, 0.7, 'summary.json\nRun Statistics', '#37474f', fontsize=7.5)

    # Legend
    legend_items = [
        (C_GPU, 'GPU-accelerated (CUDA)'),
        (C_DOG, 'Dog pipeline'),
        (C_PERSON, 'Person pipeline'),
        (C_ANALYTICS, 'GPU analytics (CuPy/cuDF)'),
        (C_CPU, 'CPU operations'),
    ]
    for i, (color, label) in enumerate(legend_items):
        ax.add_patch(FancyBboxPatch((0.3, 2.0 - i*0.35), 0.4, 0.25,
                     boxstyle="round,pad=0.02", facecolor=color, edgecolor='none'))
        ax.text(0.95, 2.12 - i*0.35, label, fontsize=7, va='center',
                fontfamily='sans-serif', color='#333333')

    plt.tight_layout()
    plt.savefig(OUT / 'fig1_pipeline_architecture.png', dpi=200, bbox_inches='tight',
                facecolor=C_BG, edgecolor='none')
    plt.close()
    print('  Generated: fig1_pipeline_architecture.png')


def generate_complete_flowchart():
    """Fig 2: Complete working flowchart from video input to output."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 16))
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 20)
    ax.axis('off')
    fig.patch.set_facecolor(C_BG)

    ax.text(4, 19.6, 'DogVision - Complete Working Flowchart', fontsize=13,
            ha='center', fontweight='bold', fontfamily='sans-serif', color='#1a1a2e')

    y = 19.0
    step = 1.1

    # Step 1
    box(ax, 4, y, 5, 0.6, '1. Open Video Source (MP4 / Webcam / RTSP)', C_CPU, fontsize=8)
    y -= step; arrow(ax, 4, y+0.55, 4, y+0.3)

    # Step 2
    box(ax, 4, y, 5, 0.6, '2. Read Frame (cv2.VideoCapture)', C_CPU, fontsize=8)
    y -= step; arrow(ax, 4, y+0.55, 4, y+0.3)

    # Step 3
    box(ax, 4, y, 5, 0.6, '3. Upload Frame to GPU (cudaMemcpy H->D)', C_GPU, fontsize=8)
    y -= step; arrow(ax, 4, y+0.55, 4, y+0.3)

    # Step 4
    box(ax, 4, y, 5, 0.7, '4. YOLOv8 TensorRT FP16 Inference\n53 conv layers on 3584 CUDA cores', C_GPU, fontsize=8)
    y -= step; arrow(ax, 4, y+0.6, 4, y+0.35)

    # Step 5
    box(ax, 4, y, 5, 0.6, '5. CUDA NMS - Filter 8400 -> 2-10 detections', C_GPU, fontsize=8)
    y -= step; arrow(ax, 4, y+0.55, 4, y+0.3)

    # Step 6
    box(ax, 4, y, 5, 0.6, '6. ByteTrack - Assign persistent track IDs', C_GPU, fontsize=8)
    y -= step; arrow(ax, 4, y+0.55, 4, y+0.3)

    # Step 7
    box(ax, 4, y, 5, 0.6, '7. Transfer results to CPU (tiny: ~0.1ms)', '#78909c', fontsize=8)
    y -= step; arrow(ax, 4, y+0.55, 4, y+0.3)

    # Step 8: Route
    box(ax, 4, y, 5, 0.6, '8. Route by class: dog(16) / person(0)', C_CPU, fontsize=8)

    # Split
    y -= step
    arrow(ax, 2.5, y+1.05, 2.2, y+0.3)
    arrow(ax, 5.5, y+1.05, 5.8, y+0.3)

    # Step 9a
    box(ax, 2.2, y, 2.8, 0.7, '9a. Bite Risk\nAnalyzer\n(4-factor score)', C_DOG, fontsize=7)

    # Step 9b
    box(ax, 5.8, y, 2.8, 0.7, '9b. Access\nControl\n(time check)', C_PERSON, fontsize=7)

    # Decision boxes
    y -= step
    # Bite decision
    box(ax, 2.2, y, 2.8, 0.5, 'Score >= 0.40?', C_ALERT_LIGHT, text_color=C_ALERT, fontsize=8)
    arrow(ax, 2.2, y-0.65+0.4, 2.2, y-0.25)
    ax.text(1.1, y, 'YES: Bite\nAlert', fontsize=7, color=C_ALERT, fontweight='bold')

    # Access decision
    box(ax, 5.8, y, 2.8, 0.5, 'Outside hours?', C_WARN_LIGHT, text_color=C_WARN, fontsize=8)
    arrow(ax, 5.8, y-0.65+0.4, 5.8, y-0.25)
    ax.text(7.0, y, 'YES: Access\nViolation', fontsize=7, color=C_WARN, fontweight='bold')

    # Merge
    y -= step
    arrow(ax, 2.2, y+0.55, 4, y+0.3)
    arrow(ax, 5.8, y+0.55, 4, y+0.3)

    # Step 10
    box(ax, 4, y, 5, 0.6, '10. Store in CuPy Ring Buffer (GPU)', C_ANALYTICS, fontsize=8)
    y -= step; arrow(ax, 4, y+0.55, 4, y+0.3)

    # Step 11
    box(ax, 4, y, 5, 0.6, '11. Every 30 frames: cuDF Analytics (GPU)', C_ANALYTICS, fontsize=8)
    y -= step; arrow(ax, 4, y+0.55, 4, y+0.3)

    # Step 12
    box(ax, 4, y, 5, 0.6, '12. Draw annotations + HUD overlay (CPU)', C_CPU, fontsize=8)
    y -= step; arrow(ax, 4, y+0.55, 4, y+0.3)

    # Step 13
    box(ax, 4, y, 5, 0.6, '13. Write frame to output video', C_CPU, fontsize=8)
    y -= step; arrow(ax, 4, y+0.55, 4, y+0.3)

    # Loop back
    box(ax, 4, y, 5, 0.5, 'More frames? -> Loop to Step 2', '#78909c', fontsize=8)

    # Legend
    y -= 0.9
    legend_items = [
        (C_GPU, 'GPU (CUDA)'),
        (C_ANALYTICS, 'GPU Analytics'),
        (C_CPU, 'CPU'),
        (C_DOG, 'Dog Pipeline'),
        (C_PERSON, 'Person Pipeline'),
    ]
    for i, (color, label) in enumerate(legend_items):
        ax.add_patch(FancyBboxPatch((0.5 + i*1.5, y), 0.3, 0.25,
                     boxstyle="round,pad=0.02", facecolor=color, edgecolor='none'))
        ax.text(0.95 + i*1.5, y+0.12, label, fontsize=6.5, va='center', fontfamily='sans-serif')

    plt.tight_layout()
    plt.savefig(OUT / 'fig2_complete_flowchart.png', dpi=200, bbox_inches='tight',
                facecolor=C_BG, edgecolor='none')
    plt.close()
    print('  Generated: fig2_complete_flowchart.png')


def generate_gpu_vs_cpu():
    """Fig 3: GPU vs CPU performance bar chart."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.patch.set_facecolor(C_BG)

    # FPS comparison
    ax = axes[0]
    categories = ['YOLOv8n', 'YOLOv8s', 'YOLOv8m', 'YOLOv8l']
    gpu_fps = [45, 35, 25, 18]
    cpu_fps = [15, 5, 2, 1]

    x = np.arange(len(categories))
    w = 0.35
    ax.bar(x - w/2, gpu_fps, w, label='GPU (TRT FP16)', color=C_GPU, edgecolor='white', linewidth=0.5)
    ax.bar(x + w/2, cpu_fps, w, label='CPU (FP32)', color=C_CPU_LIGHT, edgecolor='#999', linewidth=0.5)
    ax.set_ylabel('Frames Per Second', fontsize=9, fontfamily='sans-serif')
    ax.set_title('FPS by Model Variant', fontsize=10, fontweight='bold', fontfamily='sans-serif')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=8)
    ax.legend(fontsize=8)
    ax.set_facecolor(C_BG)
    ax.axhline(y=25, color=C_ALERT, linestyle='--', alpha=0.5, linewidth=0.8)
    ax.text(3.5, 26, 'Real-time threshold (25 FPS)', fontsize=7, color=C_ALERT, ha='right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Latency comparison
    ax = axes[1]
    metrics = ['Inference\n(p50)', 'Inference\n(p95)', 'Analytics\n(cuDF)', 'Ring Buffer\nAppend']
    gpu_ms = [15, 18, 5, 0.1]
    cpu_ms = [250, 310, 80, 5]

    x = np.arange(len(metrics))
    ax.bar(x - w/2, gpu_ms, w, label='GPU', color=C_GPU, edgecolor='white', linewidth=0.5)
    ax.bar(x + w/2, cpu_ms, w, label='CPU', color=C_CPU_LIGHT, edgecolor='#999', linewidth=0.5)
    ax.set_ylabel('Milliseconds (lower is better)', fontsize=9, fontfamily='sans-serif')
    ax.set_title('Latency Comparison', fontsize=10, fontweight='bold', fontfamily='sans-serif')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=7.5)
    ax.legend(fontsize=8)
    ax.set_yscale('log')
    ax.set_facecolor(C_BG)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUT / 'fig3_gpu_vs_cpu.png', dpi=200, bbox_inches='tight',
                facecolor=C_BG, edgecolor='none')
    plt.close()
    print('  Generated: fig3_gpu_vs_cpu.png')


def generate_tensorrt_optimization():
    """Fig 4: TensorRT optimization pipeline."""
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 6)
    ax.axis('off')
    fig.patch.set_facecolor(C_BG)

    ax.text(4.5, 5.7, 'TensorRT FP16 Optimization Pipeline', fontsize=12,
            ha='center', fontweight='bold', fontfamily='sans-serif', color='#1a1a2e')

    # PyTorch model
    box(ax, 1.8, 4.8, 3, 0.6, 'PyTorch Model\nyolov8m.pt (FP32, 50 MB)', '#455a64', fontsize=8)

    arrow(ax, 3.3, 4.5, 4.5, 4.0)

    # Step 1: Layer Fusion
    box(ax, 4.5, 3.5, 4.5, 0.7,
        'Step 1: Layer Fusion\nConv + BatchNorm + ReLU (3 ops) -> 1 fused kernel\nReduces kernel launch overhead by ~3x',
        C_TRT, fontsize=7.5)

    arrow(ax, 4.5, 3.15, 4.5, 2.7)

    # Step 2: FP16 Quantization
    box(ax, 4.5, 2.2, 4.5, 0.7,
        'Step 2: FP16 Quantization\n32-bit floats -> 16-bit floats\nHalves memory bandwidth, 2x Tensor Core speed',
        C_TRT, fontsize=7.5)

    arrow(ax, 4.5, 1.85, 4.5, 1.4)

    # Step 3: Kernel Auto-Tuning
    box(ax, 4.5, 0.9, 4.5, 0.7,
        'Step 3: Kernel Auto-Tuning\nTests multiple implementations per layer\nSelects fastest for YOUR specific GPU',
        C_TRT, fontsize=7.5)

    arrow(ax, 6.75, 0.9, 8.0, 1.5)

    # Output
    box(ax, 7.2, 2.0, 3, 0.6, 'TensorRT Engine\nyolov8m.engine (FP16, ~30 MB)', C_GPU, fontsize=8)

    # Speedup annotation
    ax.text(7.2, 1.3, '2-3x faster than PyTorch\n17x faster than CPU', fontsize=8,
            ha='center', color=C_GPU, fontweight='bold', fontfamily='sans-serif')

    plt.tight_layout()
    plt.savefig(OUT / 'fig4_tensorrt_optimization.png', dpi=200, bbox_inches='tight',
                facecolor=C_BG, edgecolor='none')
    plt.close()
    print('  Generated: fig4_tensorrt_optimization.png')


def generate_data_residency():
    """Fig 5: Data residency map showing GPU vs CPU memory."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis('off')
    fig.patch.set_facecolor(C_BG)

    ax.text(5, 6.7, 'Data Residency Map - Where Data Lives During Processing', fontsize=12,
            ha='center', fontweight='bold', fontfamily='sans-serif', color='#1a1a2e')

    # CPU column
    ax.add_patch(FancyBboxPatch((0.3, 0.3), 3.5, 5.8, boxstyle="round,pad=0.1",
                 facecolor=C_CPU_LIGHT, edgecolor=C_CPU, linewidth=1.5, alpha=0.3))
    ax.text(2.05, 6.0, 'CPU RAM', fontsize=10, ha='center', fontweight='bold', color=C_CPU)

    # GPU column
    ax.add_patch(FancyBboxPatch((4.5, 0.3), 5.2, 5.8, boxstyle="round,pad=0.1",
                 facecolor=C_GPU_LIGHT, edgecolor=C_GPU, linewidth=1.5, alpha=0.3))
    ax.text(7.1, 6.0, 'GPU VRAM', fontsize=10, ha='center', fontweight='bold', color=C_GPU)

    # CPU items
    box(ax, 2.05, 5.2, 2.8, 0.45, 'Video file (disk I/O)', C_CPU, fontsize=7.5)
    box(ax, 2.05, 4.5, 2.8, 0.45, 'BGR frame (numpy)', C_CPU, fontsize=7.5)
    box(ax, 2.05, 1.0, 2.8, 0.45, 'Stats dict (tiny)', C_CPU, fontsize=7.5)
    box(ax, 2.05, 0.5, 2.8, 0.45, 'HUD + output video', C_CPU, fontsize=7.5)

    # GPU items
    box(ax, 7.1, 4.5, 3.5, 0.45, 'GPU Tensor (FP16)', C_GPU, fontsize=7.5)
    box(ax, 7.1, 3.8, 3.5, 0.45, 'YOLOv8 TRT Inference', C_GPU, fontsize=7.5)
    box(ax, 7.1, 3.1, 3.5, 0.45, 'Detection Tensors', C_GPU, fontsize=7.5)
    box(ax, 7.1, 2.4, 3.5, 0.45, 'CuPy Ring Buffer', C_ANALYTICS, fontsize=7.5)
    box(ax, 7.1, 1.7, 3.5, 0.45, 'cuDF DataFrame', C_ANALYTICS, fontsize=7.5)
    box(ax, 7.1, 1.0, 3.5, 0.45, 'Analytics Results', C_ANALYTICS, fontsize=7.5)

    # Arrows
    arrow_label(ax, 2.05, 4.95, 2.05, 4.73, '', C_ARROW)
    arrow_label(ax, 3.45, 4.5, 5.35, 4.5, 'cudaMemcpy (~1ms)', C_GPU)
    arrow(ax, 7.1, 4.27, 7.1, 4.03)
    arrow(ax, 7.1, 3.57, 7.1, 3.33)
    arrow(ax, 7.1, 2.87, 7.1, 2.63)
    arrow(ax, 7.1, 2.17, 7.1, 1.93)
    arrow(ax, 7.1, 1.47, 7.1, 1.23)
    arrow_label(ax, 5.35, 1.0, 3.45, 1.0, '.to_dict() (~0.1ms)', C_ALERT)

    # Annotation
    ax.text(5, 0.15, 'Data crosses PCIe bus only TWICE: upload frame (1ms) + retrieve stats (0.1ms)',
            fontsize=8, ha='center', fontweight='bold', color=C_ALERT, fontfamily='sans-serif')

    plt.tight_layout()
    plt.savefig(OUT / 'fig5_data_residency.png', dpi=200, bbox_inches='tight',
                facecolor=C_BG, edgecolor='none')
    plt.close()
    print('  Generated: fig5_data_residency.png')


def generate_bite_risk_diagram():
    """Fig 6: Bite risk 4-factor scoring visualization."""
    fig, ax = plt.subplots(1, 1, figsize=(9, 5))
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 6)
    ax.axis('off')
    fig.patch.set_facecolor(C_BG)

    ax.text(4.5, 5.7, 'Bite Risk 4-Factor Scoring System', fontsize=12,
            ha='center', fontweight='bold', fontfamily='sans-serif', color='#1a1a2e')

    # Input
    box(ax, 1.5, 4.8, 2.5, 0.5, 'Dog Detection\n(from YOLO GPU)', C_DOG, fontsize=8)
    box(ax, 4.5, 4.8, 2.5, 0.5, 'Person Detection\n(from YOLO GPU)', C_PERSON, fontsize=8)

    # Arrows to factors
    for x_target in [1.2, 3.2, 5.2, 7.2]:
        arrow(ax, 3.0, 4.55, x_target, 3.95)
        arrow(ax, 5.5, 4.55, x_target, 3.95)

    # 4 factors
    factors = [
        (1.2, 'PROXIMITY\n30%', '2.0x diagonal', C_DOG),
        (3.2, 'OVERLAP\n25%', 'IoU > 0.03', '#1565c0'),
        (5.2, 'LUNGE\n25%', '>25% growth', C_WARN),
        (7.2, 'SUSTAINED\n20%', '3+ frames', C_ANALYTICS),
    ]
    for x, label, thresh, color in factors:
        box(ax, x, 3.5, 1.7, 0.7, label, color, fontsize=8)
        ax.text(x, 3.05, thresh, fontsize=7, ha='center', color='#555',
                fontfamily='sans-serif', style='italic')

    # Merge arrows
    for x in [1.2, 3.2, 5.2, 7.2]:
        arrow(ax, x, 2.8, 4.2, 2.15)

    # Weighted sum
    box(ax, 4.2, 1.8, 4, 0.5,
        'Composite Score = weighted sum (0.0 - 1.0)', '#37474f', fontsize=8)

    arrow(ax, 4.2, 1.55, 4.2, 1.1)

    # Decision
    box(ax, 2.5, 0.7, 2.5, 0.5, 'Score >= 0.40\nBITE ALERT!', C_ALERT, fontsize=8)
    box(ax, 6.0, 0.7, 2.5, 0.5, 'Score < 0.40\nNo alert', '#4caf50', fontsize=8)

    arrow(ax, 3.2, 1.1, 2.5, 0.95)
    arrow(ax, 5.2, 1.1, 6.0, 0.95)

    ax.text(2.8, 1.15, 'YES', fontsize=7, color=C_ALERT, fontweight='bold')
    ax.text(5.5, 1.15, 'NO', fontsize=7, color='#4caf50', fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUT / 'fig6_bite_risk_scoring.png', dpi=200, bbox_inches='tight',
                facecolor=C_BG, edgecolor='none')
    plt.close()
    print('  Generated: fig6_bite_risk_scoring.png')


def generate_multistream_grid():
    """Fig 7: Multi-stream 2x2 CCTV grid architecture."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 6)
    ax.axis('off')
    fig.patch.set_facecolor(C_BG)

    ax.text(4, 5.7, 'Multi-Stream 2x2 CCTV Grid Architecture', fontsize=12,
            ha='center', fontweight='bold', fontfamily='sans-serif', color='#1a1a2e')

    # 4 input cameras
    for i, (x, label) in enumerate([(1, 'CAM 0'), (3, 'CAM 1'), (5, 'CAM 2'), (7, 'CAM 3')]):
        box(ax, x, 5.0, 1.5, 0.45, label, C_CPU, fontsize=8)
        arrow(ax, x, 4.77, x, 4.35)

    # 4 YOLO boxes
    for x in [1, 3, 5, 7]:
        box(ax, x, 4.0, 1.5, 0.5, 'YOLOv8\nBiteRisk\nAccess', C_GPU, fontsize=6.5)

    # Arrows down
    for x in [1, 3, 5, 7]:
        arrow(ax, x, 3.75, x, 3.2)

    # Grid
    box(ax, 2, 2.6, 3.2, 1.0, 'CAM 0          CAM 1\n\nCAM 2          CAM 3', '#37474f', fontsize=8)
    ax.text(2, 1.95, '2x2 Grid (1280x720)', fontsize=7, ha='center', color='#777',
            fontfamily='sans-serif')

    # Arrows to grid
    arrow(ax, 1, 3.2, 1.2, 3.1)
    arrow(ax, 3, 3.2, 2.5, 3.1)
    arrow(ax, 5, 3.2, 2.8, 3.1)
    arrow(ax, 7, 3.2, 3.2, 3.1)

    # Output
    arrow(ax, 2, 2.1, 2, 1.5)
    box(ax, 2, 1.1, 3.2, 0.5, 'multi_stream_output.mp4', '#37474f', fontsize=8)

    # Annotations
    box(ax, 6, 2.5, 3.2, 0.8,
        'Each camera:\n- Independent tracker IDs\n- Independent bite analyzer\n- Per-camera access rules',
        C_GPU_LIGHT, text_color='#333', fontsize=7, bold=False)

    plt.tight_layout()
    plt.savefig(OUT / 'fig7_multistream_grid.png', dpi=200, bbox_inches='tight',
                facecolor=C_BG, edgecolor='none')
    plt.close()
    print('  Generated: fig7_multistream_grid.png')


def generate_evaluation_chart():
    """Fig 8: Evaluation results bar chart."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 4.5))
    fig.patch.set_facecolor(C_BG)

    videos = ['dogbite', 'testdiog', 'CCTV\nPeople', 'House\nBreak-in', 'XlZXsvOuuRc', 'House\n(restricted)']
    dogs = [1, 41, 0, 3, 55, 3]
    persons = [3, 33, 348, 11, 19, 11]
    bites = [73, 469, 0, 29, 71, 29]
    access = [0, 0, 0, 0, 0, 2646]

    x = np.arange(len(videos))
    w = 0.2
    ax.bar(x - 1.5*w, dogs, w, label='Dogs', color=C_DOG, edgecolor='white', linewidth=0.5)
    ax.bar(x - 0.5*w, persons, w, label='Persons', color=C_PERSON, edgecolor='white', linewidth=0.5)
    ax.bar(x + 0.5*w, bites, w, label='Bite Alerts', color=C_ALERT, edgecolor='white', linewidth=0.5)
    ax.bar(x + 1.5*w, access, w, label='Access Violations', color=C_WARN, edgecolor='white', linewidth=0.5)

    ax.set_ylabel('Count', fontsize=9, fontfamily='sans-serif')
    ax.set_title('Evaluation Results Across 6 Test Videos', fontsize=11,
                 fontweight='bold', fontfamily='sans-serif')
    ax.set_xticks(x)
    ax.set_xticklabels(videos, fontsize=7.5)
    ax.legend(fontsize=8, loc='upper left')
    ax.set_facecolor(C_BG)
    ax.set_yscale('symlog', linthresh=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUT / 'fig8_evaluation_results.png', dpi=200, bbox_inches='tight',
                facecolor=C_BG, edgecolor='none')
    plt.close()
    print('  Generated: fig8_evaluation_results.png')


if __name__ == '__main__':
    print('Generating diagrams...')
    generate_pipeline_architecture()
    generate_complete_flowchart()
    generate_gpu_vs_cpu()
    generate_tensorrt_optimization()
    generate_data_residency()
    generate_bite_risk_diagram()
    generate_multistream_grid()
    generate_evaluation_chart()
    print(f'\nAll diagrams saved to {OUT}/')
