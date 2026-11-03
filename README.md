# SolarTwin-PINN: Physics-Informed Edge–Cloud 3D Digital Twin for Solar-Electric Vehicles

[![Q1 Research Benchmark](https://img.shields.io/badge/Q1%20Research-SolarTwin--PINN-0284c7?style=for-the-badge)](https://github.com/ShafinRME/SolarTwin-PINN)
[![ESP32-S3 TinyML](https://img.shields.io/badge/ESP32--S3%20INT8-2.14%20ms%20%7C%2014.8%20KB-10b981?style=for-the-badge)](./firmware_esp32s3)
[![Three.js Digital Twin](https://img.shields.io/badge/3D%20Digital%20Twin-Three.js%20%2B%20WebSerial-8b5cf6?style=for-the-badge)](./web_digital_twin)

> **Official Code, Firmware, Benchmark Dataset & Interactive 3D Web Digital Twin Repository**  
> **Manuscript Title:** *A Physics-Informed Edge–Cloud Digital Twin for Ultra-Fast Global MPPT and Battery Health Prognostics in Solar-Electric Vehicles under Dynamic Partial Shading*  
> **Author:** **Md. Shafin Ahmed**  
> *Department of Robotics and Mechatronics Engineering, University of Dhaka, Dhaka 1000, Bangladesh*  
> **Contact:** `shafin.rmedu@gmail.com`

---

## 🌟 Key Scientific Contributions

1. **Five-Point Sparse Probing & INT8 Physics-Informed TinyML (`2.14 ms`):**  
   Eliminates dense $I\text{--}V$ curve sweeps by sampling only $K=5$ sparse voltage–current points ($\alpha_k \in \{0.20, 0.38, 0.55, 0.72, 0.88\}V_{\text{oc}}$) in $<8\text{ ms}$ and executing a symmetric INT8-quantized Physics-Informed Neural Network (`14.8 KB` Flash) in **`2.14 ms`** on an **ESP32-S3** microcontroller.
2. **Bishop Single-Diode & Schottky Bypass Inflection Prior:**  
   Embeds physical diode current-continuity and power-optimality residuals directly into the training objective, preventing local-peak stagnation under severe 2-peak and 3-peak urban partial shading.
3. **Interactive 3D Web Digital Twin & Cloud Battery SoH Prognostics:**  
   Features a browser-native **Three.js 3D Solar-EV Roof Digital Twin** with real-time USB WebSerial hardware telemetry (`115200 baud`), automated fault diagnosis, and a cloud-side 3S 18650 Li-Ion State-of-Health (SoH) estimator achieving **`RMSE = 0.088%`** over 600 charge–discharge cycles.

---

## 📊 Quantitative Q1 Benchmark Summary (1,000 Dynamic Partial Shading Scenarios)

| MPPT Algorithm | Overall $\eta_{\text{MPPT}}$ (%) | 2-Peak PSC (%) | 3-Peak PSC (%) | $V_{\text{RMSE}}$ (V) | Settling Time (ms) | 10h Energy (Wh) | MCU Flash (KB) | Wilcoxon $p$-value |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Perturb & Observe (P&O) | 87.54% | 82.56% | 87.68% | 9.343 V | 414.60 ms | 165.91 Wh | 2.4 KB | $5.20 \times 10^{-163}$ |
| Incremental Conductance (INC) | 87.45% | 81.43% | 87.67% | 9.357 V | 380.43 ms | 165.81 Wh | 3.1 KB | $8.01 \times 10^{-155}$ |
| Particle Swarm Optimization (PSO) | 95.69% | 95.71% | 95.42% | 3.213 V | 1419.89 ms | 164.17 Wh | 6.8 KB | $3.45 \times 10^{-158}$ |
| Grey Wolf Optimizer (GWO) | 96.77% | 96.82% | 96.63% | 3.559 V | 1180.58 ms | 170.22 Wh | 7.5 KB | $9.08 \times 10^{-146}$ |
| Standard Data-Driven MLP | 94.08% | 91.58% | 92.84% | 5.131 V | **2.05 ms** | 185.42 Wh | 14.2 KB | $2.84 \times 10^{-157}$ |
| 1D-CNN Full-Sweep Estimator | 98.54% | 98.56% | 98.01% | 3.355 V | 38.40 ms | 193.44 Wh | 94.6 KB | $1.71 \times 10^{-117}$ |
| **Proposed SolarTwin-PINN (INT8)** | **99.72%** | **99.72%** | **98.88%** | **2.342 V** | **2.14 ms** | **195.89 Wh** | **14.8 KB** | **Reference (Best)** |

---

## 📂 Repository Structure

- [`web_digital_twin/`](./web_digital_twin): Interactive Three.js 3D Solar-EV Digital Twin, Dual-Axis Bishop $P\text{--}V$/$I\text{--}V$ Oscilloscope, 7-Algorithm Live Energy Race, USB WebSerial Bridge, and One-Click CSV/LaTeX/PNG Exporter.
- [`ml_pipeline/`](./ml_pipeline):
  - [`physics_pv_simulator.py`](./ml_pipeline/physics_pv_simulator.py): Coupled Bishop Single-Diode + Schottky Bypass-Diode multi-string PV solver.
  - [`train_pinn_mppt.py`](./ml_pipeline/train_pinn_mppt.py): Physics-Informed Neural Network (PINN) training & distillation pipeline.
  - [`SolarTwin_Colab_Pro_Master.ipynb`](./ml_pipeline/SolarTwin_Colab_Pro_Master.ipynb): Full Google Colab Pro GPU notebook (NASA POWER Dhaka 5-year telemetry, 50,000 PSC scenarios, Ablation Studies 1 & 2, INT8 quantization, and Cloud Battery SoH estimator).
- [`firmware_esp32s3/`](./firmware_esp32s3):
  - [`solartwin_esp32s3_hil.ino`](./firmware_esp32s3/solartwin_esp32s3_hil.ino): Embedded C++ firmware for ESP32-S3-DevKitC-1 (`INA226`, `ADS1115`, `BH1750`, `DS18B20`, 20 kHz LEDC PWM, and 5-point sparse MOSFET probe).
  - [`esp32_pinn_weights.h`](./firmware_esp32s3/esp32_pinn_weights.h): Auto-generated symmetric INT8 neural network weights (`14.8 KB`).
- [`paper_manuscript/`](./paper_manuscript): Elsevier two-column `elsarticle` LaTeX manuscript (`main.tex`), BibTeX database (`references.bib`), Research Highlights (`highlights.txt`), and Cover Letter (`cover_letter.tex`).
