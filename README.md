# 🦜 Cacatua Noise

Herramienta interna para validar calidad de micrófono y ruido de fondo. Corta.

## 🚀 Ejecución
Simplemente dale doble click a **`cacatuanoise.bat`**.

El script se encarga de instalar dependencias y levantar la app solo.

## 🎚️ Calibración (Ajuste de VARA)
Si sienten que el criterio está muy exigente o muy regalón:

1.  Abran `cacatuanoise.py`.
2.  Busquen `def get_classification` (aprox línea 36).
3.  Ajusten los números del SNR según necesiten:

```python
    if snr < 11.0: return 4 # Calle / Moto
    if snr < 20.0: return 3 # Cafetería / Ruido alto - Aquí fui un poco más exigente que el ejemplo de lvl 3, porque no está realmente dificil de entender ese audio.
    if snr < 35.0: return 2 # Casa normal (Aceptable)
    if snr < 56.0: return 1 # Bueno
    # Lo que sobra es LVL 0 (Estudio)
```

## 🛠️ Estructura (Para cuando metan mano)
*   **`TitanCouncil`**: El cerebro. Usa **WebRTC VAD** (Modo 3) + **Librosa**. Recibe audio RAW `float32`.
*   **`StyleCouncil`**: Mide "dinámica" y "ritmo" (evita que suenen robóticos).
*   **`AudioAnalysisThread`**: Maneja **Doble Buffer** (uno en `dB` para la UI, otro en `Raw` para el análisis de Titan).

## ⚠️ Ojo al Piojo
*   **Si todo da "Estudio" (LVL 0)**: Seguramente tienen activada la cancelación de ruido por hardware (Nvidia Broadcast, Krisp, etc). La app mide lo que le llega (usa el audio loopback).
