import sys
import os
import time
import numpy as np
import pyaudiowpatch as pyaudio
from collections import deque
from scipy import signal
import librosa
import numpy as np
import webrtcvad
import struct
import pyqtgraph as pg
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QLabel, QProgressBar, QGroupBox, 
                               QComboBox, QPushButton, QCheckBox, QStackedWidget,
                               QFormLayout)
from PySide6.QtCore import QThread, Signal, Qt, QTimer, QRectF, QPointF
from PySide6.QtGui import QPixmap, QFont, QColor, QPalette, QPainter, QPen, QBrush, QConicalGradient, QPolygonF

def calculate_modal_noise_floor(db_array):
    if len(db_array) == 0: return -90.0
    
    valid_db = db_array[db_array > -85.0]
    if len(valid_db) < 10: return -90.0

    lower_half = valid_db[valid_db < np.percentile(valid_db, 60)]
    
    if len(lower_half) == 0: return np.mean(valid_db)

    hist, bins = np.histogram(lower_half, bins=int(np.ptp(lower_half)) + 1)
    
    max_bin_index = np.argmax(hist)
    modal_floor = (bins[max_bin_index] + bins[max_bin_index+1]) / 2
    
    return modal_floor

def get_classification(noise_floor, snr, voice_p90):
    """
    Clasificación con ZONAS FRONTERIZAS (Borderline) para reducir subjetividad.
    Se han creado márgenes de +/- 1.5 a 2 dB en los puntos de corte.
    """
    
    # 0. CASO SIN SEÑAL
    if voice_p90 < -55.0: 
        return 0, "SIN SEÑAL / MUTE", "#607d8b"

    # --- ZONA ROJA (Mala Calidad) ---
    
    # LVL 4 PURO (SNR < 13)
    # Antes cortabas en 15. Ahora < 13 es indiscutiblemente malo.
    if snr < 13.0:
        return 4, "LVL 4: RUIDOSO (Indiscutible)", "#d50000" # Rojo Intenso

    # BORDE 4/3 (SNR 13 - 17)
    # Zona gris alrededor del antiguo 15.
    if snr < 17.0:
        return 3.5, "⚠️ LVL 4/3: BORDE (Muy Ruidoso)", "#ff3d00" # Rojo-Naranja

    # --- ZONA NARANJA (Ruido Notable) ---

    # LVL 3 PURO (SNR 17 - 18.5)
    # Rango estrecho donde claramente es ruido de oficina fuerte pero no calle.
    if snr < 18.5:
        return 3, "LVL 3: RUIDO NOTABLE", "#ff6d00" # Naranja

    # BORDE 3/2 (SNR 18.5 - 22.5)
    # Zona gris alrededor del antiguo 20. ¿Es ventilador fuerte o normal?
    if snr < 22.5:
        return 2.5, "⚠️ LVL 3/2: BORDE (Ruidoso/Medio)", "#ffab00" # Naranja-Amarillo

    # --- ZONA AMARILLA (Aceptable) ---

    # LVL 2 PURO (SNR 22.5 - 33)
    # El estándar casero.
    if snr < 33.0:
        return 2, "LVL 2: ACEPTABLE (Estándar)", "#ffd600" # Amarillo

    # BORDE 2/1 (SNR 33 - 37)
    # Zona gris alrededor del antiguo 35. ¿Es casero o casi estudio?
    if snr < 37.0:
        return 1.5, "⚠️ LVL 2/1: BORDE (Muy Bueno)", "#c6ff00" # Amarillo-Lima

    # --- ZONA VERDE (Buena Calidad) ---

    # LVL 1 PURO (SNR 37 - 54)
    # Habitación silenciosa.
    if snr < 54.0:
        return 1, "LVL 1: BUENO (Silencioso)", "#64dd17" # Verde Lima

    # BORDE 1/0 (SNR 54 - 58)
    # Zona gris alrededor del antiguo 56. ¿Es silencio o silencio absoluto?
    if snr < 58.0:
        return 0.5, "✨ LVL 1/0: BORDE (Casi Perfecto)", "#00e676" # Verde Neón suave

    # --- ZONA PERFECTA ---
    
    # LVL 0 PURO (SNR > 58)
    return 0, "💎 LVL 0: ESTUDIO (Perfecto)", "#00c853" # Verde Intenso

class TitanCouncil:
    @staticmethod
    def evaluate(audio_buffer_float, sample_rate):
        if len(audio_buffer_float) < 100:
            return -90.0, 0.0, -90.0

        data = np.array(audio_buffer_float, dtype=np.float32)

        vad_rate = 32000
        if sample_rate != vad_rate:
            data = librosa.resample(data, orig_sr=sample_rate, target_sr=vad_rate)
        
        pcm_data = (data * 32767).astype(np.int16)

        vad = webrtcvad.Vad()
        vad.set_mode(3)

        frame_duration_ms = 30
        frame_len = int(vad_rate * frame_duration_ms / 1000)
        
        voice_energy = []
        noise_energy = []

        n_frames = len(pcm_data) // frame_len
        
        for i in range(n_frames):
            start = i * frame_len
            end = start + frame_len
            chunk = pcm_data[start:end]
            
            chunk_bytes = chunk.tobytes()
            
            try:
                is_speech = vad.is_speech(chunk_bytes, vad_rate)
                
                float_chunk = data[start:end]
                rms = np.sqrt(np.mean(float_chunk**2))
                db_val = 20 * np.log10(rms + 1e-9)
                
                if is_speech:
                    voice_energy.append(db_val)
                else:
                    noise_energy.append(db_val)
            except Exception as e:
                pass 

        if len(voice_energy) == 0:
            final_voice = -90.0
            final_floor = np.median(noise_energy) if len(noise_energy) > 0 else np.mean(data)
            return final_floor, 0.0, final_voice

        final_voice = np.percentile(voice_energy, 90)
        
        if len(noise_energy) > 0:
            real_noise = [x for x in noise_energy if x > -85.0]
            if len(real_noise) > 0:
                final_floor = np.median(real_noise)
            else:
                final_floor = -90.0
        else:
            final_floor = np.min(voice_energy) - 20

        final_snr = final_voice - final_floor

        print(f"TITAN FIX: Voz={len(voice_energy)} frames | Ruido={len(noise_energy)} frames")
        print(f"RESULTADO: Piso={final_floor:.1f} | Voz={final_voice:.1f} | SNR={final_snr:.1f}")

        return final_floor, final_snr, final_voice


class AudioAnalysisThread(QThread):
    level_update = Signal(float) 
    gauge_update = Signal(float, float, float, bool) 
    test_finished = Signal(dict) 
    phrase_started = Signal()
    phrase_finished = Signal(dict)
    visual_data = Signal(np.ndarray)

    def __init__(self, device_idx):
        super().__init__()
        self.idx = device_idx
        self.running = True
        
        self.local_min = 0
        self.local_max = -90
        
        self.pre_roll_db = deque(maxlen=5) 
        self.event_db = [] 
        
        self.pre_roll_raw = deque(maxlen=5) 
        self.event_raw = [] 
        
        self.is_speaking = False
        self.speech_start_time = 0
        self.last_speech_time = 0

    def run(self):
        p = pyaudio.PyAudio()
        stream = None
        try:
            # --- CORRECCIÓN 1: Pequeña pausa para que el driver de audio despierte ---
            time.sleep(0.2) 
            
            info = p.get_device_info_by_index(self.idx)
            rate = int(info["defaultSampleRate"])
            # Forzamos 2 canales si es loopback para evitar errores
            ch = info["maxInputChannels"] if info["maxInputChannels"] > 0 else 2
            
            chunk_ms = 100 
            chunk = int(rate * chunk_ms / 1000) # Esto suele dar ~4800 muestras
            
            stream = p.open(format=pyaudio.paInt16, channels=ch, rate=rate, input=True, 
                            input_device_index=self.idx, frames_per_buffer=chunk)

            while self.running:
                try:
                    # Lectura segura
                    raw_data = stream.read(chunk, exception_on_overflow=False)
                    
                    # SI LLEGA VACÍO (Pasa mucho con Loopback en silencio), saltamos
                    if not raw_data: 
                        time.sleep(0.01) # Pequeña siesta para no quemar CPU
                        continue 
                    
                    # Conversión y resto del código...
                    arr_int16 = np.frombuffer(raw_data, dtype=np.int16)
                    # ... (sigue tu código normal)
                    arr_float = arr_int16.astype(np.float32) / 32768.0
                    
                    # --- CORRECCIÓN 2: Enviar copia segura al gráfico ---
                    if len(arr_float) > 0:
                        self.visual_data.emit(arr_float.copy())
                    
                    # Calcular dB para la aguja
                    rms = np.sqrt(np.mean(arr_float**2))
                    db = 20 * np.log10(rms + 1e-9)
                    
                    self.level_update.emit(db)
                    self.gauge_update.emit(db, self.local_min, self.local_max, self.is_speaking)
                    
                    curr_time = time.time()
                    if db > -55:
                        self.last_speech_time = curr_time
                        if not self.is_speaking:
                            self.is_speaking = True
                            self.speech_start_time = curr_time
                            self.event_db = list(self.pre_roll_db)
                            self.event_raw = list(self.pre_roll_raw)
                            self.phrase_started.emit()
                        self.event_db.append(db)
                        self.event_raw.append(arr_float)
                    else:
                        if self.is_speaking:
                            self.event_db.append(db)
                            self.event_raw.append(arr_float)
                            if (curr_time - self.last_speech_time) > 1.5:
                                duration = self.last_speech_time - self.speech_start_time
                                if duration > 0.5:
                                    if len(self.event_raw) > 0:
                                        full_audio = np.concatenate(self.event_raw)
                                        piso, snr, voz = TitanCouncil.evaluate(full_audio, rate)
                                        self.test_finished.emit({
                                            "p10": piso, "p90": voz, "snr": snr, 
                                            "duration": duration, "stability": 0
                                        })
                                        prosody = StyleCouncil.evaluate(full_audio, rate)
                                        prosody["duration"] = duration
                                        self.phrase_finished.emit(prosody)
                                self.is_speaking = False
                                self.event_db = []
                                self.event_raw = []
                        else:
                            self.pre_roll_db.append(db)
                            self.pre_roll_raw.append(arr_float)

                except Exception:
                    continue # Si falla un frame, no rompas el programa

        except Exception as e:
            print(f"ERROR: {e}")
        finally:
            if stream: stream.close()
            p.terminate()

    def stop(self):
        self.running = False
        # self.wait()  <--- ¡BORRA O COMENTA ESTA LÍNEA! ES LA CAUSA DEL CONGELAMIENTO.
        # Al quitarla, la interfaz no se bloqueará esperando a un micrófono que no responde.

    def reset_state(self):
        self.event_db = []
        self.pre_roll_db.clear()
        self.event_raw = []      # Reset raw
        self.pre_roll_raw.clear() # Reset raw
        self.is_speaking = False

class StyleCouncil:
    """
    StyleCouncil V2: Detector de Humanidad basado en Caos y Frecuencia Fundamental.
    Distingue: Robot (Plano) vs Leído (Rítmico) vs Natural (Caótico).
    """
    @staticmethod
    def evaluate(audio_float, sample_rate=32000):
        # Mínimo de muestras para análisis fiable
        if len(audio_float) < sample_rate * 0.5:
            return {"score": 0, "label": "...", "color": "#444", "metrics": [0, 0, 0]}

        y = np.array(audio_float)
        
        # --- 1. ANÁLISIS DE TONO (F0 / Pitch) ---
        # Usamos YIN para detectar la frecuencia fundamental (la vibración de la garganta)
        # Solo analizamos frecuencias de voz humana (50Hz a 300Hz) para filtrar ruido
        f0 = librosa.yin(y, fmin=50, fmax=300, sr=sample_rate)
        
        # Limpiamos NaN (donde no detectó tono)
        f0_clean = f0[~np.isnan(f0)]
        
        if len(f0_clean) < 10:
            # Si no hay tono detectable, es susurro o ruido
            pitch_variability = 0
        else:
            # Calculamos la excursión semitonal (qué tanto sube y baja la voz musicalmente)
            # Una voz robótica varía menos de 2 semitonos. Una natural varía más de 5.
            pitch_std = np.std(f0_clean)
            pitch_variability = min(100, (pitch_std / 20.0) * 100) # Normalizamos (20Hz std es muy alto)

        # --- 2. ANÁLISIS DE ARTIFICIALIDAD (Spectral Flatness) ---
        # La voz sintética suele tener "residuos" digitales o armónicos demasiado perfectos.
        # La planitud espectral mide qué tan parecido al ruido blanco es el sonido.
        flatness = librosa.feature.spectral_flatness(y=y)[0]
        # La voz humana natural tiene una planitud baja (muchos picos armónicos).
        # El ruido o vocoders malos tienen planitud alta.
        avg_flatness = np.mean(flatness)
        organic_score = 100 - min(100, avg_flatness * 500) # Invertimos: Más planitud = Menos orgánico

        # --- 3. ANÁLISIS DE RITMO (Predicibilidad) ---
        # Detectamos los golpes de voz (sílabas)
        onset_env = librosa.onset.onset_strength(y=y, sr=sample_rate)
        
        # Autocorrelación: Vemos si el patrón de volumen se repite a sí mismo (LEÍDO/PAUTEADO)
        # Si la señal se parece mucho a sí misma desplazada, es rítmica (mala señal para naturalidad)
        ac = librosa.autocorrelate(onset_env, max_size=sample_rate // 2) # Analizamos hasta 0.5s de lag
        
        # El pico de autocorrelación (descartando el lag 0) nos dice qué tan repetitiva es
        if len(ac) > 1:
            rhythm_repetitiveness = np.max(ac[1:]) / (ac[0] + 1e-9) # Normalizado
        else:
            rhythm_repetitiveness = 0
            
        # Invertimos: Más repetitivo = Menos espontáneo
        spontaneity_score = 100 - min(100, rhythm_repetitiveness * 150)

        # --- LÓGICA DE DECISIÓN (ÁRBOL DE DECISIÓN) ---
        
        # Métricas para el gráfico [Melodía, Humanidad, Espontaneidad]
        metrics = [pitch_variability, organic_score, spontaneity_score]
        
        label = "---"
        color = "#888"
        total_score = np.mean(metrics)

        # 1. DETECTOR DE ROBOT (Falta de tono o tono muy plano)
        if pitch_variability < 15:
            label = "🤖 ROBÓTICO / PLANO"
            color = "#ff1744" # Rojo Intenso
            total_score = 10 # Penalización fuerte

        # 2. DETECTOR DE LECTURA (Tono varía, pero ritmo muy repetitivo)
        elif spontaneity_score < 40:
            label = "📖 LEÍDO / PAUTEADO"
            color = "#ff9100" # Naranja
            
        # 3. DETECTOR DE DRAMATIZACIÓN (Demasiada variación de tono artificial)
        elif pitch_variability > 90:
            label = "🎭 SOBREACTUADO"
            color = "#d500f9" # Violeta

        # 4. ZONA NATURAL
        elif total_score > 60:
            label = "🗣️ NATURAL / FLUIDO"
            color = "#00e676" # Verde
            
        else:
            label = "😐 FORMAL / SERIO"
            color = "#ffd600" # Amarillo

        return {
            "score": total_score,
            "label": label,
            "color": color,
            "metrics": metrics
        }

class NaturalnessRadar(QWidget):
    def __init__(self):
        super().__init__()
        # Aumentamos tamaño mínimo
        self.setMinimumSize(250, 220)
        self.metrics = [0, 0, 0] 
        self.max_metrics = [0, 0, 0] # Memoria de picos
        
        # --- CAMBIO AQUÍ: ETIQUETAS MÁS PRECISAS ---
        # Melodía: ¿Varía el tono? (No es robot)
        # Orgánico: ¿Suena humano? (No es sintético)
        # Caos: ¿Es irregular? (No es leído)
        self.labels = ["MELODÍA", "ORGÁNICO", "CAOS"]

    def set_metrics(self, metrics_list):
        self.metrics = metrics_list
        # Actualizamos la sombra: toma el máximo entre lo que había y lo nuevo
        self.max_metrics = [max(o, n) for o, n in zip(self.max_metrics, metrics_list)]
        self.update()

    def reset_shadow(self):
        # Llamar esto cuando empiece una nueva frase si quieres limpiar la sombra
        self.max_metrics = [0, 0, 0]

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        w, h = self.width(), self.height()
        center = QPointF(w/2, h/2 + 10)
        radius = min(w, h) / 2 - 35 # Un poco más de margen para textos

        # 1. FONDO (Triángulo Base)
        painter.setPen(QPen(QColor("#333"), 1))
        painter.setBrush(Qt.NoBrush)
        
        angles = [-90, 30, 150] 
        base_poly = QPolygonF()
        points_100 = []
        for a in angles:
            rad = np.radians(a)
            p = QPointF(center.x() + radius * np.cos(rad), center.y() + radius * np.sin(rad))
            points_100.append(p)
            base_poly.append(p)
        
        painter.drawPolygon(base_poly)
        for p in points_100: painter.drawLine(center, p)

        # 2. DIBUJAR SOMBRA (MÁXIMOS) - Gris transparente
        max_poly = QPolygonF()
        for i, angle in enumerate(angles):
            val = self.max_metrics[i] / 100.0
            val = max(0.05, val)
            rad = np.radians(angle)
            p = QPointF(center.x() + (radius * val) * np.cos(rad), center.y() + (radius * val) * np.sin(rad))
            max_poly.append(p)
        
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(255, 255, 255, 30)) # Blanco muy transparente
        painter.drawPolygon(max_poly)

        # 3. DIBUJAR VALORES ACTUALES - Color vivo
        data_poly = QPolygonF()
        for i, angle in enumerate(angles):
            val = self.metrics[i] / 100.0
            val = max(0.05, val)
            rad = np.radians(angle)
            p = QPointF(center.x() + (radius * val) * np.cos(rad), center.y() + (radius * val) * np.sin(rad))
            data_poly.append(p)
        
        avg_score = sum(self.metrics) / 3
        if avg_score > 60: col = QColor(0, 230, 118, 180) 
        elif avg_score > 35: col = QColor(255, 214, 0, 180)
        else: col = QColor(255, 23, 68, 180)

        painter.setPen(QPen(col.lighter(), 2))
        painter.setBrush(col)
        painter.drawPolygon(data_poly)

        # 4. ETIQUETAS
        painter.setPen(QColor("#aaa"))
        painter.setFont(QFont("Segoe UI", 9, QFont.Bold))
        offsets = [(0, -20), (25, 10), (-35, 10)]
        
        for i, p in enumerate(points_100):
            # Posición etiqueta
            txt_rect = QRectF(p.x() + offsets[i][0] - 30, p.y() + offsets[i][1] - 10, 80, 20)
            painter.drawText(txt_rect, Qt.AlignCenter, self.labels[i])
            
            # Valor numérico
            val_rect = QRectF(txt_rect.x(), txt_rect.y() + 14, 80, 20)
            painter.setPen(col.lighter())
            painter.drawText(val_rect, Qt.AlignCenter, f"{int(self.metrics[i])}%")
            painter.setPen(QColor("#aaa"))




class AudioVisualizer(QWidget):
    def __init__(self):
        super().__init__()
        # AUMENTO DE TAMAÑO: Forzamos una altura mínima mayor
        self.setMinimumHeight(300) 
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        pg.setConfigOption('background', '#121212')
        pg.setConfigOption('foreground', '#888')

        # --- Gráfico 1: Onda (Osciloscopio) ---
        self.wave_plot = pg.PlotWidget(title="Monitor de Onda")
        self.wave_plot.setYRange(-1, 1)
        self.wave_plot.showGrid(x=False, y=True, alpha=0.2)
        self.wave_plot.setMouseEnabled(x=False, y=False)
        self.wave_plot.hideAxis('bottom')
        self.wave_plot.hideAxis('left')
        self.wave_plot.getPlotItem().hideButtons()
        
        # Curva principal (Verde neón brillante)
        self.wave_curve = self.wave_plot.plot(pen=pg.mkPen('#00e676', width=2))
        layout.addWidget(self.wave_plot)

        # --- Gráfico 2: Espectro (FFT) ---
        self.fft_plot = pg.PlotWidget(title="Espectro de Frecuencia")
        self.fft_plot.setLogMode(x=True, y=False)
        self.fft_plot.setYRange(0, 1) # Mantenemos 0-1 pero multiplicaremos la señal
        self.fft_plot.showGrid(x=True, y=True, alpha=0.2)
        self.fft_plot.setMouseEnabled(x=False, y=False)
        self.fft_plot.hideAxis('left')
        self.fft_plot.getPlotItem().hideButtons()
        
        # Curva de "Sombra" (Máximos históricos) - Gris oscuro transparente
        self.fft_max_curve = self.fft_plot.plot(pen=pg.mkPen('#444', width=1), fillLevel=0, brush=(100, 100, 100, 30))
        # Curva Principal - Azul eléctrico
        self.fft_curve = self.fft_plot.plot(pen=pg.mkPen('#2979ff', width=2), fillLevel=0, brush=(41, 121, 255, 80))
        
        layout.addWidget(self.fft_plot)

        # Buffers
        self.data_buffer = np.zeros(10000)
        self.fft_max_buffer = None # Se inicializa dinámicamente

    def update_data(self, audio_chunk):
        if audio_chunk is None or len(audio_chunk) == 0: return

        chunk_len = len(audio_chunk)
        
        # 1. MONITOR DE ONDA
        if chunk_len > len(self.data_buffer):
            self.data_buffer = np.zeros(chunk_len * 2)
        
        self.data_buffer = np.roll(self.data_buffer, -chunk_len)
        self.data_buffer[-chunk_len:] = audio_chunk
        self.wave_curve.setData(self.data_buffer)

        # 2. ESPECTRO (FFT) MEJORADO
        try:
            fft_data = np.fft.rfft(audio_chunk)
            # SENSIBILIDAD: Multiplicamos por 25 para que las barras suban más
            fft_mag = (np.abs(fft_data) / chunk_len) * 25
            
            # Inicializar buffer de sombra si es necesario
            if self.fft_max_buffer is None or len(self.fft_max_buffer) != len(fft_mag):
                self.fft_max_buffer = np.zeros_like(fft_mag)
            
            # CÁLCULO DE SOMBRA:
            # El buffer máximo decae lentamente (x 0.95) o sube si el nuevo valor es mayor
            self.fft_max_buffer = np.maximum(self.fft_max_buffer * 0.92, fft_mag)
            
            x_axis = np.linspace(20, 20000, len(fft_mag))
            
            # Dibujamos primero la sombra (atrás) y luego la señal (frente)
            self.fft_max_curve.setData(x_axis, self.fft_max_buffer)
            self.fft_curve.setData(x_axis, fft_mag)
            
        except Exception:
            pass 


class GaugeWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(250, 200) 
        self.current_db = -90
        
        self.history = deque(maxlen=50) 
        self.peak_hold = -90

    def update_values(self, current_db):
        self.current_db = max(-90, min(0, current_db))
        self.history.append(self.current_db)
        
        if self.current_db > self.peak_hold:
            self.peak_hold = self.current_db
        else:
            self.peak_hold -= 0.5 
            
        self.update()

    def get_angle_for_db(self, db_value):
        """Convierte dB (-90 a 0) en Grados Qt (180 a 0)"""
        # Aseguramos límites
        val = max(-90, min(0, db_value))
        # Mapeo: -90 -> 180, 0 -> 0.
        # Fórmula lineal: angle = -2 * val
        return -2 * val

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        center = QPointF(w / 2, h * 0.85) 
        radius = min(w, h * 2) / 2 - 25

        # 1. FONDO NEGRO
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#111"))
        painter.drawPie(QRectF(center.x() - radius, center.y() - radius, radius * 2, radius * 2), 0, 180 * 16)

        # 2. ESCALA (Rayitas)
        painter.save()
        painter.translate(center)
        # No rotamos el canvas entero, rotaremos cada línea individualmente
        
        for i in range(0, 91, 2): # De 0 a 90 (que representa -90 a 0 dB)
            db_val = -90 + i
            angle = self.get_angle_for_db(db_val)
            
            painter.save()
            # Nota: Qt rota en sentido horario por defecto. 
            # Como nuestros ángulos van de 180 (izq) a 0 (der), 
            # necesitamos rotar -angle para ir "hacia arriba/izquierda"
            painter.rotate(-angle) 
            
            # Color según severidad
            if db_val > -10: c = QColor("#d50000") # Rojo
            elif db_val > -25: c = QColor("#ff6d00")
            elif db_val > -45: c = QColor("#ffd600")
            else: c = QColor("#00c853") # Verde
            
            is_major = (db_val % 10 == 0)
            painter.setPen(QPen(c, 3 if is_major else 1))
            painter.drawLine(int(radius - (15 if is_major else 8)), 0, int(radius), 0)
            painter.restore()
        painter.restore()

        # 3. AGUJA
        painter.save()
        painter.translate(center)
        # Usamos la misma función de ángulo
        curr_angle = self.get_angle_for_db(self.current_db)
        painter.rotate(-curr_angle) 
        
        painter.setBrush(QColor("#ff3d00"))
        painter.setPen(Qt.NoPen)
        # Dibujamos la aguja apuntando a la derecha (0 grados), la rotación se encarga del resto
        needle = QPolygonF([QPointF(0, -3), QPointF(radius - 5, 0), QPointF(0, 3)])
        painter.drawPolygon(needle)
        painter.restore()

        # 4. TEXTO
        painter.setPen(QColor("white"))
        painter.setFont(QFont("Consolas", 24, QFont.Bold))
        painter.drawText(QRectF(center.x()-50, center.y()-50, 100, 40), Qt.AlignCenter, f"{self.current_db:.1f}")

class CacatuaWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cacatua Noise")
        self.resize(550, 800)
        self.setWindowIcon(QPixmap("cacatua_icon.png"))
        
        # Tema Oscuro Moderno
        self.setStyleSheet("""
            QMainWindow { background-color: #121212; }
            QLabel { color: #e0e0e0; font-family: 'Segoe UI', sans-serif; }
            QGroupBox { 
                border: 2px solid #333; 
                border-radius: 8px; 
                margin-top: 10px; 
                font-weight: bold; 
                color: #00e676; 
                padding-top: 20px;
            }
        """)

        self.analysis_thread = None
        self.init_ui()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(15)
        layout.setContentsMargins(25, 25, 25, 25)

        # header
        header = QHBoxLayout()
        icon_lbl = QLabel()
        pixmap = QPixmap("cacatua_icon.png")
        if not pixmap.isNull():
            icon_lbl.setPixmap(pixmap.scaled(50, 50, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        header.addWidget(icon_lbl)
        
        title = QLabel("CACATUA NOISE")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #00e676; letter-spacing: 1px;")
        header.addWidget(title)
        
        header.addStretch()
        
        self.chk_ontop = QCheckBox("Siempre Visible")
        self.chk_ontop.setStyleSheet("color: #aaa; margin-right: 5px;")
        self.chk_ontop.toggled.connect(self.toggle_ontop)
        header.addWidget(self.chk_ontop)

        self.btn_reset = QPushButton("🗑️ RESET")
        self.btn_reset.setFixedSize(80, 30)
        self.btn_reset.setStyleSheet("""
            QPushButton { background-color: #424242; color: white; border: 1px solid #666; border-radius: 4px; font-weight: bold; }
            QPushButton:hover { background-color: #616161; border-color: #888; }
        """)
        self.btn_reset.clicked.connect(self.reset_app)
        header.addWidget(self.btn_reset)
        
        layout.addLayout(header)

        # Device
        lbl_source = QLabel("Fuente de Sonido:")
        lbl_source.setStyleSheet("color: #00e676; font-weight: bold; margin-bottom: 2px;")
        layout.addWidget(lbl_source)

        self.combo_dev = QComboBox()
        self.combo_dev.setStyleSheet("background-color: #222; color: white; padding: 5px; border: 1px solid #444;")
        # self.refresh_devices()  <-- MOVIDO MAS ABAJO
        # self.combo_dev.currentIndexChanged.connect(self.start_thread) <-- MOVIDO MAS ABAJO
        layout.addWidget(self.combo_dev)
        
        # LAYOUT HORIZONTAL PRINCIPAL
        main_h_layout = QHBoxLayout()
        
        # IZQUIERDA: Medidores Visuales
        left_panel = QVBoxLayout()
        self.gauge = GaugeWidget()
        left_panel.addWidget(self.gauge, alignment=Qt.AlignCenter)
        
        # Dashboard Gráfico
        self.visualizer = AudioVisualizer() 
        # Quitamos setFixedSize para que se expanda, o le damos un mínimo generoso
        self.visualizer.setMinimumSize(420, 300) 
        left_panel.addWidget(self.visualizer)
        
        # AHORA SÍ INICIALIZAMOS DISPOSITIVOS (Porque start_thread usa self.visualizer)
        self.refresh_devices()
        self.combo_dev.currentIndexChanged.connect(self.start_thread)
        
        main_h_layout.addLayout(left_panel)

        # DERECHA: Datos y Textos
        right_panel = QVBoxLayout()
        
        # --- SECTION: VOICE DURATION & QUICK STATUS ---
        # --- SECTION: VOICE DURATION & QUICK STATUS ---
        dur_group = QGroupBox("Análisis de Prosodia")
        dur_layout = QVBoxLayout()
        
        h_voz = QHBoxLayout()
        
        # Izquierda: Textos
        v_textos = QVBoxLayout()
        self.lbl_duration = QLabel("0.00 s")
        self.lbl_duration.setStyleSheet("font-size: 40px; font-weight: bold; color: white;")
        
        self.lbl_style = QLabel("---")
        self.lbl_style.setAlignment(Qt.AlignCenter)
        self.lbl_style.setFixedSize(200, 40)
        self.lbl_style.setStyleSheet("font-size: 14px; background-color: #333; color: #aaa; border-radius: 4px;")
        
        v_textos.addWidget(self.lbl_duration)
        v_textos.addWidget(self.lbl_style)
        v_textos.addStretch()
        h_voz.addLayout(v_textos)
        
        # Derecha: EL NUEVO RADAR
        self.radar = NaturalnessRadar()
        h_voz.addWidget(self.radar)
        
        dur_group.setLayout(h_voz)
        
        # Detalles ocultos o extras
        self.lbl_prosody_details = QLabel("") 
        
        # AQUÍ AÑADIMOS EL NIVEL DE RUIDO (Resultado del último test)
        self.lbl_result_mini = QLabel("Nivel de Ambiente: PENDIENTE DE TEST")
        self.lbl_result_mini.setAlignment(Qt.AlignCenter)
        self.lbl_result_mini.setStyleSheet("color: #777; font-style: italic; margin-top: 5px;")
        
        # Como dur_group ahora tiene layout horizontal, añadimos estos widgets al layout principal de la derecha
        # O podemos meterlos en el v_textos si queremos que estén juntos
        v_textos.addWidget(self.lbl_result_mini)
        
        right_panel.addWidget(dur_group)


        # --- SECTION: 30s ANALYSIS (HIDDEN BUT STRUCTURED) ---
        test_group = QGroupBox("Certificación de Entorno (Continuo)")
        test_layout = QVBoxLayout()
        
        # Status Label
        self.lbl_test_status = QLabel("Esperando voz para certificar...")
        self.lbl_test_status.setAlignment(Qt.AlignCenter)
        self.lbl_test_status.setStyleSheet("color: #00e676; font-size: 12px; font-style: italic;")
        test_layout.addWidget(self.lbl_test_status)
        
        # RESULTADOS DETALLADOS
        self.res_widget = QWidget()
        res_layout = QFormLayout(self.res_widget)
        
        self.lbl_res_level = QLabel("--")
        self.lbl_res_snr = QLabel("-- dB")
        
        res_layout.addRow("Clasificación:", self.lbl_res_level)
        res_layout.addRow("SNR Final:", self.lbl_res_snr)
        
        self.res_widget.setVisible(False)
        test_layout.addWidget(self.res_widget)
        
        test_group.setLayout(test_layout)
        right_panel.addWidget(test_group)
        
        main_h_layout.addLayout(right_panel)
        layout.addLayout(main_h_layout)


        
        layout.addStretch()

    def toggle_ontop(self, checked):
        if checked:
            self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
        else:
            self.setWindowFlags(self.windowFlags() & ~Qt.WindowStaysOnTopHint)
        self.show()

    def refresh_devices(self):
        self.combo_dev.blockSignals(True)
        self.combo_dev.clear()
        
        p = pyaudio.PyAudio()
        try:
            # 1. Intentamos obtener el Loopback por defecto de Windows
            default_loopback = None
            try:
                # Esta función es exclusiva de la librería pyaudiowpatch
                default_info = p.get_default_wasapi_loopback()
                default_loopback = default_info
            except OSError:
                pass # No se pudo determinar el default
            
            target_index = -1
            
            # 2. Llenamos la lista SOLO con Loopbacks
            for i in range(p.get_device_count()):
                d = p.get_device_info_by_index(i)
                
                # Filtro estricto: Debe ser input y tener "loopback" en el nombre
                if d['maxInputChannels'] > 0 and "loopback" in d['name'].lower():
                    self.combo_dev.addItem(f"🔄 {d['name']}", i)
                    
                    # Si es el que Windows dice que es el default, guardamos su índice del combo
                    if default_loopback and d['index'] == default_loopback['index']:
                        target_index = self.combo_dev.count() - 1
            
            # 3. Selección Automática Inteligente
            if target_index >= 0:
                self.combo_dev.setCurrentIndex(target_index)
            elif self.combo_dev.count() > 0:
                self.combo_dev.setCurrentIndex(0) # Si no hay default, elige el primero
            else:
                self.combo_dev.addItem("⚠️ No se detectó Audio del Sistema", -1)

        finally:
            p.terminate()
            self.combo_dev.blockSignals(False)
            
            # Opcional: Iniciar automáticamente si se encontró uno válido
            if self.combo_dev.count() > 0 and self.combo_dev.currentData() >= 0:
                self.start_thread(self.combo_dev.currentIndex())

    def start_thread(self, index):
        data = self.combo_dev.currentData()
        if data == -2: # Show all
             self.combo_dev.clear()
             p = pyaudio.PyAudio()
             for i in range(p.get_device_count()):
                 d = p.get_device_info_by_index(i)
                 if d['maxInputChannels'] > 0: self.combo_dev.addItem(f"🎙️ {d['name']}", i)
             return
        
        if data is not None and data >= 0:
            if self.analysis_thread: self.analysis_thread.stop()
            self.analysis_thread = AudioAnalysisThread(data)
            # self.analysis_thread.level_update.connect(lambda db: self.level_bar.setValue(int(db)))
            self.analysis_thread.phrase_started.connect(self.on_phrase_start)
            self.analysis_thread.phrase_finished.connect(self.on_phrase_end)
            
            self.analysis_thread.test_finished.connect(self.update_certification)
            
            self.analysis_thread.gauge_update.connect(lambda cur, min_v, max_v, act: self.gauge.update_values(cur))
            self.analysis_thread.visual_data.connect(self.visualizer.update_data)
            
            self.analysis_thread.start()

    def on_phrase_start(self):
        self.lbl_style.setText("HABLANDO...")
        self.lbl_style.setStyleSheet("background-color: #444; color: white; font-size: 14px; border-radius: 4px; padding: 5px;")

    def on_phrase_end(self, d):
        dur = d["duration"]
        self.lbl_duration.setText(f"{dur:.2f} s")
        
        label = d.get("label", "---") # Usamos .get por si acaso
        color = d.get("color", "#888")
        
        self.lbl_style.setText(label)
        self.lbl_style.setStyleSheet(f"background-color: {color}; color: #000; font-weight: bold; border-radius: 4px; padding: 5px;")
        
        # ACTUALIZAR RADAR
        if "metrics" in d:
            self.radar.set_metrics(d["metrics"])

    def run_test(self):
        # Deprecated
        pass

    def on_test_progress(self, val, msg):
        pass

    def update_certification(self, res):
        lvl, label, color = get_classification(res['p10'], res['snr'], res['p90'])
        
        self.lbl_res_level.setText(f"{label}")
        self.lbl_res_level.setStyleSheet(f"color: {color}; font-size: 16px; font-weight: bold;")
        
        self.lbl_res_snr.setText(f"SNR: {res['snr']:.1f} dB | Piso: {res['p10']:.1f} dB")
        
        self.lbl_result_mini.setText(f"Ambiente (Última frase): {label}")
        self.lbl_result_mini.setStyleSheet(f"color: {color}; font-weight: bold; font-style: normal;")
        
        self.res_widget.setVisible(True)

    def reset_app(self):
        self.lbl_duration.setText("0.00 s")
        self.lbl_style.setText("---")
        self.lbl_style.setStyleSheet("font-size: 14px; background-color: #333; color: #aaa; border-radius: 4px;")
        
        self.lbl_result_mini.setText("Nivel de Ambiente: PENDIENTE DE TEST")
        self.lbl_result_mini.setStyleSheet("color: #777; font-style: italic; margin-top: 5px;")
        
        self.lbl_test_status.setText("Analizando ventana de 30s en tiempo real...")
        self.res_widget.setVisible(False)
        # self.level_bar.setValue(-90)

        if self.analysis_thread:
            self.analysis_thread.reset_state()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    window = CacatuaWindow()
    window.show()
    sys.exit(app.exec())