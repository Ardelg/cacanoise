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
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QLabel, QProgressBar, QGroupBox, 
                               QComboBox, QPushButton, QCheckBox, QStackedWidget,
                               QFormLayout)
from PySide6.QtCore import QThread, Signal, Qt, QTimer, QRectF, QPointF
from PySide6.QtGui import QPixmap, QFont, QColor, QPalette, QPainter, QPen, QBrush, QConicalGradient, QPolygonF

# --- CONFIGURACIÓN DE CLASIFICACIÓN ---
def calculate_modal_noise_floor(db_array):
    # Calcula el piso de ruido usando histograma (Moda)
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
    Clasificación ajustada a tu escala:
    - SNR Alto (> 35) = LVL 0 (Estudio)
    - SNR Bajo (< 15) = LVL 4 (Calle)
    """
    
    # 1. CASO SILENCIO O MICRÓFONO APAGADO
    if voice_p90 < -50.0: 
        return 0, "SIN SEÑAL / MUTE", "#607d8b"

    # 2. LVL 4: RUIDO EXTREMO (Calle, Viento, Obra)
    # Tus pruebas dieron SNR 14.1 para esto.
    if snr < 11.0:
        return 4, "LVL 4: RUIDOSO (Calle/Viento)", "#d50000" # ROJO

    # 3. LVL 3: RUIDO ALTO (Oficina ruidosa, Café)
    if snr < 20.0:
        return 3, "LVL 3: RUIDO NOTABLE", "#ff6d00" # NARANJA

    # 4. LVL 2: ESTÁNDAR (Ambiente casero normal)
    if snr < 35.0:
        return 2, "LVL 2: ACEPTABLE", "#ffd600" # AMARILLO

    # 5. LVL 1: BUENO (Habitación silenciosa)
    if snr < 50.0:
        return 1, "LVL 1: BUENO", "#64dd17" # VERDE CLARO

    # 6. LVL 0: ESTUDIO (Silencio absoluto de fondo)
    # Tus pruebas dieron SNR 59.4 para esto.
    return 0, "LVL 0: ESTUDIO (Perfecto)", "#00e676" # VERDE NEÓN

class TitanCouncil:
    """
    TitanCouncil v2: Recibe AUDIO PURO (Waveform), no decibeles.
    """
    @staticmethod
    def evaluate(audio_buffer_float, sample_rate):
        # audio_buffer_float debe ser un np.array de float32 (-1.0 a 1.0)
        if len(audio_buffer_float) < 100:
            return -90.0, 0.0, -90.0

        # --- 1. PRE-PROCESAMIENTO ---
        # WebRTC necesita datos puros.
        # Aseguramos que sea mono y float32
        data = np.array(audio_buffer_float, dtype=np.float32)

        # Resampleo obligatorio a 32000Hz (Punto dulce de WebRTC)
        vad_rate = 32000
        if sample_rate != vad_rate:
            # Resampleamos usando librosa
            # Esto puede tardar unos ms, pero es necesario
            data = librosa.resample(data, orig_sr=sample_rate, target_sr=vad_rate)
        
        # Convertir a Int16 (-32768 a 32767) para WebRTC
        pcm_data = (data * 32767).astype(np.int16)

        # --- 2. VAD LOOP ---
        vad = webrtcvad.Vad()
        vad.set_mode(3) # Modo Agresivo

        frame_duration_ms = 30
        frame_len = int(vad_rate * frame_duration_ms / 1000)
        
        voice_energy = []
        noise_energy = []

        # Crear frames
        n_frames = len(pcm_data) // frame_len
        
        for i in range(n_frames):
            start = i * frame_len
            end = start + frame_len
            chunk = pcm_data[start:end]
            
            # Convertir a bytes para la libreria C++
            chunk_bytes = chunk.tobytes()
            
            try:
                # Preguntamos al oraculo: ¿Es voz?
                is_speech = vad.is_speech(chunk_bytes, vad_rate)
                
                # Calculamos la energía EN ESTE FRAME ESPECÍFICO
                # Usamos el float original para mejor precisión matemática
                float_chunk = data[start:end]
                rms = np.sqrt(np.mean(float_chunk**2))
                db_val = 20 * np.log10(rms + 1e-9)
                
                if is_speech:
                    voice_energy.append(db_val)
                else:
                    noise_energy.append(db_val)
            except Exception as e:
                pass 

        # --- 3. RESULTADOS ---
        # Si WebRTC no detectó NADA de voz, es que es puro ruido o silencio
        if len(voice_energy) == 0:
            final_voice = -90.0
            # Si hay ruido detectado, ese es el piso. Si no, todo es silencio.
            final_floor = np.median(noise_energy) if len(noise_energy) > 0 else np.mean(data)
            return final_floor, 0.0, final_voice

        # Cálculo normal
        final_voice = np.percentile(voice_energy, 90) # P90 de la voz
        
        if len(noise_energy) > 0:
            # Filtramos silencio digital (-90) del cálculo de ruido ambiente
            real_noise = [x for x in noise_energy if x > -85.0]
            if len(real_noise) > 0:
                final_floor = np.median(real_noise)
            else:
                final_floor = -90.0 # Era silencio digital puro
        else:
            # Caso raro: Habló sin parar ni respirar 1 segundo
            final_floor = np.min(voice_energy) - 20 # Estimación

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

    def __init__(self, device_idx):
        super().__init__()
        self.idx = device_idx
        self.running = True
        
        self.local_min = 0
        self.local_max = -90
        
        # --- BUFFER 1: DECIBELES (Para Prosodia y Gráficas) ---
        self.pre_roll_db = deque(maxlen=5) 
        self.event_db = [] 
        
        # --- BUFFER 2: AUDIO CRUDO (Para Titan/WebRTC) ---
        # Guardamos los arrays numpy crudos
        self.pre_roll_raw = deque(maxlen=5) 
        self.event_raw = [] 
        
        self.is_speaking = False
        self.speech_start_time = 0
        self.last_speech_time = 0

    def run(self):
        p = pyaudio.PyAudio()
        try:
            info = p.get_device_info_by_index(self.idx)
            rate = int(info["defaultSampleRate"])
            ch = int(info["maxInputChannels"]) or 1
            chunk = int(rate * 0.1) # 100ms
            
            stream = p.open(format=pyaudio.paInt16, channels=ch, rate=rate, input=True, 
                            input_device_index=self.idx, frames_per_buffer=chunk)

            while self.running:
                data = stream.read(chunk, exception_on_overflow=False)
                arr = np.frombuffer(data, dtype=np.int16)
                
                # --- AUDIO CRUDO (FLOAT) ---
                # Esto es lo que necesita Titan (Onda de sonido real)
                arr_float = arr.astype(np.float32) / 32768.0
                
                # --- AUDIO DB (VOLUMEN) ---
                rms = np.sqrt(np.mean(arr_float**2))
                db = 20 * np.log10(rms + 1e-9)
                self.level_update.emit(db)

                # Visuales
                if db > -50: 
                    if db < self.local_min: self.local_min = db
                    if db > self.local_max: self.local_max = db
                self.gauge_update.emit(db, self.local_min, self.local_max, self.is_speaking)
                
                curr_time = time.time()

                # --- MÁQUINA DE ESTADOS ---
                if db > -55: 
                    self.last_speech_time = curr_time
                    
                    if not self.is_speaking:
                        self.is_speaking = True
                        self.speech_start_time = curr_time
                        
                        # Volcamos Pre-Roll de AMBOS buffers
                        self.event_db = list(self.pre_roll_db)
                        self.event_raw = list(self.pre_roll_raw) # Importante
                        
                        self.phrase_started.emit()

                    # Grabamos en AMBOS buffers
                    self.event_db.append(db)
                    self.event_raw.append(arr_float) # Guardamos el array de audio

                else:
                    if self.is_speaking:
                        # Grabamos la cola del silencio
                        self.event_db.append(db)
                        self.event_raw.append(arr_float)
                        
                        if (curr_time - self.last_speech_time) > 1.5:
                            duration = self.last_speech_time - self.speech_start_time
                            
                            if duration > 0.5:
                                # 1. PREPARAMOS EL AUDIO RAW PARA TITAN
                                # Concatenamos todos los chunks pequeños en un solo audio largo
                                if len(self.event_raw) > 0:
                                    full_audio_chunk = np.concatenate(self.event_raw)
                                    
                                    # LLAMADA A TITAN con AUDIO REAL
                                    piso, snr, voz = TitanCouncil.evaluate(full_audio_chunk, sample_rate=rate)

                                    self.test_finished.emit({
                                        "p10": piso,
                                        "p90": voz,
                                        "snr": snr,
                                        "stability": 0,
                                        "duration": duration
                                    })
                                
                                # 2. LLAMADA A STYLE CON BUFFER DB (Sigue igual)
                                prosody_data = StyleCouncil.evaluate(self.event_db)
                                prosody_data["duration"] = duration
                                self.phrase_finished.emit(prosody_data)
                            
                            # RESET
                            self.is_speaking = False
                            self.event_db = []
                            self.event_raw = [] # Limpieza
                            self.local_min = 0 
                            self.local_max = -90

                    else:
                        # MODO ESPERA: Alimentamos Pre-Rolls
                        self.pre_roll_db.append(db)
                        self.pre_roll_raw.append(arr_float) # Guardamos audio crudo

        except Exception as e: print(f"Error Thread: {e}")
        finally: p.terminate()

    def stop(self):
        self.running = False
        self.wait()

    def reset_state(self):
        self.event_db = []
        self.pre_roll_db.clear()
        self.event_raw = []      # Reset raw
        self.pre_roll_raw.clear() # Reset raw
        self.is_speaking = False

class StyleCouncil:
    # Sistema de votación para determinar naturalidad (Dinámica, Ritmo, Fluidez)
    @staticmethod
    def evaluate(db_values):
        if not db_values or len(db_values) < 5:
            return {"style": "Insuficiente", "color": "#888", "details": "Muy corto"}

        arr = np.array(db_values)
        
        # 1. Dinámica (Desviación Estándar)
        std_dev = np.std(arr)
        score_dynamics = 2 if std_dev > 4.5 else (1 if std_dev > 2.5 else 0)
        
        # 2. Ritmo (Coeficiente de Variación de Picos)
        height_thresh = np.percentile(arr, 20) + 5 
        peaks, _ = signal.find_peaks(arr, height=height_thresh, distance=2)
        
        rhythm_cv = 0
        if len(peaks) > 2:
            intervals = np.diff(peaks)
            if np.mean(intervals) > 0:
                rhythm_cv = np.std(intervals) / np.mean(intervals)
        
        score_rhythm = 2 if rhythm_cv > 0.35 else (1 if rhythm_cv > 0.18 else 0)
        
        # 3. Fluidez (Ratio de relleno)
        p80 = np.percentile(arr, 80)
        threshold_silence = p80 - 15 
        active_samples = np.sum(arr > threshold_silence)
        fill_ratio = active_samples / len(arr)
        
        score_flow = 2 if fill_ratio < 0.75 else (1 if fill_ratio < 0.90 else 0)

        # Veredicto
        total_score = score_dynamics + score_rhythm + score_flow
        
        if total_score <= 1:
            final_style = "🤖 ROBÓTICO / PLANO"
            color = "#ff1744"
            reason = "Tono monótono y ritmo fijo"
        elif total_score <= 3:
            final_style = "📖 LEÍDO / FORMAL"
            color = "#ffd740"
            reason = "Buena dicción, ritmo pautado"
        else:
            final_style = "🗣️ NATURAL / ESPONTÁNEO"
            color = "#00e676"
            reason = "Variación tonal y rítmica alta"

        debug_info = f"DYN({score_dynamics}) RYTH({score_rhythm}) FLOW({score_flow})"
        
        return {
            "style": final_style,
            "style_color": color,
            "reason": reason,
            "debug": debug_info,
            "metrics": f"CV:{rhythm_cv:.2f} SD:{std_dev:.1f}"
        }




class GaugeWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(250, 200) 
        self.current_db = -90
        self.min_trace = -90
        self.max_trace = -90
        self.is_active = False

    def update_values(self, current, min_val, max_val, active):
        self.current_db = max(-90, min(0, current))
        self.min_trace = max(-90, min(0, min_val))
        self.max_trace = max(-90, min(0, max_val))
        self.is_active = active
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w, h = self.width(), self.height()
        center = QPointF(w / 2, h * 0.85) # Centro un poco más abajo
        radius = min(w, h * 2) / 2 - 25

        # --- 1. FONDO OSCURO (Sphere effect) ---
        grad_bg = QConicalGradient(center, -90)
        grad_bg.setColorAt(0, QColor("#1a1a1a"))
        grad_bg.setColorAt(1, QColor("#2a2a2a"))
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#111"))
        painter.drawPie(QRectF(center.x() - radius, center.y() - radius, radius * 2, radius * 2), 0, 180 * 16)

        # --- 2. ESCALA Y TICKS (Las rayitas) ---
        # Vamos de -90dB (180 grados) a 0dB (0 grados)
        painter.save()
        painter.translate(center)
        
        total_ticks = 45 # Cantidad de rayitas
        for i in range(total_ticks + 1):
            val_db = -90 + (90 * (i / total_ticks)) # RE-ADDED MISSING LINE
            
            # CAMBIO AQUÍ: Multiplica por 2 positivo.
            # -90db * 2 = -180 (Gira antihorario hasta la izquierda)
            # -45db * 2 = -90 (Gira antihorario hasta arriba)
            angle = 2 * val_db 
            
            painter.save()
            painter.rotate(angle) # Ahora rotará hacia arriba
            
            # Color del Tick según severidad
            if val_db > -15: tick_color = QColor("#ff1744") # Rojo Crítico
            elif val_db > -35: tick_color = QColor("#ff9100") # Naranja
            elif val_db > -50: tick_color = QColor("#ffea00") # Amarillo
            else: tick_color = QColor("#00e676") # Verde
            
            # Si es tick principal (cada 10dB) es más grueso
            is_major = (i % 5 == 0)
            length = 15 if is_major else 8
            width = 3 if is_major else 1
            
            painter.setPen(QPen(tick_color, width))
            painter.drawLine(int(radius - length), 0, int(radius), 0)
            
            # Texto de números
            if is_major:
                font = QFont("Segoe UI", 8, QFont.Bold)
                painter.setFont(font)
                painter.setPen(QColor("#888"))
                # Dibujar texto un poco más adentro
                text_radius = radius - 30
                # Hay que des-rotar el texto para que se lea recto, o dejarlo rotado.
                # Para simplificar, lo dibujamos rotado o usamos coordenadas polares.
                # Simplemente dibujamos una linea guia invisible si quisieramos texto.
            
            painter.restore()
        painter.restore()

        # --- 3. EL RASTRO (Sombra dinámica) ---
        if self.is_active and self.max_trace > self.min_trace:
            rect = QRectF(center.x() - (radius-5), center.y() - (radius-5), (radius-5) * 2, (radius-5) * 2)
            angle_min = -2 * self.min_trace
            angle_max = -2 * self.max_trace
            
            # Invertimos lógica de ángulos para drawArc
            # 180 deg es izquierda (-90db). 0 deg es derecha (0db).
            # startAngle en Qt es 1/16th de grado.
            start = int(angle_min * 16) 
            span = int((angle_max - angle_min) * 16)
            
            pen_trace = QPen(QColor(255, 255, 255, 40), 10) # Sombra sutil
            pen_trace.setCapStyle(Qt.FlatCap)
            painter.setPen(pen_trace)
            painter.drawArc(rect, start, span)

        # --- 4. LA AGUJA (Fina y elegante) ---
        painter.save()
        painter.translate(center)
        # CAMBIO AQUÍ: Usa multiplicación positiva
        painter.rotate(2 * self.current_db) 
        
        painter.setBrush(QColor("#ff3d00")) # Naranja deportivo
        painter.setPen(Qt.NoPen)
        
        # Poligono de aguja afilada
        needle = QPolygonF([
            QPointF(0, -2), 
            QPointF(radius - 5, 0), # Punta
            QPointF(0, 2),
            QPointF(-10, 0) # Contrapeso
        ])
        painter.drawPolygon(needle)
        
        # Circulo central (tapa de la aguja)
        painter.setBrush(QColor("#222"))
        painter.setPen(QPen(QColor("#444"), 2))
        painter.drawEllipse(QPointF(0,0), 8, 8)
        
        painter.restore()

        # --- 5. TEXTO DIGITAL ---
        painter.setPen(QColor("white"))
        painter.setFont(QFont("Consolas", 22, QFont.Bold))
        text_rect = QRectF(center.x() - 60, center.y() - 60, 120, 40)
        painter.drawText(text_rect, Qt.AlignCenter, f"{self.current_db:.1f}")
        
        painter.setFont(QFont("Segoe UI", 10))
        painter.setPen(QColor("#888"))
        unit_rect = QRectF(center.x() - 60, center.y() - 25, 120, 20)
        painter.drawText(unit_rect, Qt.AlignCenter, "dB FS")

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
        
        # Checkbox Siempre Visible
        self.chk_ontop = QCheckBox("Siempre Visible")
        self.chk_ontop.setStyleSheet("color: #aaa; margin-right: 5px;")
        self.chk_ontop.toggled.connect(self.toggle_ontop)
        header.addWidget(self.chk_ontop)

        # Botón RESET
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
        self.refresh_devices()
        self.combo_dev.currentIndexChanged.connect(self.start_thread)
        layout.addWidget(self.combo_dev)

        # Inserta el Manómetro
        self.gauge = GaugeWidget()
        # Puedes centrarlo en un layout
        layout.addWidget(self.gauge, alignment=Qt.AlignCenter)

        if self.combo_dev.count() > 0:
            self.start_thread(0)

        # Meter
        self.level_bar = QProgressBar()
        self.level_bar.setRange(-90, 0)
        self.level_bar.setTextVisible(False)
        self.level_bar.setFixedHeight(20)
        self.level_bar.setStyleSheet("""
            QProgressBar { background-color: #222; border: 1px solid #444; border-radius: 10px; }
            QProgressBar::chunk { background-color: #00e676; border-radius: 10px; }
        """)
        layout.addWidget(self.level_bar)

        # --- SECTION: VOICE DURATION & QUICK STATUS ---
        dur_group = QGroupBox("Análisis de Voz (Prosodia)")
        dur_layout = QVBoxLayout()
        
        hbox = QHBoxLayout()
        # Tiempo grande
        self.lbl_duration = QLabel("0.00 s")
        self.lbl_duration.setStyleSheet("font-size: 40px; font-weight: bold; color: white;")
        hbox.addWidget(self.lbl_duration)
        
        # Etiqueta de Estilo (Robótico vs Natural)
        self.lbl_style = QLabel("---")
        self.lbl_style.setAlignment(Qt.AlignCenter)
        self.lbl_style.setFixedSize(200, 40)
        self.lbl_style.setStyleSheet("font-size: 14px; background-color: #333; color: #aaa; border-radius: 4px;")
        hbox.addWidget(self.lbl_style)
        dur_layout.addLayout(hbox)
        
        # Detalles técnicos pequeños
        self.lbl_prosody_details = QLabel("Dinamismo: -- | Ritmo (CV): --")
        self.lbl_prosody_details.setStyleSheet("color: #aaa; margin-top: 5px;")
        dur_layout.addWidget(self.lbl_prosody_details)

        # AQUÍ AÑADIMOS EL NIVEL DE RUIDO (Resultado del último test)
        self.lbl_result_mini = QLabel("Nivel de Ambiente: PENDIENTE DE TEST")
        self.lbl_result_mini.setAlignment(Qt.AlignCenter)
        self.lbl_result_mini.setStyleSheet("color: #777; font-style: italic; margin-top: 5px;")
        dur_layout.addWidget(self.lbl_result_mini)

        dur_group.setLayout(dur_layout)
        layout.addWidget(dur_group)

        
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
        layout.addWidget(test_group)
        
        layout.addStretch()

    def toggle_ontop(self, checked):
        if checked:
            self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
        else:
            self.setWindowFlags(self.windowFlags() & ~Qt.WindowStaysOnTopHint)
        self.show()

    def refresh_devices(self):
        self.combo_dev.clear()
        p = pyaudio.PyAudio()
        try:
            for i in range(p.get_device_count()):
                d = p.get_device_info_by_index(i)
                # Filtramos por Loopback/Stereo Mix per preferencia del usuario
                if d['maxInputChannels'] > 0 and ("loopback" in d['name'].lower() or "stereomix" in d['name'].lower()):
                    self.combo_dev.addItem(f"🔄 {d['name']}", i)
            
            if self.combo_dev.count() == 0:
                self.combo_dev.addItem("⚠️ Mostrar Todos (No Loopback Encontrado)", -2)
        finally:
            p.terminate()

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
            self.analysis_thread.level_update.connect(lambda db: self.level_bar.setValue(int(db)))
            self.analysis_thread.phrase_started.connect(self.on_phrase_start)
            self.analysis_thread.phrase_finished.connect(self.on_phrase_end)
            
            # 30s Connections (Continuous)
            # 30s Connections (Continuous)
            self.analysis_thread.test_finished.connect(self.update_certification)
            
            # CONEXIÓN NUEVA
            self.analysis_thread.gauge_update.connect(self.gauge.update_values)
            
            self.analysis_thread.start()

    def on_phrase_start(self):
        self.lbl_style.setText("HABLANDO...")
        self.lbl_style.setStyleSheet("background-color: #444; color: white; font-size: 14px; border-radius: 4px; padding: 5px;")

    def on_phrase_end(self, data):
        # 1. Duración
        dur = data["duration"]
        self.lbl_duration.setText(f"{dur:.2f} s")
        
        # 2. Estilo y Color
        style = data["style"]
        color = data["style_color"]
        self.lbl_style.setText(style)
        self.lbl_style.setStyleSheet(f"background-color: {color}; color: #000; font-weight: bold; border-radius: 4px; padding: 5px;")
        
        # AQUI MOSTRAMOS LA RAZÓN Y LOS VOTOS DE LOS JUECES
        # data['reason'] te dice "Por qué" en humano.
        # data['metrics'] te da los números técnicos.
        self.lbl_prosody_details.setText(
            f"{data['reason']} | {data['metrics']}"
        )

    def run_test(self):
        # Deprecated
        pass

    def on_test_progress(self, val, msg):
        pass

    def update_certification(self, res):
        # CAMBIO: Ahora pasamos 3 argumentos: Piso, SNR y el Volumen de Voz (p90)
        lvl, label, color = get_classification(res['p10'], res['snr'], res['p90'])
        
        # Actualizamos el UI con los resultados
        self.lbl_res_level.setText(f"{label}")
        self.lbl_res_level.setStyleSheet(f"color: {color}; font-size: 16px; font-weight: bold;")
        
        # Usamos lbl_res_snr para mostrar los detalles técnicos como pidió el usuario
        # Aunque el usuario sugirió lbl_snr y lbl_floor separados, adaptamos al UI actual que es un FormLayout
        self.lbl_res_snr.setText(f"SNR: {res['snr']:.1f} dB | Piso: {res['p10']:.1f} dB")
        
        # Actualizamos el label mini
        self.lbl_result_mini.setText(f"Ambiente (Última frase): {label}")
        self.lbl_result_mini.setStyleSheet(f"color: {color}; font-weight: bold; font-style: normal;")
        
        self.res_widget.setVisible(True)

    def reset_app(self):
        # 1. Reset UI Elements
        self.lbl_duration.setText("0.00 s")
        self.lbl_style.setText("---")
        self.lbl_style.setStyleSheet("font-size: 14px; background-color: #333; color: #aaa; border-radius: 4px;")
        
        self.lbl_result_mini.setText("Nivel de Ambiente: PENDIENTE DE TEST")
        self.lbl_result_mini.setStyleSheet("color: #777; font-style: italic; margin-top: 5px;")
        
        self.lbl_test_status.setText("Analizando ventana de 30s en tiempo real...")
        self.res_widget.setVisible(False)
        self.level_bar.setValue(-90)

        # 2. Reset Thread State
        if self.analysis_thread:
            self.analysis_thread.reset_state()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    window = CacatuaWindow()
    window.show()
    sys.exit(app.exec())