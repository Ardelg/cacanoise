import sys
import os
import time
import numpy as np
import pyaudiowpatch as pyaudio
from collections import deque
from scipy import signal
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QLabel, QProgressBar, QGroupBox, 
                               QComboBox, QPushButton, QCheckBox, QStackedWidget,
                               QFormLayout)
from PySide6.QtCore import QThread, Signal, Qt, QTimer
from PySide6.QtGui import QPixmap, QFont, QColor, QPalette

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

def get_classification(noise_floor, snr, stability):
    # Clasificación robusta basada en EDE/Broadcast
    
    # 1. CRITICAL
    if snr < 6.0: return 4, "4. CRITICAL (Voz inaudible)", "#d50000"
    if noise_floor > -35.0: return 4, "4. CRITICAL (Saturado)", "#d50000"

    # 2. HIGH RISK
    is_noisy_floor = noise_floor > -43.0 
    is_poor_snr = snr < 15.0 
    
    if is_noisy_floor or is_poor_snr:
        reason = "Ruidoso" if is_noisy_floor else "Voz baja"
        return 3, f"3. HIGH ({reason})", "#ff6d00"

    # 3. QUALITY ZONES
    if noise_floor > -55.0:
        detail = "Constante" if stability < 4.0 else "Irregular"
        return 2, f"2. MODERATE ({detail})", "#ffd600"

    if snr > 25.0:
        return 1, "1. LOW (Studio Quality)", "#00e676"
    else:
        return 2, "2. MODERATE (Voz suave)", "#ffd600"

    if noise_floor < -75.0:
        return 0, "0. SILENT (Vacío)", "#00bfa5"

class NoiseCouncil:
    # Sistema de votación de 3 rutas para determinar el ruido de fondo real.
    @staticmethod
    def evaluate(audio_buffer):
        if len(audio_buffer) == 0: return -90.0, {}

        data = np.array(audio_buffer)
        data = data[data > -100] 
        if len(data) < 10: return -90.0, {}

        # 1. Juez Modal (Estadístico)
        try:
            lower_half = data[data < np.mean(data)]
            if len(lower_half) > 0:
                hist, bins = np.histogram(lower_half, bins=int(np.ptp(lower_half)) + 1)
                max_bin = np.argmax(hist)
                vote_modal = (bins[max_bin] + bins[max_bin+1]) / 2
            else:
                vote_modal = np.percentile(data, 10)
        except:
            vote_modal = np.percentile(data, 10)

        # 2. Juez VAD (Segregador)
        try:
            p95_voice = np.percentile(data, 95)
            threshold = p95_voice - 20.0
            noise_samples = data[data < threshold]
            
            if len(noise_samples) > len(data) * 0.1:
                vote_vad = np.mean(noise_samples)
            else:
                vote_vad = np.percentile(data, 5)
        except:
            vote_vad = -90.0

        # 3. Juez Pesimista (P5)
        vote_p5 = np.percentile(data, 5)

        # Consenso
        judges = [vote_modal, vote_vad, vote_p5]
        consensus = np.median(judges)
        
        # Ajuste de seguridad
        if vote_modal > consensus:
            final_noise = (consensus + vote_modal) / 2
        else:
            final_noise = consensus

        return final_noise, {"Moda": vote_modal, "VAD": vote_vad, "P5": vote_p5}

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

class AudioAnalysisThread(QThread):
    level_update = Signal(float) 
    test_finished = Signal(dict) # Emitirá el veredicto al final de la frase
    phrase_started = Signal()
    # CAMBIAMOS ESTO: De Signal(float) a Signal(dict) para enviar métricas complejas
    phrase_finished = Signal(dict)

    def __init__(self, device_idx):
        super().__init__()
        self.idx = device_idx
        self.running = True
        
        # MEMORIA CONTINUA (30 segundos)
        self.noise_window = deque(maxlen=300) 
        
        # Buffers de voz
        self.phrase_buffer = [] 
        self.is_speaking = False
        self.speech_start_time = 0
        self.last_speech_time = 0

    def calculate_smart_noise_floor(self):
        """
        Wrapper que ahora usa la lógica de Histograma Modal.
        Mantiene el nombre para compatibilidad, pero usa la función robusta.
        """
        if not self.noise_window:
            return -90.0
        return calculate_modal_noise_floor(np.array(list(self.noise_window)))


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
                arr_float = arr.astype(np.float32) / 32768.0
                
                rms = np.sqrt(np.mean(arr_float**2))
                db = 20 * np.log10(rms + 1e-9)
                self.level_update.emit(db)

                # ALIMENTACIÓN CONTINUA: El sistema siempre escucha el fondo
                self.noise_window.append(db)
                
                curr_time = time.time()

                # --- DETECCIÓN DE VOZ ---
                if db > -45: 
                    if not self.is_speaking:
                        self.is_speaking = True
                        self.speech_start_time = curr_time
                        self.phrase_buffer = []

                        # Borra la memoria de los audios anteriores.
                        # Así el análisis se hará SOLO con los datos del audio actual.
                        self.noise_window.clear() 

                        self.phrase_started.emit()
                    self.last_speech_time = curr_time
                    self.phrase_buffer.append(db)

                elif self.is_speaking and (curr_time - self.last_speech_time) > 2.0:
                    # Fin de frase detectado
                    duration = self.last_speech_time - self.speech_start_time
                    
                    if duration > 0.5:
                        # 1. Obtenemos toda la ventana de ruido reciente
                        raw_noise_data = list(self.noise_window)
                        
                        # 2. Invocamos al Consejo
                        final_floor, votes = NoiseCouncil.evaluate(raw_noise_data)
                        
                        # 3. Calculamos métricas de voz sobre el buffer actual
                        voice_samples = np.array(self.phrase_buffer)
                        real_p90 = np.percentile(voice_samples, 90) if len(voice_samples) > 0 else -60
                        
                        # 4. Resultado final robusto
                        snr_final = real_p90 - final_floor
                        stability = np.std(raw_noise_data)

                        # Imprimimos debug para que tú (Ariel) veas qué votó cada juez
                        print(f"JUECES: Moda={votes['Moda']:.1f} | VAD={votes['VAD']:.1f} | P5={votes['P5']:.1f} -> FINAL: {final_floor:.1f}")

                        self.test_finished.emit({
                            "p10": final_floor, # Le pasamos el consenso como si fuera el p10
                            "p90": real_p90,
                            "snr": snr_final,
                            "stability": stability,
                            "duration": duration
                        })
                        # --- NUEVO BLOQUE DE PROSODIA ---
                        # Ya no pasamos threshold fijo, el analizador lo calcula solo.
                        prosody_data = StyleCouncil.evaluate(self.phrase_buffer)
                        
                        prosody_data["duration"] = duration
                        self.phrase_finished.emit(prosody_data)
                    
                    self.is_speaking = False
                    self.event_env_buffer = []

        except Exception as e: print(f"Error: {e}")
        finally: p.terminate()

    def stop(self):
        self.running = False
        self.wait()

    def reset_state(self):
        self.phrase_buffer = []
        self.noise_window.clear()
        self.is_speaking = False


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
            self.analysis_thread.test_finished.connect(self.update_certification)
            
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
        lvl, label, color = get_classification(res['p10'], res['snr'], res['stability'])
        
        # Actualizamos el UI con los resultados de LA FRASE que acaba de terminar
        self.lbl_res_level.setText(f"{label}")
        self.lbl_res_level.setStyleSheet(f"color: {color}; font-size: 18px; font-weight: bold;")
        
        # Mostramos que el cálculo fue sobre la duración real de la voz
        self.lbl_res_snr.setText(
            f"Certificado en {res['duration']:.1f}s | SNR: {res['snr']:.1f}dB | Piso: {res['p10']:.1f}dB"
        )
        
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