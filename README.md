# Cacatua Noise

**Cacatua Noise** es una herramienta avanzada de análisis de audio diseñada para monitorear y certificar la calidad del entorno sonoro en tiempo real. Utiliza algoritmos de procesamiento de señal y estadísticas ponderadas ("Noise Council") para ofrecer métricas precisas sobre el ruido de fondo, la relación señal-ruido (SNR) y la prosodia de la voz.

---

## ⚠️ Aviso Importante (Disclaimer)

**Esta herramienta tiene fines exclusivamente de apoyo técnico y métrico.**

Los resultados mostrados por el software **NO representan una evaluación definitiva** ni reemplazan el criterio profesional. La evaluación definitiva de la calidad del audio se basa indispensablemente en **escuchar atentamente** y utilizar la percepción humana para distinguir con precisión los matices entre la voz del usuario y el ruido de fondo.

Esta aplicación sirve como una guía cuantitativa para ayudar a identificar problemas, pero el oído humano es el juez final.

---

## Requisitos Previos

Para utilizar Cacatua Noise, es necesario tener instalado **Python** en su sistema operativo.

*   📥 **Descargar Python**: [https://www.python.org/downloads/](https://www.python.org/downloads/)

*Asegúrese de marcar la casilla "Add Python to PATH" durante la instalación.*

---

## Instrucciones de Instalación y Uso

El proyecto está diseñado para ser "Plug & Play" mediante el script de automatización incluido. No es necesario abrir terminales ni configurar entornos manualmente.

### Pasos para iniciar:

1.  Descargue o clone este repositorio en su computadora.
2.  Ubique el archivo **`cacatuanoise.bat`** en la carpeta principal.
3.  Haga **doble clic** sobre `cacatuanoise.bat`.

### ¿Qué hace el script?
Automáticamente realizará las siguientes tareas la primera vez que se ejecute:
1.  Verificará si Python está instalado.
2.  Creará un entorno virtual aislado (`.venv`) para no afectar su sistema.
3.  Instalará todas las librerías necesarias (`requirements.txt`).
4.  Iniciará la aplicación **Cacatua Noise**.

Para ejecuciones posteriores, el script detectará que todo está listo y abrirá la aplicación inmediatamente.

### Configuración de Audio (Importante)

El programa funciona como un **"espía de audio"** pasivo. No interviene, modifica ni se conecta directamente a otras aplicaciones (Google Meet, Zoom, etc.). Simplemente escucha lo que sale por tus parlantes o auriculares.

*   **Selección de Fuente**: En el menú desplegable "Fuente de Sonido", debes elegir el dispositivo **por donde TÚ estás escuchando el audio**.
    *   🎧 Si estás usando **auriculares**, selecciona tus auriculares en la lista.
    *   🔊 Si estás usando **parlantes**, selecciona los parlantes.

**Nota**: La herramienta usa la función "Loopback" para capturar el audio del sistema tal cual lo escuchas tú.

---

## Solución de Problemas

*   **Si el archivo .bat se cierra inmediatamente**: Intente ejecutarlo desde una ventana de CMD para ver el error. Generalmente se debe a que Python no está instalado o no se agregó al PATH.
*   **Si falta alguna librería**: Puede forzar la reinstalación ejecutando el script desde la terminal con el comando: `cacatuanoise.bat --reinstall`
