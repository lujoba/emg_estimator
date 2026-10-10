"""Constantes do protocolo experimental (Prosthetic Control - October 2024)."""

# task_id 1..8 nos nomes dos arquivos -> índice 0..7
TASK_DICT = {
    "1": "Index", "2": "Middle", "3": "RingPinky", "4": "Thumb",
    "5": "I.Pinch", "6": "M.Pinch", "7": "Tripod", "8": "Power",
}
TASKS = list(TASK_DICT.values())

FINGERS = ["Polegar", "Indicador", "Medio", "Anelar", "Minimo"]
N_FINGERS = len(FINGERS)

N_CHANNELS = 128      # HD-sEMG monopolar
FS_EMG = 2048.0       # Hz (antes do downsample para 1024 Hz)
FS_ANGLES = 100.0     # Hz (cinemática)

WINDOW_SIZE = 100     # amostras por janela
WINDOW_STEP = 10
