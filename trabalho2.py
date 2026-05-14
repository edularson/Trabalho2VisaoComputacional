import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox
import threading
import os
import wave
import struct
import math
import time
import random

try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    MEDIAPIPE_OK = True
except (ImportError, AttributeError):
    MEDIAPIPE_OK = False

try:
    import pygame
    pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
    PYGAME_OK = True
except ImportError:
    PYGAME_OK = False

#  GERAÇÃO DE SONS
NOTES = {
    "do":  261.63,
    "re":  293.66,
    "mi":  329.63,
    "fa":  349.23,
    "sol": 392.00,
}

SOUND_DIR = "sounds"

def gerar_wav(filename, freq, duration=0.8, sample_rate=44100):
    os.makedirs(SOUND_DIR, exist_ok=True)
    path = os.path.join(SOUND_DIR, filename)
    if os.path.exists(path):
        return path

    n = int(sample_rate * duration)
    with wave.open(path, "w") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)

        for i in range(n):
            t = i / sample_rate

            vibrato = 1 + 0.005 * math.sin(2 * math.pi * 5 * t)
            f_vibrato = freq * vibrato

            onda = (math.sin(2 * math.pi * f_vibrato * t) +
                    0.3 * math.sin(2 * math.pi * 2 * f_vibrato * t) +
                    0.1 * math.sin(2 * math.pi * 3 * f_vibrato * t))
            onda /= 1.4

            ataque_sopro = 0
            if t < 0.1:
                ruido = random.uniform(-1, 1)
                intensidade_sopro = (0.1 - t) * 5
                ataque_sopro = ruido * intensidade_sopro * 0.2

            env = min(t / 0.1, 1.0) * max(1.0 - (t - (duration - 0.2)) / 0.2, 0.0)

            val = 32767 * env * (onda + ataque_sopro)
            sample = int(max(-32768, min(32767, val)))
            f.writeframes(struct.pack("<h", sample))
    return path

def gerar_todos_sons():
    sons = {}
    for nome, freq in NOTES.items():
        path = gerar_wav(f"{nome}.wav", freq)
        if PYGAME_OK:
            sons[nome] = pygame.mixer.Sound(path)
    return sons


#  DETECTOR ARUCO
def get_aruco_detector():
    dicionario = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    params = cv2.aruco.DetectorParameters()
    return cv2.aruco.ArucoDetector(dicionario, params)

DETECTOR = get_aruco_detector()
MARKER_SIZE_CM = 5.0

FOCAL = 600.0

CAMERA_MATRIX = np.array([
    [FOCAL, 0,     640.0],
    [0,     FOCAL, 360.0],
    [0,     0,     1.0  ]
], dtype=np.float64)
DIST_COEFFS = np.zeros((4, 1))

HALF = MARKER_SIZE_CM / 2
OBJ_POINTS = np.array([
    [-HALF,  HALF, 0],
    [ HALF,  HALF, 0],
    [ HALF, -HALF, 0],
    [-HALF, -HALF, 0],
], dtype=np.float64)

def desenha_cubo(frame, rvec, tvec):
    h = MARKER_SIZE_CM

    vertices_3d = np.float32([
        [-HALF, -HALF,  0],
        [ HALF, -HALF,  0],
        [ HALF,  HALF,  0],
        [-HALF,  HALF,  0],
        [-HALF, -HALF,  h],
        [ HALF, -HALF,  h],
        [ HALF,  HALF,  h],
        [-HALF,  HALF,  h],
    ])

    pts2d, _ = cv2.projectPoints(vertices_3d, rvec, tvec, CAMERA_MATRIX, DIST_COEFFS)
    pts = pts2d.reshape(-1, 2).astype(int)

    base = pts[:4]
    topo = pts[4:]

    # Arestas da base (verde)
    for i in range(4):
        cv2.line(frame, tuple(base[i]), tuple(base[(i + 1) % 4]), (0, 180, 0), 2)

    # Arestas do topo (amarelo)
    for i in range(4):
        cv2.line(frame, tuple(topo[i]), tuple(topo[(i + 1) % 4]), (255, 220, 0), 2)

    # Arestas verticais (branco)
    for i in range(4):
        cv2.line(frame, tuple(base[i]), tuple(topo[i]), (255, 255, 255), 2)

    # Faces semitransparentes
    overlay = frame.copy()
    faces = [
        ([topo[0], topo[1], base[1], base[0]], (200, 230, 255)),
        ([topo[1], topo[2], base[2], base[1]], (180, 210, 240)),
        ([topo[0], topo[1], topo[2], topo[3]], (160, 200, 255)),
    ]
    for pontos, cor in faces:
        cv2.fillPoly(overlay, [np.array(pontos)], cor)
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

def distancia_aruco(frame):
    cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = DETECTOR.detectMarkers(cinza)

    if ids is None or len(ids) < 1:
        cv2.putText(frame, "Mostre um marcador ArUco", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return frame

    cv2.aruco.drawDetectedMarkers(frame, corners, ids)

    for i, corner in enumerate(corners):
        img_points = corner[0].astype(np.float64)

        ok, rvec, tvec = cv2.solvePnP(
            OBJ_POINTS, img_points, CAMERA_MATRIX, DIST_COEFFS,
            flags=cv2.SOLVEPNP_IPPE_SQUARE
        )

        if not ok:
            continue

        rvec = rvec.flatten()
        tvec = tvec.flatten()

        c = corner[0]
        cx = int(c[:, 0].mean())
        cy = int(c[:, 1].mean())

        dist_z = float(tvec[2])
        mid = ids[i][0]
        cv2.putText(frame, f"ID{mid} | {dist_z:.1f}cm", (cx - 40, cy - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 220, 0), 2)

        desenha_cubo(frame, rvec, tvec)

    return frame


#  MÓDULO 2 – OCARINA

FURO_MAP = {1: "do", 2: "re", 3: "mi", 4: "fa", 5: "sol"}
FURO_CORES = {
    "do":  (255,  80,  80),
    "re":  (255, 180,  50),
    "mi":  (100, 220,  80),
    "fa":  (50,  180, 255),
    "sol": (180,  80, 255),
}

_ultima_nota = {}

def tocar_nota(nome, sons):
    if not PYGAME_OK or nome not in sons:
        return
    agora = time.time()
    if agora - _ultima_nota.get(nome, 0) > 0.5:
        sons[nome].play()
        _ultima_nota[nome] = agora

def ocarina_frame(frame, sons):
    cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = DETECTOR.detectMarkers(cinza)

    if ids is None:
        cv2.putText(frame, "Sem marcadores detectados", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return frame

    detected_ids = set(ids.flatten())
    centros = {}
    for i, corner in enumerate(corners):
        mid = ids[i][0]
        c = corner[0]
        cx = int(c[:, 0].mean())
        cy = int(c[:, 1].mean())
        centros[mid] = (cx, cy)

    if 0 in centros:
        rx, ry = centros[0]
        overlay = frame.copy()
        cv2.ellipse(overlay, (rx, ry), (90, 55), 0, 0, 360, (180, 120, 60), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.ellipse(frame, (rx, ry), (90, 55), 0, 0, 360, (120, 80, 30), 2)
        cv2.putText(frame, "OCARINA", (rx - 35, ry + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 220, 150), 1)

    for fid, nota in FURO_MAP.items():
        cor = FURO_CORES[nota]
        coberto = fid not in detected_ids

        cx, cy = centros.get(fid, (None, None))

        if cx is not None:
            estado_cor = (0, 0, 0) if coberto else cor
            cv2.circle(frame, (cx, cy), 18, estado_cor, -1)
            cv2.circle(frame, (cx, cy), 18, (255, 255, 255), 1)
            cv2.putText(frame, nota, (cx - 10, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        if coberto:
            tocar_nota(nota, sons)

    y0 = frame.shape[0] - 30
    cv2.putText(frame, "Cubra os marcadores para tocar", (10, y0),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    return frame


#  MÓDULO 3 – AR SEM MARCADORES

def desenhar_objeto_3d(frame, cx, cy, raio=60):
    t = time.time()
    angulo = t * 1.5

    phi = (1 + math.sqrt(5)) / 2
    verts_raw = [
        (0, 1, phi), (0, -1, phi), (0, 1, -phi), (0, -1, -phi),
        (1, phi, 0), (-1, phi, 0), (1, -phi, 0), (-1, -phi, 0),
        (phi, 0, 1), (-phi, 0, 1), (phi, 0, -1), (-phi, 0, -1),
    ]
    faces = [
        (0,1,8),(0,8,4),(0,4,5),(0,5,9),(0,9,1),
        (1,6,8),(8,6,10),(8,10,4),(4,10,2),(4,2,5),
        (5,2,11),(5,11,9),(9,11,7),(9,7,1),(1,7,6),
        (3,6,7),(3,7,11),(3,11,2),(3,2,10),(3,10,6),
    ]

    cos_a, sin_a = math.cos(angulo), math.sin(angulo)

    def rot_y(x, y, z):
        return cos_a * x + sin_a * z, y, -sin_a * x + cos_a * z

    fov = 300

    def project(x, y, z):
        z += 3
        px = int(cx + fov * x / z)
        py = int(cy - fov * y / z)
        return px, py

    pts2d = []
    for vx, vy, vz in verts_raw:
        rx, ry, rz = rot_y(vx, vy, vz)
        scale = raio / phi
        pts2d.append(project(rx * scale / fov, ry * scale / fov, rz * scale / fov))

    overlay = frame.copy()
    for face in faces:
        i0, i1, i2 = face
        v0 = np.array(verts_raw[i0])
        v1 = np.array(verts_raw[i1])
        v2 = np.array(verts_raw[i2])

        def rot_v(v):
            x, y, z = v
            return np.array([cos_a * x + sin_a * z, y, -sin_a * x + cos_a * z])

        n = np.cross(rot_v(v1 - v0), rot_v(v2 - v0))
        if n[2] > 0:
            pts = np.array([pts2d[i0], pts2d[i1], pts2d[i2]], dtype=np.int32)
            intensity = int(80 + 120 * abs(n[2]) / (np.linalg.norm(n) + 1e-6))
            cv2.fillPoly(overlay, [pts], (intensity, intensity // 2, 255))
            cv2.polylines(overlay, [pts], True, (200, 150, 255), 1)

    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    sx, sy = cx, cy + raio + 10
    cv2.ellipse(frame, (sx, sy), (raio // 2, raio // 6), 0, 0, 360, (30, 30, 30), -1)

def ar_mao_frame(frame, hands):
    if not MEDIAPIPE_OK or hands is None:
        cv2.putText(frame, "MediaPipe nao disponivel", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return frame

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hlm in result.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, hlm, mp.solutions.hands.HAND_CONNECTIONS,
                mp.solutions.drawing_utils.DrawingSpec(color=(100, 255, 100), thickness=1, circle_radius=2),
                mp.solutions.drawing_utils.DrawingSpec(color=(50, 200, 50), thickness=1),
            )
            h, w = frame.shape[:2]
            p0 = hlm.landmark[0]
            p9 = hlm.landmark[9]
            cx = int((p0.x + p9.x) / 2 * w)
            cy = int((p0.y + p9.y) / 2 * h)
            raio = int(abs(p9.y - p0.y) * h * 1.2)
            raio = max(30, min(raio, 100))
            desenhar_objeto_3d(frame, cx, cy, raio)
    else:
        cv2.putText(frame, "Mostre a mao para a camera", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)
    return frame


#  GUI PRINCIPAL

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Visão Computacional – TP2")
        self.root.configure(bg="#1e1e2e")
        self.root.resizable(False, False)

        self.modo = tk.StringVar(value="metrologia")
        self.rodando = False
        self.cap = None
        self.thread = None

        self.sons = {}
        self.hands = None

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.fechar)

    def _build_ui(self):
        tk.Label(self.root, text="Visão Computacional – Trabalho 2",
                 bg="#1e1e2e", fg="#cdd6f4",
                 font=("Courier New", 14, "bold")).pack(pady=(12, 4))

        tk.Label(self.root, text="Alunos: Eduardo Larson e Thiago Ceron de Almeida",
                 bg="#1e1e2e", fg="#6c7086",
                 font=("Courier New", 9)).pack(pady=(0, 10))

        frame_btns = tk.Frame(self.root, bg="#1e1e2e")
        frame_btns.pack()

        modos = [
            ("metrologia", "Metrologia"),
            ("ocarina",    "Ocarina"),
            ("ar_mao",     "AR Mão"),
        ]
        self.btns = {}
        for val, label in modos:
            b = tk.Button(frame_btns, text=label, width=14,
                          bg="#313244", fg="#cdd6f4",
                          activebackground="#45475a", activeforeground="#cdd6f4",
                          relief="flat", font=("Courier New", 10),
                          command=lambda v=val: self.selecionar_modo(v))
            b.pack(side="left", padx=5, pady=4)
            self.btns[val] = b

        self.btn_iniciar = tk.Button(frame_btns, text="▶  Iniciar",
                                     bg="#a6e3a1", fg="#1e1e2e",
                                     activebackground="#94e2d5",
                                     relief="flat", font=("Courier New", 10, "bold"),
                                     width=12, command=self.toggle_camera)
        self.btn_iniciar.pack(side="left", padx=15, pady=4)

        self.canvas = tk.Canvas(self.root, width=1280, height=720,
                                bg="#11111b", highlightthickness=0)
        self.canvas.pack(padx=10, pady=8)

        frame_inf = tk.Frame(self.root, bg="#1e1e2e")
        frame_inf.pack(fill="x", padx=10, pady=(0, 10))

        self.label_status = tk.Label(frame_inf, text="Câmera desligada",
                                     bg="#1e1e2e", fg="#6c7086",
                                     font=("Courier New", 9))
        self.label_status.pack(side="right")

        self.selecionar_modo("metrologia")

    def selecionar_modo(self, modo):
        self.modo.set(modo)
        for val, btn in self.btns.items():
            btn.configure(bg="#45475a" if val == modo else "#313244")

        if modo == "ocarina" and not self.sons:
            self.label_status.configure(text="Gerando sons…")
            self.root.update()
            self.sons = gerar_todos_sons()

        if modo == "ar_mao" and self.hands is None:
            if not MEDIAPIPE_OK:
                messagebox.showwarning("MediaPipe", "Erro ao carregar MediaPipe.")
                return
            self.hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7,
            )

        self.label_status.configure(text=f"Modo: {modo.upper()}")

    def toggle_camera(self):
        if self.rodando:
            self.parar_camera()
        else:
            self.iniciar_camera()

    def iniciar_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if not self.cap.isOpened():
            messagebox.showerror("Erro", "Não foi possível abrir a câmera.")
            return

        self.rodando = True
        self.btn_iniciar.configure(text="⏹  Parar", bg="#f38ba8")
        self.label_status.configure(text=f"Câmera ativa – {self.modo.get().upper()}")
        self.thread = threading.Thread(target=self._loop_camera, daemon=True)
        self.thread.start()

    def parar_camera(self):
        self.rodando = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.btn_iniciar.configure(text="▶  Iniciar", bg="#a6e3a1", fg="#1e1e2e")
        self.label_status.configure(text="Câmera desligada")
        self.canvas.delete("all")

    def _loop_camera(self):
        from PIL import Image, ImageTk

        while self.rodando:
            if self.cap is None or not self.cap.isOpened():
                break

            ok, frame = self.cap.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue

            frame = cv2.resize(frame, (1280, 720))

            try:
                modo = self.modo.get()
                if modo == "metrologia":
                    frame = distancia_aruco(frame)
                elif modo == "ocarina":
                    frame = ocarina_frame(frame, self.sons)
                elif modo == "ar_mao":
                    frame = ar_mao_frame(frame, self.hands)
            except Exception as e:
                print(f"[ERRO no frame]: {e}")
                time.sleep(0.03)
                continue

            if frame is None:
                time.sleep(0.03)
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            imgtk = ImageTk.PhotoImage(image=img)

            def atualizar(i=imgtk):
                if self.rodando:
                    self.canvas.create_image(0, 0, anchor="nw", image=i)
                    self.canvas._img = i

            self.root.after(0, atualizar)
            time.sleep(0.03)

    def fechar(self):
        self.parar_camera()
        if self.hands:
            self.hands.close()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()