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


#  gerar sons
NOTES = {
    "do":  261.63,
    "re":  293.66,
    "mi":  329.63,
    "fa":  349.23,
    "sol": 392.00,
    "la":  440.00,
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


# detecta o aruco
def get_aruco_detector():
    dicionario = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    params = cv2.aruco.DetectorParameters()
    return cv2.aruco.ArucoDetector(dicionario, params)

DETECTOR = get_aruco_detector()
MARKER_SIZE_CM = 2.0

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

def estimar_pose(corner):
    img_points = corner[0].astype(np.float64)
    ok, rvec, tvec = cv2.solvePnP(
        OBJ_POINTS, img_points, CAMERA_MATRIX, DIST_COEFFS,
        flags=cv2.SOLVEPNP_IPPE_SQUARE
    )
    return ok, rvec, tvec

def desenha_cubo(frame, rvec, tvec, cor_base=(0, 180, 0)):
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
    for i in range(4):
        cv2.line(frame, tuple(base[i]), tuple(base[(i + 1) % 4]), cor_base, 2)
    for i in range(4):
        cv2.line(frame, tuple(topo[i]), tuple(topo[(i + 1) % 4]), (255, 220, 0), 2)
    for i in range(4):
        cv2.line(frame, tuple(base[i]), tuple(topo[i]), (255, 255, 255), 2)
    overlay = frame.copy()
    faces = [
        ([topo[0], topo[1], base[1], base[0]], (200, 230, 255)),
        ([topo[1], topo[2], base[2], base[1]], (180, 210, 240)),
        ([topo[0], topo[1], topo[2], topo[3]], (160, 200, 255)),
    ]
    for pontos, cor in faces:
        cv2.fillPoly(overlay, [np.array(pontos)], cor)
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)



# METROLOGIA

def distancia_aruco(frame):
    cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = DETECTOR.detectMarkers(cinza)

    if ids is None or len(ids) < 1:
        cv2.putText(frame, "Mostre pelo menos 1 marcador ArUco", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return frame

    cv2.aruco.drawDetectedMarkers(frame, corners, ids)

    poses = {}
    centros_px = {}
    for i, corner in enumerate(corners):
        mid = int(ids[i][0])
        ok, rvec, tvec = estimar_pose(corner)
        if not ok:
            continue
        poses[mid] = (rvec.flatten(), tvec.flatten())
        c = corner[0]
        cx = int(c[:, 0].mean())
        cy = int(c[:, 1].mean())
        centros_px[mid] = (cx, cy)

        dist_z = float(tvec.flatten()[2])
        cv2.putText(frame, f"ID{mid} | Z={dist_z:.1f}cm", (cx - 50, cy - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 220, 0), 2)
        desenha_cubo(frame, rvec, tvec)

    # distancia 3D entre cada par
    mids = sorted(poses.keys())
    y_texto = 35
    for idx_a in range(len(mids)):
        for idx_b in range(idx_a + 1, len(mids)):
            id_a, id_b = mids[idx_a], mids[idx_b]
            tvec_a = poses[id_a][1]
            tvec_b = poses[id_b][1]
            dist_3d = float(np.linalg.norm(tvec_a - tvec_b))

            pa = centros_px[id_a]
            pb = centros_px[id_b]
            cv2.line(frame, pa, pb, (0, 255, 200), 2)
            mx = (pa[0] + pb[0]) // 2
            my = (pa[1] + pb[1]) // 2
            cv2.putText(frame, f"{dist_3d:.1f}cm", (mx - 30, my - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 200), 2)

            cv2.putText(frame,
                        f"ID{id_a}<->ID{id_b}: {dist_3d:.1f} cm",
                        (10, y_texto),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 200), 2)
            y_texto += 30

    if len(mids) < 2:
        cv2.putText(frame, "Mostre 2+ marcadores para medir distancia", (10, y_texto),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

    return frame



#  OCARINA


OCARINA_REF_ID   = 0
OCARINA_HOLE_IDS = [1, 2, 3, 4, 5, 6]
OCARINA_NOTES    = {
    0: ("do",  261.63),
    1: ("re",  293.66),
    2: ("mi",  329.63),
    3: ("fa",  349.23),
    4: ("sol", 392.00),
    5: ("la",  440.00),
}
FURO_CORES = {
    "do":  (255,  80,  80),
    "re":  (255, 180,  50),
    "mi":  (100, 220,  80),
    "fa":  (50,  180, 255),
    "sol": (180,  80, 255),
    "la":  (255, 100, 200),
}
COOLDOWN_S = 0.4   # intervalo entre toques

_ocarina_estado = {
    "was_covered": {},   
    "last_played": {},   
}

def _lerp(a, b, t):
    return a + t * (b - a)

def _ponto(tl, tr, bl, br, u, v):
    return _lerp(_lerp(tl, tr, u), _lerp(bl, br, u), v)

def _desenha_ocarina_perspectivada(frame, ref_corners, holes_covered: dict):

    c = ref_corners[0].astype(float)
    tl, tr, br, bl = c[0], c[1], c[2], c[3]

    body_pts = []
    for i, t in enumerate(np.linspace(0, 1, 60)):
        angle = 2 * math.pi * t
        fatv = 0.5 - 0.5 * math.cos(angle)   
        ru = 0.55 + 0.25 * fatv               
        rv = 0.48 + 0.10 * fatv               
        u = 0.5 + ru * math.cos(angle)
        v = 0.5 + rv * math.sin(angle)
        body_pts.append(_ponto(tl, tr, bl, br, u, v))
    body_arr = np.array(body_pts, dtype=np.int32)

    cv2.fillPoly(frame, [body_arr + np.array([8, 8])], (18, 10, 5))

    overlay = frame.copy()
    cv2.fillPoly(overlay, [body_arr], (45, 95, 180))    
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    light_pts = []
    for t in np.linspace(0.55, 0.95, 24):
        angle = math.pi + math.pi * t
        fatv  = 0.5 - 0.5 * math.cos(angle)
        ru = 0.45 + 0.20 * fatv
        rv = 0.36 + 0.08 * fatv
        u = 0.5 + ru * math.cos(angle)
        v = 0.5 + rv * math.sin(angle)
        light_pts.append(_ponto(tl, tr, bl, br, u, v))
    ov2 = frame.copy()
    cv2.fillPoly(ov2, [np.array(light_pts, dtype=np.int32)], (90, 150, 220))
    cv2.addWeighted(ov2, 0.35, frame, 0.65, 0, frame)

    cv2.polylines(frame, [body_arr], True, (20, 55, 110), 2)

    dec_pts = []
    for t in np.linspace(0.1, 0.9, 20):
        u = 0.5 + 0.55 * math.cos(math.pi * t)
        v = 0.52 + 0.06 * math.sin(math.pi * t)
        dec_pts.append(_ponto(tl, tr, bl, br, u, v))
    cv2.polylines(frame, [np.array(dec_pts, dtype=np.int32)], False, (20, 55, 110), 1)

    bocal_base  = _ponto(tl, tr, bl, br, 1.45, 0.38)
    bocal_meio  = _ponto(tl, tr, bl, br, 1.70, 0.18)
    bocal_ponta = _ponto(tl, tr, bl, br, 1.80, 0.08)
    pts_bocal = np.array([bocal_base, bocal_meio, bocal_ponta], dtype=np.int32)
    cv2.polylines(frame, [pts_bocal], False, (20, 55, 110), 13)
    cv2.polylines(frame, [pts_bocal], False, (60, 120, 195), 7)
    cv2.circle(frame, tuple(bocal_ponta.astype(int)), 7, (90, 160, 215), -1)
    cv2.circle(frame, tuple(bocal_ponta.astype(int)), 7, (20, 55, 110), 2)

    centro = _ponto(tl, tr, bl, br, 0.5, 0.72).astype(int)
    cv2.putText(frame, "OCARINA", (centro[0] - 36, centro[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 230, 255), 1, cv2.LINE_AA)

    posicoes_furos = [
        (0.22, 0.32), (0.44, 0.30), (0.66, 0.32),   
        (0.28, 0.62), (0.50, 0.64), (0.72, 0.62),   
    ]
    for i, hid in enumerate(OCARINA_HOLE_IDS):
        if i >= len(posicoes_furos):
            break
        pu, pv = posicoes_furos[i]
        hole_center = _ponto(tl, tr, bl, br, pu, pv).astype(int)
        nota_nome = OCARINA_NOTES[i][0]
        cor_bgr   = FURO_CORES[nota_nome]          
        coberto   = holes_covered.get(hid, False)

        fill_cor = (15, 10, 8) if coberto else cor_bgr
        cv2.circle(frame, tuple(hole_center), 10, fill_cor, -1)
        cv2.circle(frame, tuple(hole_center), 10, (15, 40, 80), 2)
        if not coberto:
            brilho = tuple(min(v + 70, 255) for v in cor_bgr)
            cv2.circle(frame, (hole_center[0] - 3, hole_center[1] - 3), 3, brilho, -1)
        cv2.putText(frame, nota_nome,
                    (hole_center[0] - 11, hole_center[1] + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (230, 230, 255), 1, cv2.LINE_AA)


def ocarina_frame(frame, sons):
    cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = DETECTOR.detectMarkers(cinza)

    id_map = {}
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        for i in range(len(ids)):
            id_map[int(ids[i][0])] = corners[i]

    holes_covered = {hid: (hid not in id_map) for hid in OCARINA_HOLE_IDS}

    if OCARINA_REF_ID in id_map:
        _desenha_ocarina_perspectivada(frame, id_map[OCARINA_REF_ID], holes_covered)
    else:
        cv2.putText(frame, f"Mostre marcador ID={OCARINA_REF_ID} (corpo da ocarina)",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 255), 2)

    now = time.time()
    notas_tocando = []
    for i, hid in enumerate(OCARINA_HOLE_IDS):
        coberto  = holes_covered[hid]
        anterior = _ocarina_estado["was_covered"].get(hid, False)

        if coberto and not anterior:   
            ultimo = _ocarina_estado["last_played"].get(hid, 0)
            if now - ultimo > COOLDOWN_S and PYGAME_OK:
                nota_nome = OCARINA_NOTES[i][0]
                if nota_nome in sons:
                    sons[nota_nome].play()
                _ocarina_estado["last_played"][hid] = now

        if coberto:
            notas_tocando.append(OCARINA_NOTES[i][0])
        _ocarina_estado["was_covered"][hid] = coberto

    for i, hid in enumerate(OCARINA_HOLE_IDS):
        nota_nome = OCARINA_NOTES[i][0]
        coberto   = holes_covered[hid]
        simbolo   = "●" if coberto else "○"
        cor_hud   = FURO_CORES[nota_nome] if coberto else (160, 160, 160)
        cv2.putText(frame, f"{simbolo} {nota_nome}",
                    (frame.shape[1] - 120, 35 + 28 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor_hud, 2, cv2.LINE_AA)

    h_frame = frame.shape[0]
    cv2.rectangle(frame, (5, h_frame - 50), (560, h_frame - 5), (18, 18, 18), -1)
    cv2.putText(frame, "Cubra os marcadores para tocar  |  do re mi fa sol la",
                (10, h_frame - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    if notas_tocando:
        cv2.putText(frame, f"Tocando: {' + '.join(notas_tocando)}",
                    (10, h_frame - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 150), 1)

    return frame



#  Modo AR

def _carregar_modelo_obj(path):
    verts, faces = [], []
    try:
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                if parts[0] == "v":
                    verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
                elif parts[0] == "f":
                    idxs = [int(p.split("/")[0]) - 1 for p in parts[1:]]
                    for k in range(1, len(idxs) - 1):
                        faces.append((idxs[0], idxs[k], idxs[k + 1]))
        if verts and faces:
            arr = np.array(verts, dtype=np.float64)
            arr -= arr.mean(axis=0)
            scale = np.abs(arr).max()
            if scale > 0:
                arr /= scale
            return arr.tolist(), faces
    except Exception as e:
        print(f"[OBJ load error]: {e}")
    return None, None

def _icosaedro():
    phi = (1 + math.sqrt(5)) / 2
    verts = [
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
    arr = np.array(verts, dtype=np.float64)
    arr /= np.abs(arr).max()
    return arr.tolist(), faces

_MODEL_PATH = "modelo.obj"
_MODEL_VERTS, _MODEL_FACES = _carregar_modelo_obj(_MODEL_PATH)
if _MODEL_VERTS is None:
    print("[AR] modelo.obj não encontrado – usando icosaedro procedural.")
    _MODEL_VERTS, _MODEL_FACES = _icosaedro()

_MODELO_USANDO = "modelo.obj" if os.path.exists(_MODEL_PATH) else "icosaedro (fallback)"

def _rot_y(verts, angulo):
    ca, sa = math.cos(angulo), math.sin(angulo)
    return [(ca*x + sa*z, y, -sa*x + ca*z) for x, y, z in verts]

def _rot_x(verts, angulo):
    ca, sa = math.cos(angulo), math.sin(angulo)
    return [(x, ca*y - sa*z, sa*y + ca*z) for x, y, z in verts]

def desenhar_objeto_3d(frame, cx, cy, raio=70, angulo_y=0.0, angulo_x=0.0,
                       cor_base=(100, 120, 255)):
    fov = 350
    scale = raio / 1.0

    def project(x, y, z):
        z_off = z + 3.5
        if z_off == 0:
            z_off = 0.001
        px = int(cx + fov * (x * scale / fov) / z_off * fov / fov)
        py = int(cy - fov * (y * scale / fov) / z_off * fov / fov)
        return px, py

    verts_rot = _rot_x(_rot_y(_MODEL_VERTS, angulo_y), angulo_x)

    pts2d = [project(x, y, z) for x, y, z in verts_rot]

    luz = np.array([0.5, 0.8, -1.0])
    luz = luz / np.linalg.norm(luz)

    overlay = frame.copy()

    face_depths = []
    for face in _MODEL_FACES:
        i0, i1, i2 = face
        z_med = (verts_rot[i0][2] + verts_rot[i1][2] + verts_rot[i2][2]) / 3
        face_depths.append((z_med, face))
    face_depths.sort(key=lambda fd: fd[0])

    for _, face in face_depths:
        i0, i1, i2 = face
        v0 = np.array(verts_rot[i0])
        v1 = np.array(verts_rot[i1])
        v2 = np.array(verts_rot[i2])
        normal = np.cross(v1 - v0, v2 - v0)
        norm_len = np.linalg.norm(normal)
        if norm_len < 1e-8:
            continue
        normal /= norm_len
        if normal[2] > 0:
            continue
        diff = float(np.clip(-np.dot(normal, luz), 0, 1))
        intensidade = int(60 + 180 * diff)
        r = min(int(cor_base[0] * intensidade / 255), 255)
        g = min(int(cor_base[1] * intensidade / 255), 255)
        b = min(int(cor_base[2] * intensidade / 255), 255)
        pts = np.array([pts2d[i0], pts2d[i1], pts2d[i2]], dtype=np.int32)
        cv2.fillPoly(overlay, [pts], (b, g, r))
        cv2.polylines(overlay, [pts], True, (200, 180, 255), 1)

    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

    sx, sy = cx, cy + raio + 12
    sombra = frame.copy()
    cv2.ellipse(sombra, (sx, sy), (raio // 2, raio // 7), 0, 0, 360, (15, 15, 15), -1)
    cv2.addWeighted(sombra, 0.5, frame, 0.5, 0, frame)


def ar_mao_frame(frame, hands, estado_ar):

    if not MEDIAPIPE_OK or hands is None:
        cv2.putText(frame, "MediaPipe nao disponivel", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return frame

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    h, w = frame.shape[:2]

    if result.multi_hand_landmarks:
        for hlm in result.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, hlm, mp.solutions.hands.HAND_CONNECTIONS,
                mp.solutions.drawing_utils.DrawingSpec(
                    color=(100, 255, 100), thickness=1, circle_radius=2),
                mp.solutions.drawing_utils.DrawingSpec(
                    color=(50, 200, 50), thickness=1),
            )
            p0  = hlm.landmark[0] 
            p9  = hlm.landmark[9]
            p17 = hlm.landmark[17] 

            cx = int((p0.x + p9.x) / 2 * w)
            cy = int((p0.y + p9.y) / 2 * h)

            raio = int(abs(p9.y - p0.y) * h * 1.4)
            raio = max(35, min(raio, 120))

            incl = (p17.y - p9.y) * h
            angulo_x = float(np.clip(incl / 100.0, -0.6, 0.6))

            angulo_y = estado_ar["angulo_y"]

            desenhar_objeto_3d(frame, cx, cy, raio,
                               angulo_y=angulo_y, angulo_x=angulo_x,
                               cor_base=(100, 120, 255))

    else:
        cv2.putText(frame, "Mostre a mao para a camera", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)

    cv2.putText(frame, f"Modelo: {_MODELO_USANDO}", (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
    return frame


# interface 
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

        self._estado_ar = {"angulo_y": 0.0}

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

            self._estado_ar["angulo_y"] = (time.time() * 1.5) % (2 * math.pi)

            try:
                modo = self.modo.get()
                if modo == "metrologia":
                    frame = distancia_aruco(frame)
                elif modo == "ocarina":
                    frame = ocarina_frame(frame, self.sons)
                elif modo == "ar_mao":
                    frame = ar_mao_frame(frame, self.hands, self._estado_ar)
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