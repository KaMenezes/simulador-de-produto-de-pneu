# -*- coding: utf-8 -*-
"""
Por que esse produto?  (simulador didatico, versao 2)

Instalar:  pip install streamlit plotly numpy
Rodar:     python -m streamlit run app.py

ATENCAO: o modelo e ILUSTRATIVO, nao e calculo de engenharia.
Serve para mostrar o conceito: funcao do produto -> consequencia do erro.

COMO ADICIONAR UM PRODUTO NOVO
1. Copie um bloco dentro de PRODUTOS e preencha onde / funcao / se_errar.
2. Opcional: adicione perguntas em "quiz" (modelo no D07).
3. So o D07 tem simulacao 3D. Os demais mostram o texto e o quiz.
"""
import csv
import hmac
import io
import json
from datetime import datetime

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Por que esse produto?", page_icon="🛞", layout="centered")

# =====================================================================
# 1) CATALOGO DE PRODUTOS  (complete com a Qualidade antes de divulgar)
# =====================================================================
PRODUTOS = {
    "D07 (ombro)": {
        "onde": "Ombro do pneu, a zona de transicao entre o topo e o flanco.",
        "funcao": "Ajuda a dissipar o calor gerado nas extremidades quando o pneu roda.",
        "se_errar": "Sem o D07, ou com ele mal posicionado, o ombro superaquece. "
                    "Com o tempo isso pode causar falha do pneu na estrada, ate estouro.",
        "simulavel": True,
        "quiz": [
            {"p": "Onde o D07 e aplicado no pneu?",
             "op": ["No ombro, entre o topo e o flanco", "No centro da banda de rodagem",
                    "No talao, junto ao aro", "Na parede interna do pneu"],
             "ok": 0,
             "exp": "O D07 fica no ombro, a regiao de transicao entre o topo e o flanco."},
            {"p": "Para que serve o D07 com o pneu rodando?",
             "op": ["Deixar o pneu mais bonito", "Evitar que o pneu fure com pregos",
                    "Ajudar a dissipar o calor nas extremidades", "Facilitar a montagem no aro"],
             "ok": 2,
             "exp": "Ele ajuda a dissipar o calor gerado nas extremidades do pneu."},
            {"p": "O que pode acontecer se o D07 for esquecido ou retirado?",
             "op": ["Nada, e so acabamento", "O ombro superaquece e o pneu pode falhar, ate estourar",
                    "O pneu fica mais leve", "O pneu fica mais silencioso"],
             "ok": 1,
             "exp": "Sem o D07 o calor se concentra no ombro, e o pneu pode falhar na estrada."},
            {"p": "O D07 foi aplicado, mas deslocado para longe do ombro. Qual o efeito?",
             "op": ["Funciona igual", "Esfria o pneu inteiro", "So muda a aparencia",
                    "Protege pouco o ponto mais quente, quase como se nao existisse"],
             "ok": 3,
             "exp": "Posicao tambem importa: fora do ombro, o ponto mais quente fica sem protecao."},
            {"p": "Um colega quer retirar o D07 porque 'nao parece importante'. O que voce faz?",
             "op": ["Deixo, o procedimento e dele", "Retiro tambem para ganhar tempo",
                    "Explico que ele ajuda a dissipar o calor no ombro e aviso a lideranca antes",
                    "Pergunto so no fim do turno"],
             "ok": 2,
             "exp": "Quem entende a funcao evita o erro e corrige o colega antes do defeito."},
        ],
    },
    "Napas de topo": {
        "onde": "A preencher com a equipe de Qualidade.",
        "funcao": "A preencher com a equipe de Qualidade.",
        "se_errar": "A preencher.",
        "simulavel": False,
        "quiz": [],
    },
    "Produto zona baixa": {
        "onde": "A preencher com a equipe de Qualidade.",
        "funcao": "A preencher com a equipe de Qualidade.",
        "se_errar": "A preencher.",
        "simulavel": False,
        "quiz": [],
    },
    "Produto zona alta": {
        "onde": "A preencher com a equipe de Qualidade.",
        "funcao": "A preencher com a equipe de Qualidade.",
        "se_errar": "A preencher.",
        "simulavel": False,
        "quiz": [],
    },
}

# =====================================================================
# 2) MODELO ILUSTRATIVO DE TEMPERATURA
# =====================================================================
T_AMB = 25.0          # C
T_LIMITE = 110.0      # C, acima disso ha risco (ilustrativo)
T_PERIGO = 130.0      # C, risco critico (ilustrativo)
SHOULDER = np.pi / 4  # posicao do ombro na secao (rad)
DESL_MAX = 0.9        # deslocamento maximo do D07 (rad)
TAU_MIN = 20.0        # velocidade de aquecimento (min), ilustrativo

CARGAS = {"Leve": 0.3, "Normal": 0.6, "Pesada": 0.8, "Extrema": 1.0}

# (quantidade aplicada 0..1, deslocamento em rad)
PRESETS = {
    "Bem aplicado": (1.0, 0.0),
    "Mal posicionado": (1.0, DESL_MAX),
    "Ausente": (0.0, 0.0),
}
COR_CENARIO = {
    "Bem aplicado": "#1b9e77",
    "Mal posicionado": "#d95f02",
    "Ausente": "#c1121f",
    "Personalizado": "#6a3d9a",
}


def gauss(phi, centro, largura):
    d = np.angle(np.exp(1j * (phi - centro)))  # diferenca angular com wrap
    return np.exp(-(d ** 2) / (2 * largura ** 2))


def temperatura(phi, quantidade, deslocamento, carga):
    """Temperatura em regime, em funcao da posicao na secao do pneu (phi)."""
    q = 45.0 + 60.0 * carga  # calor gerado
    base = q * (0.35 + 0.65 * (gauss(phi, SHOULDER, 0.35) + gauss(phi, -SHOULDER, 0.35)))
    centro = SHOULDER + deslocamento
    cobertura = np.clip(gauss(phi, centro, 0.40) + gauss(phi, -centro, 0.40), 0.0, 1.0)
    return T_AMB + base * (1.0 - 0.55 * quantidade * cobertura)


def pico(quantidade, deslocamento, carga):
    phi = np.linspace(-np.pi, np.pi, 721)
    return float(temperatura(phi, quantidade, deslocamento, carga).max())


def temp_no_tempo(t_ss, t_min):
    return T_AMB + (t_ss - T_AMB) * (1.0 - np.exp(-t_min / TAU_MIN))


def tempo_ate_limite(t_ss):
    """Minutos ate passar do limite (None se nunca passa)."""
    if t_ss <= T_LIMITE:
        return None
    return float(-TAU_MIN * np.log(1.0 - (T_LIMITE - T_AMB) / (t_ss - T_AMB)))


def nivel(t):
    if t >= T_PERIGO:
        return "critico"
    if t >= T_LIMITE:
        return "alerta"
    return "ok"


# =====================================================================
# 3) GEOMETRIA E FIGURAS
# =====================================================================
RC, A, B, P = 0.78, 0.22, 0.26, 0.55  # raio central, meia-altura, meia-largura, "quadratura"


def secao(phi):
    c, s = np.cos(phi), np.sin(phi)
    rad = RC + A * np.sign(c) * np.abs(c) ** P
    ax = B * np.sign(s) * np.abs(s) ** P
    return rad, ax


def figura_3d(quantidade, deslocamento, carga):
    phi = np.linspace(-np.pi, np.pi, 60)
    theta = np.linspace(0, 2 * np.pi, 80)
    PHI, TH = np.meshgrid(phi, theta, indexing="ij")
    rad, ax = secao(PHI)
    T = temperatura(PHI, quantidade, deslocamento, carga)

    data = [go.Surface(
        x=rad * np.cos(TH), y=ax, z=rad * np.sin(TH), surfacecolor=T,
        cmin=T_AMB, cmax=T_PERIGO + 20, colorscale="Turbo", showscale=True,
        colorbar=dict(title="°C", thickness=12, len=0.7), name="Pneu",
        hovertemplate="%{surfacecolor:.0f} °C<extra></extra>",
    )]
    # faixa que marca onde o D07 foi aplicado
    if quantidade > 0:
        th = np.linspace(0, 2 * np.pi, 200)
        for k, sgn in enumerate((1, -1)):
            r0, a0 = secao(np.array([sgn * (SHOULDER + deslocamento)]))
            r0, a0 = float(r0[0]) * 1.012, float(a0[0]) * 1.03
            data.append(go.Scatter3d(
                x=r0 * np.cos(th), y=np.full_like(th, a0), z=r0 * np.sin(th),
                mode="lines", line=dict(color="#0b132b", width=8),
                name="Posição do D07", showlegend=(k == 0), hoverinfo="skip"))

    frames = []
    for k in range(0, 360, 10):
        t = np.radians(k)
        frames.append(go.Frame(layout=dict(scene_camera=dict(
            up=dict(x=0, y=1, z=0),
            eye=dict(x=2.0 * np.cos(t), y=0.9, z=2.0 * np.sin(t))))))

    fig = go.Figure(data=data, frames=frames)
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0), height=420,
        legend=dict(orientation="h", y=-0.02, x=0.5, xanchor="center"),
        scene=dict(
            xaxis_visible=False, yaxis_visible=False, zaxis_visible=False,
            aspectmode="data",
            camera=dict(up=dict(x=0, y=1, z=0), eye=dict(x=2.0, y=0.9, z=0.0)),
        ),
        updatemenus=[dict(
            type="buttons", showactive=False, x=0.02, y=0.98,
            buttons=[
                dict(label="▶ Girar", method="animate",
                     args=[None, dict(frame=dict(duration=80, redraw=True),
                                      fromcurrent=True, transition=dict(duration=0),
                                      mode="immediate")]),
                dict(label="⏸", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                        mode="immediate")]),
            ])],
    )
    return fig


def figura_corte(quantidade, deslocamento, carga):
    """Corte transversal do pneu: cada ponto colorido e uma posicao na secao."""
    phi = np.linspace(-np.pi, np.pi, 300)
    rad, ax = secao(phi)
    T = temperatura(phi, quantidade, deslocamento, carga)
    fig = go.Figure(go.Scatter(
        x=ax, y=rad, mode="markers",
        marker=dict(size=15, color=T, cmin=T_AMB, cmax=T_PERIGO + 20,
                    colorscale="Turbo", colorbar=dict(title="°C", thickness=12)),
        hovertemplate="%{marker.color:.0f} °C<extra></extra>", name="Temperatura"))
    if quantidade > 0:
        pts = np.array([SHOULDER + deslocamento, -(SHOULDER + deslocamento)])
        r0, a0 = secao(pts)
        fig.add_trace(go.Scatter(
            x=a0, y=r0, mode="markers+text", text=["D07", "D07"], textposition="top center",
            marker=dict(size=22, symbol="diamond", color="#ffffff",
                        line=dict(color="#0b132b", width=3)),
            name="D07", hoverinfo="skip"))
    fig.add_annotation(x=0, y=1.09, text="Banda de rodagem (topo)", showarrow=False)
    fig.add_annotation(x=0, y=0.47, text="Interior do pneu", showarrow=False)
    fig.update_layout(
        height=420, margin=dict(l=0, r=0, t=10, b=0), showlegend=False,
        xaxis=dict(visible=False, range=[-0.45, 0.45]),
        yaxis=dict(visible=False, range=[0.40, 1.15], scaleanchor="x", scaleratio=1))
    return fig


def figura_tempo(cenarios, selecionado, carga):
    t = np.linspace(0, 90, 91)
    fig = go.Figure()
    for nome, (qtd, desl) in cenarios.items():
        t_ss = pico(qtd, desl, carga)
        fig.add_trace(go.Scatter(
            x=t, y=temp_no_tempo(t_ss, t), name=nome,
            line=dict(color=COR_CENARIO[nome], width=5 if nome == selecionado else 1.5),
            opacity=1 if nome == selecionado else 0.4))
    fig.add_hline(y=T_LIMITE, line_dash="dash", annotation_text="limite")
    fig.add_hline(y=T_PERIGO, line_dash="dot", annotation_text="risco crítico")
    fig.update_layout(
        height=320, margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="Tempo rodando (min, ilustrativo)", yaxis_title="Temperatura no ombro (°C)",
        legend=dict(orientation="h", y=-0.3))
    return fig


# =====================================================================
# 3b) CAMINHAO 3D (Three.js dentro do Streamlit)
# =====================================================================
SIM_K = 0.3            # velocidade de dano acima do limite (ilustrativo)
SIM_HORIZONTE = 120.0  # minutos simulados da viagem
SIM_RATE = 2.5         # minutos simulados por segundo real (velocidade 1x)
SIM_KMH = 80           # velocidade media do caminhao (so para o contador de km)

SIM_HTML = r'''<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
*{box-sizing:border-box}
html,body{margin:0;height:100%;background:#0b132b;font-family:Arial,Helvetica,sans-serif;overflow:hidden}
#wrap{position:relative;width:100%;height:100%;overflow:hidden}
#c{display:block;width:100%;height:100%;touch-action:none}
.abs{position:absolute}
#top{left:8px;right:8px;top:8px;display:flex;gap:6px;flex-wrap:wrap;pointer-events:none}
.chip{background:rgba(11,19,43,.78);color:#fff;font-size:13px;padding:6px 10px;border-radius:8px;border-left:4px solid #556}
.chip.ok{border-left-color:#2fbf71}
.chip.orange{border-left-color:#f4a100;color:#ffd9a0}
.chip.red{border-left-color:#e5383b;color:#ffb3b5}
#integ{min-width:150px}
#hBar{height:6px;background:#334;border-radius:3px;margin-top:4px;overflow:hidden}
#hBarIn{height:100%;width:100%;background:#2fbf71}
#banner{position:absolute;top:58px;left:50%;transform:translateX(-50%);padding:8px 16px;border-radius:20px;background:rgba(11,19,43,.85);color:#fff;font-weight:bold;font-size:15px;white-space:nowrap;pointer-events:none}
#banner.red{background:#b3141a}
#banner.orange{background:#b86b00}
#banner.yellow{background:#8a7400}
#banner.ok{background:rgba(24,110,66,.92)}
#banner.flash{animation:fl .5s infinite alternate}
@keyframes fl{from{opacity:1}to{opacity:.45}}
#sec{left:8px;bottom:8px;width:170px;height:170px;border-radius:10px;pointer-events:none}
#ctrl{right:8px;bottom:8px;display:flex;flex-direction:column;gap:6px;align-items:flex-end}
.row{display:flex;gap:6px;flex-wrap:wrap;justify-content:flex-end}
button{background:rgba(11,19,43,.85);color:#fff;border:1px solid #4b5a85;border-radius:8px;padding:7px 10px;font-size:13px;cursor:pointer}
button.on{background:#1e40af;border-color:#7aa2ff}
#start{background:#1b9e77;border-color:#5fe3a1;font-weight:bold;font-size:15px}
#intro{left:50%;top:48%;transform:translate(-50%,-50%);text-align:center;color:#fff;background:rgba(11,19,43,.8);padding:14px 22px;border-radius:12px;font-size:16px;pointer-events:none;max-width:80%}
#intro small{display:block;margin-top:6px;opacity:.8;font-size:12px}
#result{display:none;left:50%;top:50%;transform:translate(-50%,-50%);width:min(88%,460px);background:rgba(11,19,43,.95);color:#fff;border:2px solid #7aa2ff;border-radius:14px;padding:18px 20px;text-align:center}
#rTitle{font-size:19px;font-weight:bold;margin-bottom:8px}
#rText{font-size:14px;line-height:1.45;margin-bottom:14px}
#err{display:none;position:absolute;left:12px;right:12px;top:40%;background:#7a1f1f;color:#fff;padding:14px;border-radius:10px;font-size:14px;z-index:5}
@media (max-width:520px){
 #sec{width:120px;height:120px}
 #ctrl{max-width:calc(100% - 136px)}
 .chip{font-size:11px;padding:4px 7px}
 #banner{top:50px;font-size:12px;padding:6px 12px}
 button{padding:6px 8px;font-size:12px}
}
</style>
</head>
<body>
<div id="wrap">
  <canvas id="c"></canvas>
  <div id="top" class="abs">
    <div class="chip" id="hT">0 min</div>
    <div class="chip" id="hKm">0 km</div>
    <div class="chip ok" id="hTemp">Ombro 25 °C</div>
    <div class="chip" id="integ"><span id="hInt">Integridade do pneu 100%</span><div id="hBar"><div id="hBarIn"></div></div></div>
  </div>
  <div id="banner" class="abs">Pneu monitorado</div>
  <div id="intro" class="abs">Toque em ▶ Iniciar para ver a viagem<small>Arraste para girar a câmera. Pinça ou rolagem para aproximar.</small></div>
  <canvas id="sec" class="abs" width="170" height="170"></canvas>
  <div id="ctrl" class="abs">
    <div class="row"><button id="start">▶ Iniciar</button><button id="reset">↻ Reiniciar</button></div>
    <div class="row"><button id="x1">1x</button><button id="x2">2x</button><button id="x4">4x</button></div>
    <div class="row"><button id="cam_side">Lateral</button><button id="cam_wheel">Pneu</button><button id="cam_rear">Traseira</button><button id="cam_top">Aérea</button><button id="auto">🎥 Auto</button></div>
  </div>
  <div id="result" class="abs"><div id="rTitle"></div><div id="rText"></div><button id="again">↻ Rodar de novo</button></div>
  <div id="err"></div>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
(function () {
"use strict";
var P = __PARAMS__;

function $(id) { return document.getElementById(id); }
function showErr(msg) { var e = $("err"); e.style.display = "block"; e.textContent = msg; }

if (typeof THREE === "undefined" || !THREE.OrbitControls) {
  showErr("Não foi possível carregar o Three.js (a rede pode estar bloqueando o site cdnjs/jsdelivr). " +
          "Tente em outra rede, ou use a aba Aprender.");
  return;
}
var renderer;
try {
  renderer = new THREE.WebGLRenderer({ canvas: $("c"), antialias: true });
} catch (e) {
  showErr("Seu navegador não conseguiu iniciar o WebGL (3D).");
  return;
}
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

// ---------- utilidades ----------
var seed = 12345;
function rnd() { seed = (seed * 1664525 + 1013904223) % 4294967296; return seed / 4294967296; }
function clamp(x, a, b) { return Math.max(a, Math.min(b, x)); }
function lerp(a, b, t) { return a + (b - a) * t; }
function M(color, o) {
  var p = { color: color, roughness: 0.8, metalness: 0.1 };
  if (o) { for (var k in o) { p[k] = o[k]; } }
  return new THREE.MeshStandardMaterial(p);
}
function B(color) { return new THREE.MeshBasicMaterial({ color: color }); }

var STOPS = [
  [25, [0.10, 0.10, 0.11]], [55, [0.16, 0.13, 0.12]], [80, [0.55, 0.16, 0.05]],
  [105, [0.95, 0.38, 0.05]], [125, [1.0, 0.65, 0.12]], [150, [1.0, 0.92, 0.55]]
];
function heat(T) {
  if (T <= STOPS[0][0]) { return STOPS[0][1]; }
  for (var i = 1; i < STOPS.length; i++) {
    if (T <= STOPS[i][0]) {
      var a = STOPS[i - 1], b = STOPS[i], k = (T - a[0]) / (b[0] - a[0]);
      return [lerp(a[1][0], b[1][0], k), lerp(a[1][1], b[1][1], k), lerp(a[1][2], b[1][2], k)];
    }
  }
  return STOPS[STOPS.length - 1][1];
}
function css(c) {
  return "rgb(" + Math.round(c[0] * 255) + "," + Math.round(c[1] * 255) + "," + Math.round(c[2] * 255) + ")";
}

// ---------- secao do pneu (mesma formula do app) ----------
var RC = 0.78, A = 0.22, BW = 0.26, PW = 0.55;
function sec(phi) {
  var c = Math.cos(phi), s = Math.sin(phi);
  return { rad: RC + A * Math.sign(c) * Math.pow(Math.abs(c), PW),
           ax: BW * Math.sign(s) * Math.pow(Math.abs(s), PW) };
}
var N = P.profile.length;
function phiOf(j) { return -Math.PI + 2 * Math.PI * j / (N - 1); }
var SHOULDER = Math.PI / 4;
function angDiff(a, b) { var d = a - b; return Math.atan2(Math.sin(d), Math.cos(d)); }

// ---------- cena ----------
var scene = new THREE.Scene();
scene.background = new THREE.Color(0x9ec9ff);
scene.fog = new THREE.Fog(0x9ec9ff, 70, 230);
var camera = new THREE.PerspectiveCamera(46, 1, 0.1, 700);
camera.position.set(1.5, 2.8, 9.5);
var controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.target.set(-0.5, 1.1, 0);
controls.enableDamping = true;
controls.maxPolarAngle = Math.PI * 0.495;
controls.minDistance = 2.5;
controls.maxDistance = 45;

scene.add(new THREE.HemisphereLight(0xffffff, 0x5b7240, 0.95));
var sun = new THREE.DirectionalLight(0xffffff, 0.85);
sun.position.set(12, 22, 9);
scene.add(sun);
var flash = new THREE.PointLight(0xffa040, 0, 30);
scene.add(flash);

var ground = new THREE.Mesh(new THREE.PlaneGeometry(900, 900), M(0x6f9a4f, { roughness: 1 }));
ground.rotation.x = -Math.PI / 2; ground.position.y = -0.03; scene.add(ground);
var road = new THREE.Mesh(new THREE.PlaneGeometry(900, 11), M(0x3b3f46, { roughness: 0.95 }));
road.rotation.x = -Math.PI / 2; scene.add(road);
[-5, 5].forEach(function (z) {
  var l = new THREE.Mesh(new THREE.BoxGeometry(900, 0.02, 0.2), B(0xf2f2f2));
  l.position.set(0, 0.012, z); scene.add(l);
});

var LOOP = 240, HALFL = 120;
var movers = [];
function addMover(obj, x) { obj.position.x = x; scene.add(obj); movers.push(obj); }
var i;
for (i = 0; i < 24; i++) {
  var dash = new THREE.Mesh(new THREE.BoxGeometry(3, 0.02, 0.25), B(0xf4d03f));
  dash.position.y = 0.016; addMover(dash, -HALFL + i * 10);
}
for (i = 0; i < 12; i++) {
  [-1, 1].forEach(function (side) {
    var pole = new THREE.Group();
    var post = new THREE.Mesh(new THREE.BoxGeometry(0.12, 1.0, 0.12), M(0xeeeeee));
    post.position.y = 0.5; pole.add(post);
    var ref = new THREE.Mesh(new THREE.BoxGeometry(0.13, 0.18, 0.13), B(0xd62828));
    ref.position.y = 0.8; pole.add(ref);
    pole.position.z = side * 5.8;
    addMover(pole, -HALFL + i * 20 + (side > 0 ? 0 : 10));
  });
}
for (i = 0; i < 40; i++) {
  var tree = new THREE.Group();
  var trunk = new THREE.Mesh(new THREE.CylinderGeometry(0.25, 0.32, 2, 8), M(0x6b4a2b));
  trunk.position.y = 1; tree.add(trunk);
  var leaves = new THREE.Mesh(new THREE.ConeGeometry(1.6, 4.4, 9), M(0x2f6b3a));
  leaves.position.y = 4; tree.add(leaves);
  var sc = 0.8 + rnd() * 0.9; tree.scale.set(sc, sc, sc);
  tree.position.z = (rnd() < 0.5 ? -1 : 1) * (9 + rnd() * 30);
  addMover(tree, -HALFL + rnd() * LOOP);
}
for (i = 0; i < 8; i++) {
  var h = 50 + rnd() * 50;
  var mt = new THREE.Mesh(new THREE.ConeGeometry(30 + rnd() * 25, h, 7), M(0x8896a6, { roughness: 1 }));
  mt.position.set(-170 + i * 50, h / 2 - 0.03, -150 - rnd() * 40); scene.add(mt);
}

// ---------- caminhao ----------
var truck = new THREE.Group(); scene.add(truck);
var body = new THREE.Group(); truck.add(body);
function box(w, h, d, mat, x, y, z, parent) {
  var m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mat);
  m.position.set(x, y, z); (parent || body).add(m); return m;
}
box(9, 0.3, 1.7, M(0x1f2328), 0, 0.85, 0);                 // chassi
box(6.2, 2.6, 2.4, M(0xe8eaed), -1.45, 2.3, 0);            // carroceria
box(6.22, 0.3, 2.42, M(0x1e40af), -1.45, 1.35, 0);         // faixa
box(2.0, 2.1, 2.3, M(0x1e40af), 3.4, 2.05, 0);             // cabine
box(0.06, 0.95, 2.1, M(0xa8d8ff, { metalness: 0.5, roughness: 0.2 }), 4.42, 2.6, 0); // para-brisa
box(1.1, 0.8, 0.03, M(0xa8d8ff, { metalness: 0.5, roughness: 0.2 }), 3.5, 2.6, 1.16);
box(1.1, 0.8, 0.03, M(0xa8d8ff, { metalness: 0.5, roughness: 0.2 }), 3.5, 2.6, -1.16);
box(0.25, 0.4, 2.45, M(0x1f2328), 4.5, 1.12, 0);           // para-choque
box(0.05, 0.5, 1.2, M(0x2b2f36), 4.42, 1.55, 0);           // grade
box(0.05, 0.22, 0.42, new THREE.MeshStandardMaterial({ color: 0xfff3b0, emissive: 0xffe58a, emissiveIntensity: 0.8 }), 4.43, 1.5, 0.85);
box(0.05, 0.22, 0.42, new THREE.MeshStandardMaterial({ color: 0xfff3b0, emissive: 0xffe58a, emissiveIntensity: 0.8 }), 4.43, 1.5, -0.85);
var stack = new THREE.Mesh(new THREE.CylinderGeometry(0.08, 0.08, 1.4, 10), M(0xb8bec6, { metalness: 0.8 }));
stack.position.set(2.5, 3.4, 1.0); body.add(stack);

// pneu: geometria por revolucao da secao
var R = 0.52, HALF = 0.16;
function tireGeometry() {
  var pts = [];
  for (var j = 0; j < N; j++) {
    var s = sec(phiOf(j));
    pts.push(new THREE.Vector2(s.rad * R, s.ax * (HALF / BW)));
  }
  var g = new THREE.LatheGeometry(pts, 48);
  g.computeVertexNormals();
  var cnt = g.attributes.position.count;
  g.setAttribute("color", new THREE.BufferAttribute(new Float32Array(cnt * 3), 3));
  g.userData = { base: g.attributes.position.array.slice(), cnt: cnt };
  return g;
}
function paint(geo, temps) {
  var col = geo.attributes.color.array, cnt = geo.userData.cnt, cols = [], j;
  for (j = 0; j < N; j++) { cols.push(heat(temps[j])); }
  for (var v = 0; v < cnt; v++) {
    var c = cols[v % N];
    col[v * 3] = c[0]; col[v * 3 + 1] = c[1]; col[v * 3 + 2] = c[2];
  }
  geo.attributes.color.needsUpdate = true;
}
function glowMaterial() {
  var m = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.9, metalness: 0.0, side: THREE.DoubleSide });
  m.onBeforeCompile = function (sh) {
    sh.fragmentShader = sh.fragmentShader.replace("#include <emissivemap_fragment>",
      "#include <emissivemap_fragment>\n totalEmissiveRadiance += vColor * smoothstep(0.30, 0.85, vColor.r) * 0.9;");
  };
  return m;
}
var coldTemps = []; for (i = 0; i < N; i++) { coldTemps.push(40); }
var coldGeo = tireGeometry(); paint(coldGeo, coldTemps);
var hubMat = M(0xb8bec6, { metalness: 0.8, roughness: 0.35 });
var barMat = M(0x7a1f1f);

function makeWheel(x, z, monitored) {
  var pivot = new THREE.Group(); pivot.position.set(x, R, z); truck.add(pivot);
  var spin = new THREE.Group(); pivot.add(spin);
  var geo = monitored ? tireGeometry() : coldGeo;
  if (monitored) { paint(geo, coldTemps); }
  var tire = new THREE.Mesh(geo, glowMaterial()); tire.rotation.x = Math.PI / 2; spin.add(tire);
  var hub = new THREE.Mesh(new THREE.CylinderGeometry(0.30, 0.30, 0.30, 24), hubMat);
  hub.rotation.x = Math.PI / 2; spin.add(hub);
  var bar1 = new THREE.Mesh(new THREE.BoxGeometry(0.46, 0.07, 0.34), barMat); spin.add(bar1);
  var bar2 = new THREE.Mesh(new THREE.BoxGeometry(0.46, 0.07, 0.34), barMat); bar2.rotation.z = Math.PI / 2; spin.add(bar2);
  return { pivot: pivot, spin: spin, tire: tire, geo: geo, monitored: monitored };
}
var wheels = [];
[3.4, -1.8, -3.2].forEach(function (x) {
  [1.05, -1.05].forEach(function (z) {
    wheels.push(makeWheel(x, z, x === -1.8 && z > 0));
  });
});
var MON = wheels.filter(function (w) { return w.monitored; })[0];

function textSprite(txt, color) {
  var c = document.createElement("canvas"); c.width = 512; c.height = 128;
  var g = c.getContext("2d");
  g.fillStyle = "rgba(11,19,43,0.88)"; g.fillRect(0, 0, 512, 128);
  g.strokeStyle = color; g.lineWidth = 6; g.strokeRect(3, 3, 506, 122);
  g.fillStyle = color; g.font = "bold 50px Arial"; g.textAlign = "center"; g.textBaseline = "middle";
  g.fillText(txt, 256, 66);
  var sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(c), transparent: true, depthTest: false }));
  sp.scale.set(2.6, 0.65, 1); sp.renderOrder = 10; return sp;
}
var label = textSprite(P.label, P.tone === "bad" ? "#ff6b5e" : "#5fe3a1");
label.position.set(-1.8, 2.0, 1.5); truck.add(label);

// ---------- efeitos ----------
function radialTex(stops) {
  var c = document.createElement("canvas"); c.width = 128; c.height = 128;
  var g = c.getContext("2d"), gr = g.createRadialGradient(64, 64, 0, 64, 64, 64);
  stops.forEach(function (s) { gr.addColorStop(s[0], s[1]); });
  g.fillStyle = gr; g.fillRect(0, 0, 128, 128);
  return new THREE.CanvasTexture(c);
}
var smokeTex = radialTex([[0, "rgba(80,80,80,0.9)"], [0.6, "rgba(80,80,80,0.35)"], [1, "rgba(80,80,80,0)"]]);
var fireTex = radialTex([[0, "rgba(255,245,200,1)"], [0.35, "rgba(255,160,40,0.9)"], [1, "rgba(255,70,0,0)"]]);
var ringTex = radialTex([[0, "rgba(255,255,255,0)"], [0.72, "rgba(255,255,255,0)"], [0.86, "rgba(255,255,255,0.9)"], [1, "rgba(255,255,255,0)"]]);

var smoke = [], smokeIdx = 0;
for (i = 0; i < 100; i++) {
  var sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: smokeTex, transparent: true, depthWrite: false, opacity: 0 }));
  sp.visible = false; sp.userData = { life: 0, max: 1, vel: new THREE.Vector3(), size: 1, op: 0.5 };
  scene.add(sp); smoke.push(sp);
}
function puff(pos, vel, size, life, op) {
  var sp = smoke[smokeIdx++ % smoke.length];
  sp.visible = true; sp.position.copy(pos);
  sp.userData.vel.copy(vel); sp.userData.life = life; sp.userData.max = life;
  sp.userData.size = size; sp.userData.op = op;
  sp.scale.set(size, size, 1); sp.material.opacity = op;
}
var fires = [];
for (i = 0; i < 4; i++) {
  var f = new THREE.Sprite(new THREE.SpriteMaterial({ map: fireTex, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, opacity: 0 }));
  f.visible = false; f.userData = { dur: 0.5, max: 4, delay: 0 }; scene.add(f); fires.push(f);
}
var ring = new THREE.Sprite(new THREE.SpriteMaterial({ map: ringTex, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, opacity: 0 }));
ring.visible = false; scene.add(ring);
var debris = [];
function spawnDebris(origin) {
  for (var k = 0; k < 70; k++) {
    var strip = k < 10;
    var geo = strip ? new THREE.BoxGeometry(0.5 + rnd() * 0.5, 0.02, 0.1 + rnd() * 0.05)
                    : new THREE.BoxGeometry(0.05 + rnd() * 0.22, 0.03 + rnd() * 0.12, 0.05 + rnd() * 0.22);
    var m = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ color: rnd() < 0.25 ? 0x6a2008 : 0x151515, roughness: 0.9 }));
    m.position.set(origin.x + (rnd() - 0.5) * 0.4, origin.y + (rnd() - 0.2) * 0.5, origin.z + (rnd() - 0.2) * 0.3);
    var ang = rnd() * Math.PI * 2, out = 2 + rnd() * 9;
    m.userData = { v: new THREE.Vector3(Math.cos(ang) * out * 0.7, 3 + rnd() * 8, Math.sin(ang) * out * 0.8 + 2.5),
                   w: new THREE.Vector3((rnd() - 0.5) * 16, (rnd() - 0.5) * 16, (rnd() - 0.5) * 16) };
    scene.add(m); debris.push(m);
  }
}

// ---------- estado ----------
var S = null;
function newState() {
  return { running: false, started: false, t: 0, dano: 0, exploded: false, finished: false, speed: 0,
           mul: 1, auto: true, boom: 0, smokeAcc: 0, fin: 0, resultShown: false, camMode: "" };
}
S = newState();

function resetAll() {
  var keep = { mul: S.mul, auto: S.auto };
  S = newState(); S.mul = keep.mul; S.auto = keep.auto;
  MON.tire.visible = true; MON.tire.scale.set(1, 1, 1);
  var pos = MON.geo.attributes.position, base = MON.geo.userData.base;
  for (var a = 0; a < base.length; a++) { pos.array[a] = base[a]; }
  pos.needsUpdate = true;
  paint(MON.geo, coldTemps);
  debris.forEach(function (d) { scene.remove(d); d.geometry.dispose(); d.material.dispose(); });
  debris = [];
  smoke.forEach(function (s) { s.visible = false; });
  fires.forEach(function (f) { f.visible = false; });
  ring.visible = false; flash.intensity = 0;
  body.rotation.set(0, 0, 0); truck.position.set(0, 0, 0); truck.rotation.set(0, 0, 0);
  $("result").style.display = "none";
  $("start").textContent = "▶ Iniciar";
  setCam("side", true);
}

// ---------- camera ----------
var CAMS = {
  side: { p: [1.5, 2.8, 9.5], t: [-0.5, 1.1, 0] },
  wheel: { p: [-0.6, 1.1, 3.9], t: [-1.8, 0.55, 1.05] },
  far: { p: [7, 3.6, 12], t: [-1, 1.0, 0] },
  rear: { p: [-11, 3.4, 4.5], t: [0, 1.4, 0] },
  top: { p: [0.5, 21, 3], t: [0, 0, 0] }
};
var camGoal = null;
function setCam(name, instant) {
  var c = CAMS[name]; if (!c) { return; }
  S.camMode = name;
  camGoal = { p: new THREE.Vector3(c.p[0], c.p[1], c.p[2]), t: new THREE.Vector3(c.t[0], c.t[1], c.t[2]) };
  if (instant) { camera.position.copy(camGoal.p); controls.target.copy(camGoal.t); camGoal = null; }
}
controls.addEventListener("start", function () { camGoal = null; });

// ---------- explosao ----------
function explode() {
  S.exploded = true; S.boom = 0;
  MON.tire.visible = false;
  var o = new THREE.Vector3(-1.8, R, 1.05);
  spawnDebris(o);
  fires.forEach(function (f, k) {
    f.visible = true; f.position.set(o.x + (rnd() - 0.5) * 0.6, o.y + 0.3 + k * 0.15, o.z + 0.3);
    f.userData = { dur: 0.5 + k * 0.12, max: 3.2 + k * 1.1, delay: k * 0.05 };
    f.material.opacity = 0;
  });
  ring.visible = true; ring.position.set(o.x, o.y + 0.3, o.z + 0.2);
  flash.position.set(o.x, o.y + 0.8, o.z + 0.8); flash.intensity = 7;
  for (var k = 0; k < 24; k++) {
    puff(new THREE.Vector3(o.x + (rnd() - 0.5) * 0.8, o.y + rnd() * 0.8, o.z + rnd() * 0.8),
         new THREE.Vector3(-S.speed * 0.6 + (rnd() - 0.5) * 3, 1 + rnd() * 3, 1 + rnd() * 3),
         1.4 + rnd() * 1.4, 2.5 + rnd() * 2, 0.75);
  }
  if (S.auto) { setCam("far"); }
}

// ---------- HUD ----------
var secCv = $("sec"), secCtx = secCv.getContext("2d");
function drawSection(temps) {
  var W = secCv.width, H = secCv.height;
  secCtx.clearRect(0, 0, W, H);
  secCtx.fillStyle = "rgba(11,19,43,0.78)"; secCtx.fillRect(0, 0, W, H);
  var sc = H * 0.62, cx = W / 2, cy = H * 0.50, j;
  for (j = 0; j < N; j++) {
    var s = sec(phiOf(j));
    secCtx.fillStyle = css(heat(temps[j]));
    secCtx.beginPath();
    secCtx.arc(cx + s.ax * sc * 1.9, cy - (s.rad - 0.78) * sc * 1.9, 3.4, 0, 6.2832);
    secCtx.fill();
  }
  P.d07.forEach(function (ph) {
    var s = sec(ph), x = cx + s.ax * sc * 1.9, y = cy - (s.rad - 0.78) * sc * 1.9;
    secCtx.fillStyle = "#ffffff"; secCtx.strokeStyle = "#0b132b"; secCtx.lineWidth = 2;
    secCtx.beginPath(); secCtx.moveTo(x, y - 7); secCtx.lineTo(x + 6, y); secCtx.lineTo(x, y + 7); secCtx.lineTo(x - 6, y); secCtx.closePath();
    secCtx.fill(); secCtx.stroke();
  });
  secCtx.fillStyle = "#ffffff"; secCtx.font = "bold 11px Arial"; secCtx.textAlign = "center";
  secCtx.fillText("CORTE DO PNEU", W / 2, 13);
  secCtx.font = "10px Arial";
  secCtx.fillText(P.d07.length ? "◆ = D07" : "sem D07", W / 2, H - 22);
  var gx = 14, gw = W - 28, gy = H - 14;
  for (var q = 0; q < gw; q++) {
    secCtx.fillStyle = css(heat(25 + 125 * q / gw)); secCtx.fillRect(gx + q, gy, 1, 6);
  }
}

function setBanner(txt, cls) { var b = $("banner"); b.textContent = txt; b.className = cls || ""; }
function showResult(title, text) {
  $("rTitle").textContent = title; $("rText").textContent = text;
  $("result").style.display = "block"; S.resultShown = true;
}

// ---------- loop ----------
var clock = performance.now();
function peakTemp(t) { return P.tamb + (P.tss - P.tamb) * (1 - Math.exp(-t / P.tau)); }

function tick(dt) {
  var sdt = S.running ? dt : 0;
  var fT = 1 - Math.exp(-S.t / P.tau), temps = [], j;
  for (j = 0; j < N; j++) { temps.push(P.tamb + (P.profile[j] - P.tamb) * fT); }

  // velocidade
  if (S.exploded) { S.speed = Math.max(0, S.speed - 4.5 * sdt); }
  else if (S.finished) { S.speed = Math.max(0, S.speed - 3 * sdt); }
  else if (S.started) { S.speed = Math.min(25, S.speed + 8 * sdt); }

  // relogio da simulacao e dano
  if (!S.exploded && !S.finished && S.started) {
    var dtmin = sdt * P.rate * S.mul;
    S.t += dtmin;
    var T = peakTemp(S.t);
    var fr = Math.max(0, (T - P.tlim) / (P.tper - P.tlim));
    S.dano = Math.min(1, S.dano + P.K * fr * dtmin);
    if (S.dano >= 1) { explode(); }
    else if (S.t >= P.horizon) { S.finished = true; S.fin = 0; }
  }

  // mundo e rodas
  var dx = S.speed * sdt;
  movers.forEach(function (o) { o.position.x -= dx; if (o.position.x < -HALFL) { o.position.x += LOOP; } });
  wheels.forEach(function (w) { w.spin.rotation.z -= (S.speed / R) * sdt; });

  // pneu monitorado: cor e deformacao
  if (S.started && !S.exploded) {
    paint(MON.geo, temps);
    var pos = MON.geo.attributes.position, base = MON.geo.userData.base;
    if (S.dano > 0.2) {
      var amp = 0.10 * S.dano * S.dano;
      for (var v = 0; v < pos.count; v++) {
        var jj = v % N, ph = phiOf(jj);
        var g = Math.exp(-Math.pow(angDiff(ph, SHOULDER), 2) / 0.3) + Math.exp(-Math.pow(angDiff(ph, -SHOULDER), 2) / 0.3);
        var fct = 1 + amp * g;
        pos.array[v * 3] = base[v * 3] * fct; pos.array[v * 3 + 1] = base[v * 3 + 1]; pos.array[v * 3 + 2] = base[v * 3 + 2] * fct;
      }
      pos.needsUpdate = true;
    }
  }

  // vibracao antes da falha e depois dela
  var shake = 0;
  if (S.started && !S.exploded && S.dano > 0.6) { shake = 0.012 * (S.dano - 0.6) / 0.4; }
  if (S.exploded) { shake = 0.28 * Math.exp(-S.boom * 2.2); }
  truck.position.y = (rnd() - 0.5) * shake * 2; truck.position.z = (rnd() - 0.5) * shake;
  if (S.exploded) {
    S.boom += sdt;
    body.rotation.x = lerp(body.rotation.x, 0.04, Math.min(1, 3 * sdt));
    truck.rotation.y = 0.05 * Math.exp(-S.boom * 1.2) * Math.sin(S.boom * 7);
  }

  // fumaca do pneu quente
  if (S.started && !S.exploded && S.dano > 0.35 && sdt > 0) {
    S.smokeAcc += sdt * (4 + 30 * S.dano);
    while (S.smokeAcc >= 1) {
      S.smokeAcc -= 1;
      puff(new THREE.Vector3(-1.8 + (rnd() - 0.5) * 0.4, 0.9, 1.15),
           new THREE.Vector3(-S.speed * 0.7 + (rnd() - 0.5), 0.8 + rnd(), 0.3 + rnd() * 0.6),
           0.5 + rnd() * 0.4, 1.2 + rnd(), 0.35 + 0.35 * S.dano);
    }
  }
  smoke.forEach(function (s) {
    if (!s.visible) { return; }
    var u = s.userData; u.life -= sdt;
    if (u.life <= 0) { s.visible = false; return; }
    var k = 1 - u.life / u.max;
    s.position.addScaledVector(u.vel, sdt);
    var sz = u.size * (1 + 2.6 * k); s.scale.set(sz, sz, 1);
    s.material.opacity = u.op * (1 - k);
  });
  fires.forEach(function (f) {
    if (!f.visible) { return; }
    var u = f.userData, tt = S.boom - u.delay;
    if (tt < 0) { f.material.opacity = 0; return; }
    if (tt > u.dur) { f.visible = false; return; }
    var k = tt / u.dur, sz = 0.6 + u.max * (1 - Math.pow(1 - k, 2));
    f.scale.set(sz, sz, 1); f.material.opacity = 1 - k;
  });
  if (ring.visible) {
    var kr = S.boom / 0.7;
    if (kr > 1) { ring.visible = false; }
    else { var rs = 1 + 14 * kr; ring.scale.set(rs, rs, 1); ring.material.opacity = 0.9 * (1 - kr); }
  }
  flash.intensity = S.exploded ? Math.max(0, 7 * (1 - S.boom / 0.45)) : 0;
  debris.forEach(function (d) {
    var u = d.userData;
    u.v.y -= 9.8 * sdt;
    d.position.addScaledVector(u.v, sdt);
    d.rotation.x += u.w.x * sdt; d.rotation.y += u.w.y * sdt; d.rotation.z += u.w.z * sdt;
    if (d.position.y < 0.04) {
      d.position.y = 0.04;
      if (Math.abs(u.v.y) > 1) { u.v.y *= -0.35; } else { u.v.y = 0; }
      u.v.x += (-S.speed - u.v.x) * Math.min(1, 4 * sdt);
      u.v.z *= (1 - Math.min(1, 2.5 * sdt)); u.w.multiplyScalar(1 - Math.min(1, 3 * sdt));
    } else {
      u.v.x += (-S.speed - u.v.x) * Math.min(1, 0.4 * sdt);
    }
    if (d.position.x < -HALFL) { d.visible = false; }
  });

  // camera automatica
  if (S.auto && S.started && !S.exploded && !S.finished) {
    if (S.dano > 0.5 && S.camMode !== "wheel") { setCam("wheel"); }
  }
  if (camGoal) {
    camera.position.lerp(camGoal.p, Math.min(1, 2.2 * dt));
    controls.target.lerp(camGoal.t, Math.min(1, 2.2 * dt));
  }
  controls.update();

  // HUD
  var Tp = peakTemp(S.t);
  $("hT").textContent = Math.round(S.t) + " min";
  $("hKm").textContent = Math.round(S.t / 60 * P.kmh) + " km";
  var hTemp = $("hTemp");
  hTemp.textContent = "Ombro " + Math.round(Tp) + " °C";
  hTemp.className = "chip " + (Tp >= P.tper ? "red" : (Tp >= P.tlim ? "orange" : "ok"));
  var integ = Math.round((1 - S.dano) * 100);
  $("hBarIn").style.width = integ + "%";
  $("hBarIn").style.background = integ > 60 ? "#2fbf71" : (integ > 30 ? "#f4b400" : "#e5383b");
  $("hInt").textContent = "Integridade do pneu " + integ + "%";
  if (S.exploded) { setBanner("💥 ESTOURO!", "red flash"); }
  else if (S.finished) { setBanner("✅ Chegou ao destino", "ok"); }
  else if (!S.started) { setBanner("Pneu monitorado: " + P.label, ""); }
  else if (S.dano > 0.7) { setBanner("🔴 FALHA IMINENTE", "red flash"); }
  else if (Tp >= P.tlim) { setBanner("🟠 Ombro acima do limite", "orange"); }
  else if (Tp >= 85) { setBanner("🟡 Pneu aquecendo", "yellow"); }
  else { setBanner("🟢 Temperatura normal", "ok"); }
  drawSection(temps);

  // resultado
  if (S.exploded && !S.resultShown && S.boom > 3.6) {
    showResult("💥 O pneu estourou aos " + Math.round(S.t) + " min (~" + Math.round(S.t / 60 * P.kmh) + " km)",
               P.msgFalha);
    if (S.auto) { setCam("far"); }
  }
  if (S.finished) {
    S.fin += sdt;
    if (!S.resultShown && S.fin > 2.5) {
      showResult("✅ O pneu chegou ao destino", P.msgOk + " Pico de " + Math.round(P.tss) + " °C no ombro.");
    }
  }
}

function frame(now) {
  var dt = Math.min(0.05, (now - clock) / 1000); clock = now;
  tick(dt);
  renderer.render(scene, camera);
  requestAnimationFrame(frame);
}

function resize() {
  var w = $("wrap").clientWidth || 600, h = $("wrap").clientHeight || 500;
  renderer.setSize(w, h, false);
  camera.aspect = w / h; camera.updateProjectionMatrix();
}
window.addEventListener("resize", resize);

// ---------- botoes ----------
$("start").onclick = function () {
  if (!S.started) { S.started = true; S.running = true; $("start").textContent = "⏸ Pausar"; $("intro").style.display = "none"; return; }
  S.running = !S.running; $("start").textContent = S.running ? "⏸ Pausar" : "▶ Continuar";
};
$("reset").onclick = function () { resetAll(); };
$("again").onclick = function () { resetAll(); S.started = true; S.running = true; $("start").textContent = "⏸ Pausar"; $("intro").style.display = "none"; };
[1, 2, 4].forEach(function (m) {
  $("x" + m).onclick = function () {
    S.mul = m;
    [1, 2, 4].forEach(function (k) { $("x" + k).className = k === m ? "on" : ""; });
  };
});
["side", "wheel", "rear", "top"].forEach(function (n) {
  $("cam_" + n).onclick = function () { S.auto = false; $("auto").className = ""; setCam(n); };
});
$("auto").onclick = function () { S.auto = !S.auto; $("auto").className = S.auto ? "on" : ""; };
$("auto").className = "on"; $("x1").className = "on";

resize(); setCam("side", true);
requestAnimationFrame(frame);
})();

</script>
</body>
</html>
'''


def prever_falha(t_ss):
    """Minuto em que o pneu falha, ou None. Mesma regra da animacao."""
    t, dano, dt = 0.0, 0.0, 0.05
    while t < SIM_HORIZONTE:
        temp = T_AMB + (t_ss - T_AMB) * (1.0 - np.exp(-t / TAU_MIN))
        dano += SIM_K * max(0.0, (temp - T_LIMITE) / (T_PERIGO - T_LIMITE)) * dt
        t += dt
        if dano >= 1.0:
            return float(t)
    return None


def parametros_caminhao(cenario, qtd, desl, carga):
    phi = np.linspace(-np.pi, np.pi, 97)
    perfil = temperatura(phi, qtd, desl, carga)
    tss = float(perfil.max())
    rotulos = {"Bem aplicado": ("PNEU COM D07", "ok"),
               "Mal posicionado": ("D07 FORA DO LUGAR", "bad"),
               "Ausente": ("PNEU SEM D07", "bad"),
               "Personalizado": ("D07 PERSONALIZADO", "ok" if qtd >= 0.9 and abs(desl) < 0.2 else "bad")}
    rotulo, tom = rotulos[cenario]
    falhas = {
        "Ausente": "Sem o D07, o calor se acumulou no ombro até o pneu falhar. É exatamente isso que o D07 evita.",
        "Mal posicionado": "O D07 estava fora do ombro e não protegeu o ponto mais quente. Posição também é qualidade.",
    }
    msg_falha = falhas.get(cenario, "A quantidade ou a posição do D07 não foi suficiente para proteger o ombro.")
    if cenario == "Bem aplicado":
        msg_ok = "Com o D07 no lugar certo, o calor do ombro foi dissipado e o pneu terminou o trajeto com segurança."
    elif tss > T_LIMITE:
        msg_ok = "O pneu não chegou a falhar nesta viagem, mas ficou acima do limite, o que acelera o desgaste."
    else:
        msg_ok = ("O erro passou despercebido nesta condição de uso. "
                  "Com carga mais pesada ou viagens longas, ele aparece.")
    return {
        "profile": [round(float(x), 2) for x in perfil], "tss": tss,
        "d07": [float(SHOULDER + desl), float(-(SHOULDER + desl))] if qtd > 0 else [],
        "tamb": T_AMB, "tlim": T_LIMITE, "tper": T_PERIGO, "tau": TAU_MIN,
        "K": SIM_K, "horizon": SIM_HORIZONTE, "rate": SIM_RATE, "kmh": SIM_KMH,
        "label": rotulo, "tone": tom, "msgFalha": msg_falha, "msgOk": msg_ok,
    }


def html_caminhao(params):
    return SIM_HTML.replace("__PARAMS__", json.dumps(params))


# =====================================================================
# 4) TELAS
# =====================================================================
def controles(prefixo, padrao):
    """Controles do cenario do D07 (usados na aba Aprender e na aba Caminhao)."""
    opcoes = ["Bem aplicado", "Mal posicionado", "Ausente", "Personalizado"]
    cenario = st.radio("Como o D07 foi aplicado?", opcoes, index=opcoes.index(padrao),
                       horizontal=True, key=f"{prefixo}_cenario")
    if cenario == "Personalizado":
        qtd = st.slider("Quantidade de D07 aplicada (%)", 0, 100, 100, 5,
                        key=f"{prefixo}_qtd") / 100
        desl_pct = st.slider("Posição: deslocamento do ombro (%)", -100, 100, 0, 10,
                             key=f"{prefixo}_desl",
                             help="0 = posição correta. Negativo = em direção ao topo. "
                                  "Positivo = em direção ao flanco.")
        desl = desl_pct / 100 * DESL_MAX
    else:
        qtd, desl = PRESETS[cenario]
    uso = st.select_slider("Condição de uso do pneu", options=list(CARGAS), value="Pesada",
                           key=f"{prefixo}_uso")
    return cenario, qtd, desl, CARGAS[uso]


def aba_caminhao():
    st.subheader("Viagem do caminhão em 3D")
    st.write("Veja o caminhão na estrada com o pneu monitorado. O calor sobe no ombro e, "
             "se o D07 não estiver certo, o pneu pode **estourar**.")
    cenario, qtd, desl, carga = controles("cam", "Ausente")
    params = parametros_caminhao(cenario, qtd, desl, carga)
    falha = prever_falha(params["tss"])
    c1, c2 = st.columns(2)
    c1.metric("Pico no ombro", f"{params['tss']:.0f} °C")
    c2.metric("Resultado previsto",
              "Sem falha" if falha is None else f"Estouro aos ~{falha:.0f} min")
    components.html(html_caminhao(params), height=640)
    st.caption("Animação ilustrativa e com tempo acelerado. Os valores de temperatura e o momento "
               "da falha vêm de um modelo didático, não de ensaios reais. "
               "Se o 3D não aparecer, verifique se a rede permite carregar o Three.js.")


def painel_simulacao():
    st.subheader("Simulação: o pneu rodando")
    cenario, qtd, desl, carga = controles("sim", "Bem aplicado")

    t_max = pico(qtd, desl, carga)
    t_lim = tempo_ate_limite(t_max)
    c1, c2, c3 = st.columns(3)
    c1.metric("Pico no ombro", f"{t_max:.0f} °C")
    c2.metric("Passa do limite em", "não passa" if t_lim is None else f"~{t_lim:.0f} min")
    n = nivel(t_max)
    if n == "critico":
        c3.error("🔴 Risco crítico: o pneu pode estourar")
    elif n == "alerta":
        c3.warning("🟠 Acima do limite: falha precoce")
    else:
        c3.success("🟢 Dentro do limite")

    aba3d, abacorte, abatempo = st.tabs(["Pneu 3D", "Corte do pneu", "Aquecimento"])
    with aba3d:
        st.plotly_chart(figura_3d(qtd, desl, carga), use_container_width=True)
        st.caption("A faixa escura mostra onde o D07 foi aplicado. "
                   "Cores quentes = mais calor. Arraste para girar, ou toque em ▶ Girar.")
    with abacorte:
        st.plotly_chart(figura_corte(qtd, desl, carga), use_container_width=True)
        st.caption("Corte transversal: o ponto mais quente fica no ombro, onde o D07 deve estar.")
    with abatempo:
        cenarios = dict(PRESETS)
        if cenario == "Personalizado":
            cenarios["Personalizado"] = (qtd, desl)
        st.plotly_chart(figura_tempo(cenarios, cenario, carga), use_container_width=True)
        st.caption("Quanto mais tempo rodando, mais o ombro esquenta. "
                   "Em uso leve o erro pode passar despercebido; com carga pesada, ele aparece.")
    st.caption("Modelo ilustrativo para treinamento. Não representa dados reais de engenharia.")


def aba_aprender():
    nome = st.selectbox("Escolha o produto", list(PRODUTOS))
    info = PRODUTOS[nome]
    st.markdown(f"**Onde fica:** {info['onde']}")
    st.markdown(f"**Para que serve:** {info['funcao']}")
    st.warning(f"**Se errar:** {info['se_errar']}")
    st.divider()
    if info["simulavel"]:
        painel_simulacao()
    else:
        st.info("Simulação ainda não criada para este produto. "
                "Preencher com a Qualidade e replicar o modelo do D07.")


def aba_quiz():
    st.subheader("Quiz: você sabe o porquê?")
    st.write("Responda **antes** e **depois** de usar o simulador. "
             "A diferença mostra o quanto o treinamento ajudou.")
    perguntas = PRODUTOS["D07 (ombro)"]["quiz"]

    if "resultados" not in st.session_state:
        st.session_state.resultados = []
    if "tentativa" not in st.session_state:
        st.session_state.tentativa = 0

    rodada = st.radio("Esta rodada é:", ["Antes", "Depois"], horizontal=True,
                      help="Antes ou depois de usar o simulador.")
    quem = st.text_input("Identificação (opcional, ex.: iniciais)")

    with st.form(f"quiz_{st.session_state.tentativa}"):
        resp = []
        for i, q in enumerate(perguntas):
            resp.append(st.radio(f"{i + 1}. {q['p']}", q["op"], index=None,
                                 key=f"q{st.session_state.tentativa}_{i}"))
        enviar = st.form_submit_button("Ver resultado")

    if enviar:
        if any(r is None for r in resp):
            st.warning("Responda todas as perguntas para ver o resultado.")
        else:
            acertos = 0
            for i, q in enumerate(perguntas):
                certo = resp[i] == q["op"][q["ok"]]
                acertos += certo
                (st.success if certo else st.error)(
                    f"{i + 1}. {'Certo' if certo else 'Errado'}. {q['exp']}")
            pct = round(100 * acertos / len(perguntas))
            st.metric("Seu resultado", f"{acertos}/{len(perguntas)}", f"{pct}%")
            st.session_state.resultados.append({
                "data": datetime.now().strftime("%d/%m/%Y %H:%M"),
                "pessoa": quem or "-", "rodada": rodada,
                "acertos": acertos, "total": len(perguntas), "percentual": pct})
            if st.button("Responder de novo"):
                st.session_state.tentativa += 1
                st.rerun()

    res = st.session_state.resultados
    if res:
        st.divider()
        st.subheader("Resultados desta sessão")
        antes = [r["percentual"] for r in res if r["rodada"] == "Antes"]
        depois = [r["percentual"] for r in res if r["rodada"] == "Depois"]
        m1, m2 = st.columns(2)
        if antes:
            m1.metric("Média ANTES", f"{np.mean(antes):.0f}%")
        if depois:
            delta = f"{np.mean(depois) - np.mean(antes):+.0f} pontos" if antes else None
            m2.metric("Média DEPOIS", f"{np.mean(depois):.0f}%", delta)
        st.table(res)
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=list(res[0]))
        w.writeheader()
        w.writerows(res)
        st.download_button("Baixar resultados (CSV)", buf.getvalue().encode("utf-8-sig"),
                           file_name="resultados_quiz.csv", mime="text/csv")
        st.caption("Os resultados ficam só nesta sessão. Baixe o CSV para usar no relatório.")


def aba_sobre():
    st.subheader("Sobre este app")
    st.markdown(
        "- **Objetivo:** mostrar o porquê de cada produto aplicado na montagem do pneu, "
        "para reduzir defeitos de qualidade causados por desconhecimento.\n"
        "- **Como usar:** escolha o produto, veja a função, mude o cenário e observe o pneu. "
        "Na aba Caminhão 3D, acompanhe a viagem até o pneu aguentar ou estourar. "
        "Depois faça o quiz.\n"
        "- **Cenários do D07:** bem aplicado, mal posicionado, ausente, ou personalizado "
        "(você controla a quantidade e a posição).")
    st.warning("**Importante:** a simulação é didática. Os valores de temperatura são "
               "ilustrativos e não vêm de ensaios. O conteúdo técnico deve ser validado "
               "pela área de Qualidade antes do uso oficial.")
    st.caption("Trabalho acadêmico: Kaique de Brito Menezes, IBMR.")


def exigir_senha():
    """Pede senha antes de mostrar o app. A senha NAO fica no codigo:
    ela vem de st.secrets["APP_SENHA"] (veja o passo a passo no chat)."""
    try:
        senha = st.secrets["APP_SENHA"]
    except Exception:
        senha = None
    if not senha:
        st.warning("Modo teste: nenhuma senha configurada (APP_SENHA). "
                   "Configure antes de compartilhar o link.")
        return
    if st.session_state.get("autenticado"):
        return
    st.title("🔒 Acesso restrito")
    st.caption("Digite a senha fornecida pelo seu líder para entrar.")
    with st.form("login"):
        digitada = st.text_input("Senha", type="password")
        entrar = st.form_submit_button("Entrar")
    if entrar:
        if hmac.compare_digest(digitada.encode("utf-8"), str(senha).encode("utf-8")):
            st.session_state["autenticado"] = True
            st.rerun()
        else:
            st.error("Senha incorreta.")
    st.stop()


# =====================================================================
# 5) APP
# =====================================================================
exigir_senha()

st.title("🛞 Por que esse produto?")
st.caption("Entenda a função de cada produto com o pneu rodando na estrada.")

tab1, tab2, tab3, tab4 = st.tabs(["📘 Aprender", "🚛 Caminhão 3D", "🧪 Quiz", "ℹ️ Sobre"])
with tab1:
    aba_aprender()
with tab2:
    aba_caminhao()
with tab3:
    aba_quiz()
with tab4:
    aba_sobre()
