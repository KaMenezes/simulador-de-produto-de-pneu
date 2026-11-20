# -*- coding: utf-8 -*-
"""
Simulador didático de produtos de pneu (protótipo D07)
Instalar:  pip install streamlit plotly numpy
Rodar:     streamlit run app.py

ATENÇÃO: modelo ILUSTRATIVO, não é cálculo de engenharia.
Serve para mostrar o conceito: função do produto -> consequência do erro.
"""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Por que esse produto?", page_icon="🛞", layout="centered")

# ---------------------------------------------------------------
# 1) CATÁLOGO DE PRODUTOS (edite/complete na Sprint 1 com a Qualidade)
# ---------------------------------------------------------------
PRODUTOS = {
    "D07 (ombro)": {
        "onde": "Ombro do pneu (zona de transição entre o topo e o flanco).",
        "funcao": "Ajuda a dissipar o calor gerado nas extremidades quando o pneu roda.",
        "se_errar": "Sem o D07 (ou mal posicionado), o ombro superaquece, "
                    "o que pode causar separação de camadas e até estouro.",
        "simulavel": True,
    },
    "Napas de topo": {
        "onde": "Sob a banda de rodagem (a preencher).",
        "funcao": "A preencher com a equipe de Qualidade.",
        "se_errar": "A preencher.",
        "simulavel": False,
    },
    "Produto zona baixa": {
        "onde": "Zona baixa do pneu (a preencher).",
        "funcao": "A preencher com a equipe de Qualidade.",
        "se_errar": "A preencher.",
        "simulavel": False,
    },
    "Produto zona alta": {
        "onde": "Zona alta do pneu (a preencher).",
        "funcao": "A preencher com a equipe de Qualidade.",
        "se_errar": "A preencher.",
        "simulavel": False,
    },
}

# ---------------------------------------------------------------
# 2) MODELO SIMPLES DE TEMPERATURA (valores ilustrativos)
# ---------------------------------------------------------------
T_AMB = 25.0          # °C
T_LIMITE = 110.0      # °C -> acima disso, risco (ilustrativo)
T_PERIGO = 130.0      # °C -> risco crítico (ilustrativo)
SHOULDER = np.pi / 4  # posição do ombro na seção (rad)


def gauss(phi, centro, largura):
    d = np.angle(np.exp(1j * (phi - centro)))  # diferenca angular com wrap
    return np.exp(-(d ** 2) / (2 * largura ** 2))


def temperatura(phi, cenario, carga):
    # Temperatura em funcao da posicao na secao do pneu (phi)
    q = 45.0 + 60.0 * carga  # calor gerado (carga 0..1)
    base = q * (0.35 + 0.65 * (gauss(phi, SHOULDER, 0.35) + gauss(phi, -SHOULDER, 0.35)))
    if cenario == "Bem aplicado":
        cobertura = gauss(phi, SHOULDER, 0.45) + gauss(phi, -SHOULDER, 0.45)
        base = base * (1 - 0.55 * cobertura)
    elif cenario == "Mal posicionado":
        cobertura = gauss(phi, SHOULDER + 0.9, 0.30) + gauss(phi, -SHOULDER - 0.9, 0.30)
        base = base * (1 - 0.55 * cobertura)
    # "Ausente": sem redução
    return T_AMB + base


# ---------------------------------------------------------------
# 3) GEOMETRIA 3D DO PNEU
# ---------------------------------------------------------------
def malha_pneu(cenario, carga, n_phi=60, n_theta=80):
    phi = np.linspace(-np.pi, np.pi, n_phi)
    theta = np.linspace(0, 2 * np.pi, n_theta)
    P, TH = np.meshgrid(phi, theta, indexing="ij")
    Rc, a, b, p = 0.78, 0.22, 0.26, 0.55  # raio central, meia-altura, meia-largura, "quadratura"
    c, s = np.cos(P), np.sin(P)
    rad = Rc + a * np.sign(c) * np.abs(c) ** p
    ax = b * np.sign(s) * np.abs(s) ** p
    X, Z, Y = rad * np.cos(TH), rad * np.sin(TH), ax
    T = temperatura(P, cenario, carga)
    return X, Y, Z, T


def figura(cenario, carga):
    X, Y, Z, T = malha_pneu(cenario, carga)
    surf = go.Surface(
        x=X, y=Y, z=Z, surfacecolor=T, cmin=T_AMB, cmax=T_PERIGO + 20,
        colorscale="Turbo", showscale=True,
        colorbar=dict(title="°C", thickness=12, len=0.7),
    )
    frames = []
    for k in range(0, 360, 10):
        t = np.radians(k)
        frames.append(go.Frame(layout=dict(scene_camera=dict(
            up=dict(x=0, y=1, z=0),
            eye=dict(x=2.0 * np.cos(t), y=0.9, z=2.0 * np.sin(t))))))
    fig = go.Figure(data=[surf], frames=frames)
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0), height=420,
        scene=dict(
            xaxis_visible=False, yaxis_visible=False, zaxis_visible=False,
            aspectmode="data",
            camera=dict(up=dict(x=0, y=1, z=0), eye=dict(x=2.0, y=0.9, z=0.0)),
        ),
        updatemenus=[dict(
            type="buttons", showactive=False, x=0.02, y=0.02,
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


def figura_perfil(cenario, carga):
    phi = np.linspace(-np.pi, np.pi, 300)
    fig = go.Figure()
    for nome in ["Ausente", "Bem aplicado", "Mal posicionado"]:
        fig.add_trace(go.Scatter(
            x=np.degrees(phi), y=temperatura(phi, nome, carga), name=nome,
            line=dict(width=4 if nome == cenario else 1.5),
            opacity=1 if nome == cenario else 0.35))
    fig.add_hline(y=T_LIMITE, line_dash="dash", annotation_text="limite")
    fig.update_layout(
        height=260, margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="Posição na seção do pneu (°)", yaxis_title="Temperatura (°C)",
        legend=dict(orientation="h", y=-0.35))
    return fig


# ---------------------------------------------------------------
# 4) INTERFACE
# ---------------------------------------------------------------
st.title("🛞 Por que esse produto?")
st.caption("Entenda a função de cada produto com o pneu rodando na estrada.")

produto = st.selectbox("Escolha o produto", list(PRODUTOS.keys()))
info = PRODUTOS[produto]

st.markdown(f"**Onde fica:** {info['onde']}")
st.markdown(f"**Para que serve:** {info['funcao']}")
st.markdown(f"**Se errar:** {info['se_errar']}")

if info["simulavel"]:
    st.divider()
    st.subheader("Simulação: o pneu rodando")
    cenario = st.radio("Como o D07 foi aplicado?",
                       ["Bem aplicado", "Mal posicionado", "Ausente"], horizontal=True)
    carga = st.slider("Carga / velocidade (ilustrativo)", 0.0, 1.0, 0.6, 0.05)

    t_max = float(temperatura(np.linspace(-np.pi, np.pi, 400), cenario, carga).max())
    st.plotly_chart(figura(cenario, carga), use_container_width=True)

    c1, c2 = st.columns(2)
    c1.metric("Temperatura máx. no ombro", f"{t_max:.0f} °C")
    if t_max >= T_PERIGO:
        c2.error("🔴 Risco crítico: separação de camadas / estouro")
    elif t_max >= T_LIMITE:
        c2.warning("🟠 Acima do limite: desgaste e falha precoce")
    else:
        c2.success("🟢 Dentro do limite")

    st.plotly_chart(figura_perfil(cenario, carga), use_container_width=True)
    st.caption("Modelo ilustrativo para treinamento. Não representa dados reais de engenharia.")
else:
    st.info("Simulação ainda não criada para este produto. "
            "Preencher com a Qualidade e replicar o modelo do D07.")
