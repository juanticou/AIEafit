import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("Simulador MDP GridWorld 10x10")

# 1. Controles en la barra lateral
gamma = st.sidebar.slider("Factor de Descuento (γ)", 0.50, 0.99, 0.90, 0.01)
p_success = st.sidebar.slider("Probabilidad de Éxito", 0.50, 1.00, 0.85, 0.05)
penalty = st.sidebar.slider("Penalización de Obstáculos", -100, -10, -35, 5)

GRID_SIZE = 10
START, GOAL = (0, 0), (9, 9)
p_slip = (1.0 - p_success) / 2.0

# 2. Configurar Grid y Recompensas
np.random.seed(42)
rewards = np.random.uniform(0.1, 1.0, size=(GRID_SIZE, GRID_SIZE))
rewards[START], rewards[GOAL] = 0.0, 100.0

obstacles = [(1, 2), (2, 2), (3, 2), (4, 5), (5, 5), (6, 5), (7, 3), (8, 3), (2, 7), (3, 7), (4, 7), (7, 7), (8, 7), (0, 5)]
for obs in obstacles:
    rewards[obs] = float(penalty)

# 3. Iteración de Valor (Value Iteration)
actions = {'UP': (-1, 0), 'DOWN': (1, 0), 'LEFT': (0, -1), 'RIGHT': (0, 1)}
action_keys = list(actions.keys())
perpendiculars = {'UP': ['LEFT', 'RIGHT'], 'DOWN': ['LEFT', 'RIGHT'], 'LEFT': ['UP', 'DOWN'], 'RIGHT': ['UP', 'DOWN']}

def next_state(s, a):
    r, c = s
    dr, dc = actions[a]
    nr, nc = r + dr, c + dc
    return (nr, nc) if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE else s

V = np.zeros((GRID_SIZE, GRID_SIZE))
V[GOAL] = 100.0

for _ in range(200):
    V_new = V.copy()
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if (r, c) == GOAL: continue
            vals = []
            for a in action_keys:
                s_main = next_state((r, c), a)
                v = p_success * (rewards[s_main] + gamma * V[s_main])
                for perp in perpendiculars[a]:
                    s_p = next_state((r, c), perp)
                    v += p_slip * (rewards[s_p] + gamma * V[s_p])
                vals.append(v)
            V_new[r, c] = max(vals)
    V = V_new

# 4. Extraer Ruta Óptima
current = START
path = [current]
total_r, step = 0.0, 0
while current != GOAL and step < 50:
    best_a, best_val = None, -float('inf')
    for a in action_keys:
        s_m = next_state(current, a)
        v = p_success * (rewards[s_m] + gamma * V[s_m])
        if v > best_val: best_val, best_a = v, a
    current = next_state(current, best_a)
    path.append(current)
    total_r += rewards[current]
    step += 1

# 5. Renderizar Gráfica
fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(rewards, cmap='YlGn', alpha=0.7)

for obs in obstacles:
    ax.add_patch(plt.Rectangle((obs[1]-0.5, obs[0]-0.5), 1, 1, color='crimson', alpha=0.8))
ax.add_patch(plt.Rectangle((GOAL[1]-0.5, GOAL[0]-0.5), 1, 1, color='gold'))
ax.add_patch(plt.Rectangle((START[1]-0.5, START[0]-0.5), 1, 1, color='skyblue'))

path_r, path_c = zip(*path)
ax.plot(path_c, path_r, color='blue', marker='o', linewidth=3, label='Ruta Óptima')
ax.set_xticks(range(GRID_SIZE))
ax.set_yticks(range(GRID_SIZE))
plt.legend()

# Mostrar métricas y gráfica en la Web App
st.metric("Pasos Recorridos", step)
st.metric("Recompensa Total Acumulada", f"{total_r:.2f}")
st.pyplot(fig)