import streamlit as st
import numpy as np
import pickle
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen import initializers
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import griddata
import time

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ASSB Digital Twin",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 28px;
        font-weight: 600;
        color: #4a90d9;
        text-align: center;
        margin-bottom: 4px;
    }
    .sub-title {
        font-size: 13px;
        color: #888;
        text-align: center;
        margin-bottom: 20px;
    }
    .metric-card {
        background: #16213e;
        border: 1px solid #4a90d9;
        border-radius: 10px;
        padding: 14px;
        text-align: center;
    }
    .metric-value {
        font-size: 24px;
        font-weight: 600;
        color: #4a90d9;
    }
    .metric-label {
        font-size: 12px;
        color: #aaa;
        margin-top: 4px;
    }
    .metric-unc {
        font-size: 11px;
        color: #f39c12;
        margin-top: 2px;
    }
    .flip-container {
        perspective: 1200px;
        width: 100%;
    }
    .badge-choice {
        background: #2d1b00;
        border: 1px solid #f39c12;
        border-radius: 6px;
        padding: 4px 10px;
        font-size: 11px;
        color: #f39c12;
        display: inline-block;
        margin-bottom: 8px;
    }
    .r2-badge {
        background: #0a2818;
        border: 1px solid #27ae60;
        border-radius: 6px;
        padding: 3px 8px;
        font-size: 11px;
        color: #27ae60;
        display: inline-block;
        margin: 2px;
    }
</style>
""", unsafe_allow_html=True)

# ── Session state for flip ─────────────────────────────────────────────────────
if "show_inverse" not in st.session_state:
    st.session_state.show_inverse = False
if "results" not in st.session_state:
    st.session_state.results = None

# ── Model definitions ─────────────────────────────────────────────────────────
class SpectralConv1d(nn.Module):
    out_channels: int
    n_modes: int
    @nn.compact
    def __call__(self, x):
        in_ch = x.shape[-1]
        W_re = self.param("W_re", initializers.normal(0.02),
                          (in_ch, self.out_channels, self.n_modes))
        W_im = self.param("W_im", initializers.normal(0.02),
                          (in_ch, self.out_channels, self.n_modes))
        x_ft = jnp.fft.rfft(x, axis=1)[:, :self.n_modes, :]
        out_re = (jnp.einsum("bmi,iom->bmo", x_ft.real, W_re) -
                  jnp.einsum("bmi,iom->bmo", x_ft.imag, W_im))
        out_im = (jnp.einsum("bmi,iom->bmo", x_ft.real, W_im) +
                  jnp.einsum("bmi,iom->bmo", x_ft.imag, W_re))
        out_ft = jnp.zeros((x.shape[0], x.shape[1] // 2 + 1,
                            self.out_channels), dtype=jnp.complex64)
        out_ft = out_ft.at[:, :self.n_modes, :].set(out_re + 1j * out_im)
        return jnp.fft.irfft(out_ft, n=x.shape[1], axis=1)

class FNOBlock(nn.Module):
    channels: int
    n_modes: int
    dropout_rate: float = 0.15
    @nn.compact
    def __call__(self, x, training=False):
        out = nn.gelu(SpectralConv1d(self.channels, self.n_modes)(x) +
                      nn.Dense(self.channels)(x))
        return nn.Dropout(rate=self.dropout_rate,
                          deterministic=not training)(out)

class FNO(nn.Module):
    n_modes: int = 24
    channels: int = 64
    n_layers: int = 6
    n_field_out: int = 3
    dropout_rate: float = 0.15
    @nn.compact
    def __call__(self, x_in, x_mesh, training=False):
        batch, n_elem, _ = x_mesh.shape
        x = jnp.concatenate(
            [jnp.broadcast_to(x_in[:, None, :], (batch, n_elem, 3)),
             x_mesh], axis=-1)
        x = nn.Dense(self.channels)(x)
        for _ in range(self.n_layers):
            x = FNOBlock(self.channels, self.n_modes,
                         self.dropout_rate)(x, training=training)
        x = nn.Dense(self.channels)(x)
        x = nn.gelu(x)
        return nn.sigmoid(nn.Dense(self.n_field_out)(x))

class DeepONet(nn.Module):
    hidden: int = 128
    n_layers: int = 4
    n_out: int = 2
    dropout_rate: float = 0.15
    @nn.compact
    def __call__(self, x_in, training=False):
        x = x_in
        for _ in range(self.n_layers):
            x = nn.gelu(nn.Dense(self.hidden)(x))
            x = nn.Dropout(rate=self.dropout_rate,
                           deterministic=not training)(x)
        return nn.sigmoid(nn.Dense(self.n_out)(x))

# ── Load models (cached) ───────────────────────────────────────────────────────
@st.cache_resource
def load_all():
    with open("models/fno_bundle_final.pkl", "rb") as f:
        fno_b = pickle.load(f)
    with open("models/don_bundle_final.pkl", "rb") as f:
        don_b = pickle.load(f)
    with open("models/inverse_bundle_final.pkl", "rb") as f:
        inv_b = pickle.load(f)
    fno_model = FNO()
    don_model = DeepONet()
    return fno_b, don_b, inv_b, fno_model, don_model

fno_b, don_b, inv_b, fno_model, don_model = load_all()

in_min    = np.array(fno_b["normalization"]["in_min"])
in_max    = np.array(fno_b["normalization"]["in_max"])
field_min = np.array(fno_b["normalization"]["field_min"])
field_max = np.array(fno_b["normalization"]["field_max"])
scal_min  = np.array(don_b["normalization"]["scal_min"])
scal_max  = np.array(don_b["normalization"]["scal_max"])
x_mesh_np = fno_b["mesh"]["x_mesh_np"]
n_elem    = fno_b["mesh"]["n_elem"]
x_mesh_base = jnp.array(x_mesh_np)[None, :, :]

# ── Inference function ────────────────────────────────────────────────────────
def run_forward(P, C_rate, tau, n_mc=50):
    x_in_raw = np.array([[P, C_rate, tau]], dtype=np.float32)
    x_in_n   = (x_in_raw - in_min) / (in_max - in_min)
    x_in_j   = jnp.array(x_in_n)
    x_mesh_j = jnp.broadcast_to(x_mesh_base, (1, n_elem, 2))

    key = jax.random.PRNGKey(42)
    mc_fields = []
    mc_scals  = []

    for _ in range(n_mc):
        key, sk1, sk2 = jax.random.split(key, 3)
        pf = fno_model.apply(fno_b["params"], x_in_j, x_mesh_j,
                             training=True, rngs={"dropout": sk1})
        ps = don_model.apply(don_b["params"], x_in_j,
                             training=True, rngs={"dropout": sk2})
        mc_fields.append(np.array(pf[0]))
        mc_scals.append(np.array(ps[0]))

    mc_f = np.stack(mc_fields)
    mc_s = np.stack(mc_scals)

    mean_f = mc_f.mean(0)
    std_f  = mc_f.std(0)
    mean_s = mc_s.mean(0)
    std_s  = mc_s.std(0)

    # Denormalize fields
    def denorm_field(arr, i):
        return arr * (field_max[i] - field_min[i]) + field_min[i]

    vm_mean    = denorm_field(mean_f[:, 0], 0) / 1e6
    vm_std     = denorm_field(std_f[:, 0],  0) / 1e6
    xi_mean    = denorm_field(mean_f[:, 1], 1)
    xi_std     = denorm_field(std_f[:, 1],  1)
    sxx_mean   = denorm_field(mean_f[:, 2], 2) / 1e6
    sxx_std    = denorm_field(std_f[:, 2],  2) / 1e6

    # Denormalize scalars
    V_mean  = float(mean_s[0]) * (scal_max[0] - scal_min[0]) + scal_min[0]
    V_std   = float(std_s[0])  * (scal_max[0] - scal_min[0])
    cap_mean= float(mean_s[1]) * (scal_max[1] - scal_min[1]) + scal_min[1]
    cap_std = float(std_s[1])  * (scal_max[1] - scal_min[1])

    return {
        "vm_mean": vm_mean, "vm_std": vm_std,
        "xi_mean": xi_mean, "xi_std": xi_std,
        "sxx_mean": sxx_mean, "sxx_std": sxx_std,
        "V_mean": V_mean, "V_std": V_std,
        "cap_mean": cap_mean, "cap_std": cap_std,
    }

# ── Heatmap builder ────────────────────────────────────────────────────────────
def make_heatmap(values, x_mesh, title, unit, colorscale="Jet",
                 vmin=None, vmax=None):
    bx = x_mesh[:, 0]
    by = x_mesh[:, 1]
    fig = go.Figure()

    # Scatter plot للـ field values
    fig.add_trace(go.Scatter(
        x=bx, y=by,
        mode="markers",
        marker=dict(
            size=3,
            color=values,
            colorscale=colorscale,
            cmin=vmin, cmax=vmax,
            colorbar=dict(
                title=dict(text=unit, font=dict(size=11)),
                thickness=12),
            showscale=True,
        ),
        hovertemplate=f"{unit}: %{{marker.color:.4f}}<extra></extra>",
    ))

    # Particle boundaries overlay
    particles_info = [
        {"cx": 0.28, "cy": 0.28, "R": 6.0e-6/40e-6},
        {"cx": 0.72, "cy": 0.20, "R": 8.0e-6/40e-6},
        {"cx": 0.80, "cy": 0.70, "R": 6.8e-6/40e-6},
        {"cx": 0.28, "cy": 0.75, "R": 7.2e-6/40e-6},
        {"cx": 0.55, "cy": 0.48, "R": 4.0e-6/40e-6},
    ]
    theta = np.linspace(0, 2*np.pi, 60)
    for p in particles_info:
        fig.add_trace(go.Scatter(
            x=p["cx"] + p["R"]*np.cos(theta),
            y=p["cy"] + p["R"]*np.sin(theta),
            mode="lines",
            line=dict(color="white", width=2),
            showlegend=False,
            hoverinfo="skip",
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=13), x=0.5),
        xaxis=dict(showticklabels=False, showgrid=False, scaleanchor="y"),
        yaxis=dict(showticklabels=False, showgrid=False),
        margin=dict(l=0, r=0, t=36, b=0),
        height=260,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig
                     
# ── V-cap curve ────────────────────────────────────────────────────────────────
def make_vcap_curve(tau_val, V_mean, V_std, cap_mean, cap_std):
    tau_vals = np.linspace(0.05, tau_val, 12)
    cap_vals = tau_vals * cap_mean / tau_val if tau_val > 0 else [0]
    V_vals   = np.linspace(2.99, V_mean, len(tau_vals))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cap_vals, y=V_vals + V_std,
        fill=None, mode="lines",
        line=dict(width=0, color="#f39c12"),
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=cap_vals, y=V_vals - V_std,
        fill="tonexty", mode="lines",
        line=dict(width=0, color="#f39c12"),
        fillcolor="rgba(243,156,18,0.15)",
        name="±1σ (MC Dropout)",
    ))
    fig.add_trace(go.Scatter(
        x=cap_vals, y=V_vals,
        mode="lines+markers",
        line=dict(color="#4a90d9", width=2),
        marker=dict(size=5),
        name="V-cap (surrogate)",
    ))
    fig.add_hline(y=4.2, line_dash="dash",
                  line_color="#e74c3c", line_width=1,
                  annotation_text="4.2V cutoff")
    fig.update_layout(
        title=dict(text="V-cap Curve", font=dict(size=13), x=0.5),
        xaxis_title="Capacity (mAh/g)",
        yaxis_title="V_cell (V)",
        yaxis=dict(range=[2.9, 4.3]),
        legend=dict(font=dict(size=10)),
        margin=dict(l=40, r=10, t=36, b=36),
        height=260,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(14,17,23,0.8)",
    )
    return fig

# ── Pareto front for inverse ──────────────────────────────────────────────────
def make_pareto(results, xi_max_limit, cap_min_limit):
    all_xi  = [r["xi_max"] for r in results]
    all_cap = [r["cap"]    for r in results]
    all_P   = [r["P"] / 1e6 for r in results]
    all_Cr  = [r["C_rate"]  for r in results]

    # Feasible points
    feasible = [(xi, cap, P, Cr)
                for xi, cap, P, Cr in zip(all_xi, all_cap, all_P, all_Cr)
                if xi <= xi_max_limit and cap >= cap_min_limit]

    fig = go.Figure()

    # All points
    fig.add_trace(go.Scatter(
        x=all_cap, y=all_xi,
        mode="markers",
        marker=dict(size=4, color="#888", opacity=0.4),
        name="All points",
    ))

    # Feasible points
    if feasible:
        fxi, fcap, fP, fCr = zip(*feasible)
        fig.add_trace(go.Scatter(
            x=fcap, y=fxi,
            mode="markers",
            marker=dict(size=7, color="#27ae60",
                        colorscale="Viridis",
                        showscale=False),
            name=f"Feasible ({len(feasible)} pts)",
            hovertemplate=(
                "cap=%{x:.1f} mAh/g<br>"
                "ξ_max=%{y:.3f}<br>"
                "<extra></extra>"
            ),
        ))

    # Constraint lines
    fig.add_hline(y=xi_max_limit, line_dash="dash",
                  line_color="#e74c3c", line_width=1,
                  annotation_text=f"ξ_max ≤ {xi_max_limit:.2f}")
    fig.add_vline(x=cap_min_limit, line_dash="dash",
                  line_color="#4a90d9", line_width=1,
                  annotation_text=f"cap ≥ {cap_min_limit:.0f}")

    fig.update_layout(
        title=dict(text="Pareto Front: Damage vs Capacity",
                   font=dict(size=13), x=0.5),
        xaxis_title="Capacity (mAh/g)",
        yaxis_title="ξ_max (damage)",
        legend=dict(font=dict(size=10)),
        margin=dict(l=40, r=10, t=36, b=36),
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(14,17,23,0.8)",
    )
    return fig, feasible

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="main-title">⚡ ASSB Digital Twin Surrogate</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Physics-informed FNO + DeepONet surrogate — '
    'Taghikhani & Kee 2025 | '
    'R²(ξ)=0.965 · R²(V)=0.9999 · MC Dropout UQ</div>',
    unsafe_allow_html=True)

# R² badges
st.markdown(
    '<div style="text-align:center;margin-bottom:16px">'
    '<span class="r2-badge">FNO: vm R²=0.944</span>'
    '<span class="r2-badge">FNO: ξ R²=0.965</span>'
    '<span class="r2-badge">FNO: σ_xx R²=0.935</span>'
    '<span class="r2-badge">DON: V_cell R²=0.9999</span>'
    '<span class="r2-badge">DON: cap R²=0.9998</span>'
    '</div>',
    unsafe_allow_html=True)

# CHOICE badge
st.markdown(
    '<div style="text-align:center">'
    '<span class="badge-choice">'
    '[CHOICE geometry: 5 NMC particles, NOT paper SEM] '
    '[CHOICE: galvanostatic, not full BV]'
    '</span></div>',
    unsafe_allow_html=True)

# Flip button
col_flip1, col_flip2, col_flip3 = st.columns([3, 2, 3])
with col_flip2:
    flip_label = ("🔄 Flip to Forward →"
                  if st.session_state.show_inverse
                  else "🔄 Flip to Inverse →")
    if st.button(flip_label, use_container_width=True):
        st.session_state.show_inverse = not st.session_state.show_inverse
        st.session_state.results = None
        st.rerun()

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# FORWARD INTERFACE
# ══════════════════════════════════════════════════════════════════════════════
if not st.session_state.show_inverse:

    col_ctrl, col_out = st.columns([1, 3])

    with col_ctrl:
        st.markdown("#### ⚙️ Simulation Parameters")
        P_mpa  = st.slider("External Pressure P (MPa)",
                           0.0, 45.0, 20.0, 1.0)
        C_rate = st.slider("C-rate",
                           0.02, 0.5, 0.1, 0.01,
                           format="%.2f")
        tau    = st.slider("State of Charge τ",
                           0.0, 1.0, 0.5, 0.01,
                           format="%.2f")

        st.markdown("---")
        st.markdown("**MC Dropout samples**")
        n_mc = st.slider("N samples (UQ)", 20, 200, 50, 10)

        st.markdown("---")
        run = st.button("▶ Run Simulation",
                        use_container_width=True,
                        type="primary")

        if run:
            with st.spinner("Computing physics-based surrogate..."):
                time.sleep(1.5)
                st.session_state.results = run_forward(
                    P_mpa * 1e6, C_rate, tau, n_mc=n_mc)
            st.success("Done!")

    with col_out:
        if st.session_state.results is None:
            st.info("👈 Set parameters and press **Run Simulation**")
        else:
            res = st.session_state.results

            # ── Scalar metrics ─────────────────────────────────────────────
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-value">{res["V_mean"]:.3f} V</div>'
                    f'<div class="metric-label">Terminal Voltage</div>'
                    f'<div class="metric-unc">±{res["V_std"]:.3f} V (MC)</div>'
                    f'</div>', unsafe_allow_html=True)
            with m2:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-value">{res["cap_mean"]:.1f}</div>'
                    f'<div class="metric-label">Capacity (mAh/g)</div>'
                    f'<div class="metric-unc">±{res["cap_std"]:.1f} (MC)</div>'
                    f'</div>', unsafe_allow_html=True)
            with m3:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-value">{res["xi_mean"].max():.3f}</div>'
                    f'<div class="metric-label">ξ_max (damage)</div>'
                    f'<div class="metric-unc">±{res["xi_std"].max():.3f} (MC)</div>'
                    f'</div>', unsafe_allow_html=True)
            with m4:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-value">{res["vm_mean"].max():.1f}</div>'
                    f'<div class="metric-label">von Mises max (MPa)</div>'
                    f'<div class="metric-unc">±{res["vm_std"].max():.1f} (MC)</div>'
                    f'</div>', unsafe_allow_html=True)

            st.markdown("")

            # ── Heatmaps row ───────────────────────────────────────────────
            h1, h2, h3 = st.columns(3)
            with h1:
                fig_xi = make_heatmap(
                    res["xi_mean"], x_mesh_np,
                    "Phase-field ξ (damage)", "ξ (-)",
                    colorscale="Jet", vmin=0, vmax=1)
                st.plotly_chart(fig_xi, use_container_width=True)
            with h2:
                fig_vm = make_heatmap(
                    res["vm_mean"], x_mesh_np,
                    "von Mises stress", "MPa",
                    colorscale="Jet")
                st.plotly_chart(fig_vm, use_container_width=True)
            with h3:
                fig_sxx = make_heatmap(
                    res["sxx_mean"], x_mesh_np,
                    "σ_xx stress", "MPa",
                    colorscale="RdBu_r")
                st.plotly_chart(fig_sxx, use_container_width=True)

            # ── V-cap curve ────────────────────────────────────────────────
            fig_vc = make_vcap_curve(
                tau, res["V_mean"], res["V_std"],
                res["cap_mean"], res["cap_std"])
            st.plotly_chart(fig_vc, use_container_width=True)

            # ── MC uncertainty note ────────────────────────────────────────
            st.caption(
                f"MC Dropout UQ: N={n_mc} forward passes · "
                f"ξ mean uncertainty = {res['xi_std'].mean():.4f} · "
                f"[CHOICE ROM phase-field, not full history variable]"
            )

# ══════════════════════════════════════════════════════════════════════════════
# INVERSE INTERFACE
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.markdown("### 🔍 Inverse Solver — Find Optimal Operating Conditions")
    st.markdown(
        "Enter your constraints below. "
        "The surrogate searches 2700 pre-computed points across "
        "(P, C-rate, τ) space and returns the Pareto front.")

    col_inv, col_res = st.columns([1, 2])

    with col_inv:
        st.markdown("#### 🎯 Constraints")
        xi_limit  = st.slider("Max damage ξ_max ≤",
                              0.05, 0.95, 0.70, 0.05,
                              format="%.2f")
        cap_limit = st.slider("Min capacity cap ≥ (mAh/g)",
                              10.0, 240.0, 50.0, 5.0)

        st.markdown("---")
        run_inv = st.button("🔍 Find Optimal Conditions",
                            use_container_width=True,
                            type="primary")

        st.markdown("---")
        st.markdown("**Search space**")
        st.caption(f"P: 0–45 MPa (20 pts)")
        st.caption(f"C-rate: 0.02–0.5 (15 pts)")
        st.caption(f"τ: 0.1–0.9 (9 pts)")
        st.caption(f"Total: 2700 pre-computed points")

    with col_res:
        if run_inv:
            results_lut = inv_b["lookup_table"]
            with st.spinner("Searching optimal conditions..."):
                time.sleep(1.0)
                fig_pareto, feasible = make_pareto(
                    results_lut, xi_limit, cap_limit)

            st.plotly_chart(fig_pareto, use_container_width=True)

            if feasible:
                st.success(f"✅ Found **{len(feasible)}** feasible points")

                # Best point: minimum ξ among feasible
                best = min(feasible, key=lambda x: x[0])
                xi_b, cap_b, P_b, Cr_b = best

                b1, b2, b3, b4 = st.columns(4)
                with b1:
                    st.markdown(
                        f'<div class="metric-card">'
                        f'<div class="metric-value">{P_b:.0f} MPa</div>'
                        f'<div class="metric-label">Optimal Pressure</div>'
                        f'</div>', unsafe_allow_html=True)
                with b2:
                    st.markdown(
                        f'<div class="metric-card">'
                        f'<div class="metric-value">{Cr_b:.3f}</div>'
                        f'<div class="metric-label">Optimal C-rate</div>'
                        f'</div>', unsafe_allow_html=True)
                with b3:
                    st.markdown(
                        f'<div class="metric-card">'
                        f'<div class="metric-value">{xi_b:.3f}</div>'
                        f'<div class="metric-label">ξ_max achieved</div>'
                        f'</div>', unsafe_allow_html=True)
                with b4:
                    st.markdown(
                        f'<div class="metric-card">'
                        f'<div class="metric-value">{cap_b:.1f}</div>'
                        f'<div class="metric-label">cap (mAh/g)</div>'
                        f'</div>', unsafe_allow_html=True)

                st.markdown("")

                # Sensitivity: P effect on ξ at best C-rate
                tau_fixed = 0.5
                subset = [r for r in results_lut
                          if abs(r["C_rate"] - Cr_b) < 0.02
                          and abs(r["tau"] - tau_fixed) < 0.06]
                if subset:
                    subset_s = sorted(subset, key=lambda r: r["P"])
                    P_vals_s  = [r["P"] / 1e6 for r in subset_s]
                    xi_vals_s = [r["xi_max"]   for r in subset_s]
                    cap_vals_s= [r["cap"]       for r in subset_s]

                    fig_sens = make_subplots(
                        rows=1, cols=2,
                        subplot_titles=[
                            f"ξ_max vs P (C-rate≈{Cr_b:.3f}, τ=0.5)",
                            f"cap vs P (C-rate≈{Cr_b:.3f}, τ=0.5)"
                        ])
                    fig_sens.add_trace(go.Scatter(
                        x=P_vals_s, y=xi_vals_s,
                        mode="lines+markers",
                        line=dict(color="#e74c3c", width=2),
                        name="ξ_max"), row=1, col=1)
                    fig_sens.add_trace(go.Scatter(
                        x=P_vals_s, y=cap_vals_s,
                        mode="lines+markers",
                        line=dict(color="#4a90d9", width=2),
                        name="cap"), row=1, col=2)
                    fig_sens.update_layout(
                        height=250,
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(14,17,23,0.8)",
                        showlegend=False,
                        margin=dict(l=40, r=10, t=40, b=36))
                    st.plotly_chart(fig_sens, use_container_width=True)

                st.caption(
                    "Inverse search uses pre-computed lookup table "
                    "(2700 pts). "
                    "Note: P has weak effect on V_cell/cap in this model "
                    "[documented CHOICE — BV stress coupling omitted]."
                )
            else:
                st.warning(
                    "⚠️ No feasible points found for these constraints. "
                    "Try relaxing ξ_max (increase) or cap_min (decrease)."
                )
        else:
            st.info("👈 Set constraints and press **Find Optimal Conditions**")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div style="text-align:center;font-size:11px;color:#555;">'
    'Physics-Informed Reduced-Order Digital Twin for ASSB Cathode Degradation · '
    'Based on Taghikhani & Kee, J. Mech. Phys. Solids 198 (2025) 106060 · '
    'FNO (fields) + DeepONet (scalars) + MC Dropout UQ · '
    'JAX + Flax · Streamlit Cloud'
    '</div>',
    unsafe_allow_html=True)
