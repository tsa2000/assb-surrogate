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
import io

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
        font-size: 26px; font-weight: 700;
        color: #4a90d9; text-align: center; margin-bottom: 2px;
    }
    .sub-title {
        font-size: 12px; color: #777;
        text-align: center; margin-bottom: 12px;
    }
    .metric-card {
        background: #16213e; border: 1px solid #4a90d9;
        border-radius: 10px; padding: 12px; text-align: center;
    }
    .metric-value { font-size: 22px; font-weight: 700; color: #4a90d9; }
    .metric-label { font-size: 11px; color: #aaa; margin-top: 2px; }
    .metric-unc   { font-size: 10px; color: #f39c12; margin-top: 2px; }
    .danger-red   { color: #e74c3c; font-size: 20px; font-weight: 700; }
    .danger-yellow{ color: #f39c12; font-size: 20px; font-weight: 700; }
    .danger-green { color: #27ae60; font-size: 20px; font-weight: 700; }
    .badge-choice {
        background: #2d1b00; border: 1px solid #f39c12;
        border-radius: 5px; padding: 3px 8px;
        font-size: 10px; color: #f39c12; display: inline-block; margin: 2px;
    }
    .r2-badge {
        background: #0a2818; border: 1px solid #27ae60;
        border-radius: 5px; padding: 3px 8px;
        font-size: 10px; color: #27ae60; display: inline-block; margin: 2px;
    }
    .section-title {
        font-size: 16px; font-weight: 600;
        color: #4a90d9; margin: 16px 0 8px 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Particle geometry ──────────────────────────────────────────────────────────
L_ref = 40e-6
PARTICLES = [
    {"cx": 0.28, "cy": 0.28, "R": 6.0e-6/L_ref},
    {"cx": 0.72, "cy": 0.20, "R": 8.0e-6/L_ref},
    {"cx": 0.80, "cy": 0.70, "R": 6.8e-6/L_ref},
    {"cx": 0.28, "cy": 0.75, "R": 7.2e-6/L_ref},
    {"cx": 0.55, "cy": 0.48, "R": 4.0e-6/L_ref},
]

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
        out_ft = jnp.zeros((x.shape[0], x.shape[1]//2+1,
                            self.out_channels), dtype=jnp.complex64)
        out_ft = out_ft.at[:, :self.n_modes, :].set(out_re + 1j*out_im)
        return jnp.fft.irfft(out_ft, n=x.shape[1], axis=1)

class FNOBlock(nn.Module):
    channels: int; n_modes: int; dropout_rate: float = 0.15
    @nn.compact
    def __call__(self, x, training=False):
        out = nn.gelu(SpectralConv1d(self.channels, self.n_modes)(x) +
                      nn.Dense(self.channels)(x))
        return nn.Dropout(rate=self.dropout_rate,
                          deterministic=not training)(out)

class FNO(nn.Module):
    n_modes: int = 24; channels: int = 64; n_layers: int = 6
    n_field_out: int = 3; dropout_rate: float = 0.15
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
    hidden: int = 128; n_layers: int = 4
    n_out: int = 2; dropout_rate: float = 0.15
    @nn.compact
    def __call__(self, x_in, training=False):
        x = x_in
        for _ in range(self.n_layers):
            x = nn.gelu(nn.Dense(self.hidden)(x))
            x = nn.Dropout(rate=self.dropout_rate,
                           deterministic=not training)(x)
        return nn.sigmoid(nn.Dense(self.n_out)(x))

# ── Load models ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_all():
    with open("models/fno_bundle_final.pkl", "rb") as f:
        fno_b = pickle.load(f)
    with open("models/don_bundle_final.pkl", "rb") as f:
        don_b = pickle.load(f)
    fno_model = FNO()
    don_model  = DeepONet()
    return fno_b, don_b, fno_model, don_model

fno_b, don_b, fno_model, don_model = load_all()

in_min    = np.array(fno_b["normalization"]["in_min"])
in_max    = np.array(fno_b["normalization"]["in_max"])
field_min = np.array(fno_b["normalization"]["field_min"])
field_max = np.array(fno_b["normalization"]["field_max"])
scal_min  = np.array(don_b["normalization"]["scal_min"])
scal_max  = np.array(don_b["normalization"]["scal_max"])
x_mesh_np = fno_b["mesh"]["x_mesh_np"]
n_elem    = fno_b["mesh"]["n_elem"]
x_mesh_base = jnp.array(x_mesh_np)[None, :, :]
bx = x_mesh_np[:, 0]
by = x_mesh_np[:, 1]

# ── Inference ─────────────────────────────────────────────────────────────────
def run_forward(P, C_rate, tau, n_mc=50):
    x_in_raw = np.array([[P, C_rate, tau]], dtype=np.float32)
    x_in_n   = (x_in_raw - in_min) / (in_max - in_min)
    x_in_j   = jnp.array(x_in_n)
    x_mesh_j = jnp.broadcast_to(x_mesh_base, (1, n_elem, 2))

    key = jax.random.PRNGKey(42)
    mc_fields = []; mc_scals = []

    for _ in range(n_mc):
        key, sk1, sk2 = jax.random.split(key, 3)
        pf = fno_model.apply(fno_b["params"], x_in_j, x_mesh_j,
                             training=True, rngs={"dropout": sk1})
        ps = don_model.apply(don_b["params"], x_in_j,
                             training=True, rngs={"dropout": sk2})
        mc_fields.append(np.array(pf[0]))
        mc_scals.append(np.array(ps[0]))

    mc_f = np.stack(mc_fields); mc_s = np.stack(mc_scals)
    mean_f = mc_f.mean(0); std_f = mc_f.std(0)
    mean_s = mc_s.mean(0); std_s  = mc_s.std(0)

    def dn_f(arr, i):
        return arr * (field_max[i] - field_min[i]) + field_min[i]
    def dn_s(v, i):
        return float(v) * (scal_max[i] - scal_min[i]) + scal_min[i]

    return {
        "vm_mean":  dn_f(mean_f[:,0],0)/1e6,
        "vm_std":   dn_f(std_f[:,0], 0)/1e6,
        "xi_mean":  dn_f(mean_f[:,1],1),
        "xi_std":   dn_f(std_f[:,1], 1),
        "sxx_mean": dn_f(mean_f[:,2],2)/1e6,
        "sxx_std":  dn_f(std_f[:,2], 2)/1e6,
        "V_mean":   dn_s(mean_s[0],0),
        "V_std":    float(std_s[0])*(scal_max[0]-scal_min[0]),
        "cap_mean": dn_s(mean_s[1],1),
        "cap_std":  float(std_s[1])*(scal_max[1]-scal_min[1]),
    }

# ── Heatmap with smooth interpolation ─────────────────────────────────────────
def make_heatmap(values, title, unit, colorscale="Jet", vmin=None, vmax=None):
    xi_g = np.linspace(0, 1, 200)
    yi_g = np.linspace(0, 1, 200)
    XX, YY = np.meshgrid(xi_g, yi_g)
    try:
        ZZ = griddata((bx, by), values, (XX, YY), method="linear")
    except Exception:
        ZZ = np.full(XX.shape, np.nan)

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        x=xi_g, y=yi_g, z=ZZ,
        colorscale=colorscale,
        zmin=vmin, zmax=vmax,
        colorbar=dict(title=dict(text=unit, font=dict(size=10)), thickness=10),
        showscale=True,
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=12), x=0.5),
        xaxis=dict(showticklabels=False, showgrid=False),
        yaxis=dict(showticklabels=False, showgrid=False, scaleanchor="x"),
        margin=dict(l=0, r=0, t=32, b=0),
        height=250,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig

# ── V-cap curve ────────────────────────────────────────────────────────────────
def make_vcap(tau_val, V_mean, V_std, cap_mean, cap_std):
    n_pts   = max(6, int(tau_val * 12))
    cap_arr = np.linspace(0, cap_mean, n_pts)
    V_arr   = np.linspace(2.99, V_mean, n_pts)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cap_arr, y=V_arr + V_std, fill=None,
        mode="lines", line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(
        x=cap_arr, y=V_arr - V_std, fill="tonexty",
        mode="lines", line=dict(width=0, color="#f39c12"),
        fillcolor="rgba(243,156,18,0.15)", name="±1σ MC"))
    fig.add_trace(go.Scatter(
        x=cap_arr, y=V_arr, mode="lines+markers",
        line=dict(color="#4a90d9", width=2),
        marker=dict(size=5), name="Surrogate"))
    fig.add_hline(y=4.2, line_dash="dash", line_color="#e74c3c",
                  line_width=1, annotation_text="4.2V cutoff")
    fig.update_layout(
        title=dict(text="V-cap Curve", font=dict(size=12), x=0.5),
        xaxis_title="Capacity (mAh/g)", yaxis_title="V_cell (V)",
        yaxis=dict(range=[2.85, 4.35]),
        legend=dict(font=dict(size=9)),
        margin=dict(l=40, r=10, t=32, b=36), height=250,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(14,17,23,0.8)",
    )
    return fig

# ── Multi-tau analysis ────────────────────────────────────────────────────────
def make_tau_analysis(P, C_rate, n_mc=20):
    tau_vals = np.linspace(0.05, 1.0, 12)
    xi_maxs=[]; vm_maxs=[]; caps=[]; Vs=[]; crack_areas=[]
    for t in tau_vals:
        r = run_forward(P, C_rate, t, n_mc=n_mc)
        xi_maxs.append(r["xi_mean"].max())
        vm_maxs.append(r["vm_mean"].max())
        caps.append(r["cap_mean"])
        Vs.append(r["V_mean"])
        crack_areas.append((r["xi_mean"] > 0.5).sum() / n_elem * 100)
    return tau_vals, xi_maxs, vm_maxs, caps, Vs, crack_areas

# ── Multi C-rate comparison ───────────────────────────────────────────────────
def make_crate_comparison(P, n_mc=20):
    tau_vals  = np.linspace(0.1, 1.0, 8)
    crates    = [0.02, 0.1, 0.5]
    colors    = ["#27ae60", "#4a90d9", "#e74c3c"]
    labels    = ["0.02C (slow)", "0.1C (nominal)", "0.5C (fast)"]
    fig = make_subplots(1, 2, subplot_titles=["ξ_max vs τ", "cap vs τ"])
    for cr, col, lab in zip(crates, colors, labels):
        xi_l=[]; cap_l=[]
        for t in tau_vals:
            r = run_forward(P, cr, t, n_mc=n_mc)
            xi_l.append(r["xi_mean"].max())
            cap_l.append(r["cap_mean"])
        fig.add_trace(go.Scatter(x=tau_vals, y=xi_l, mode="lines+markers",
            line=dict(color=col, width=2), name=lab), row=1, col=1)
        fig.add_trace(go.Scatter(x=tau_vals, y=cap_l, mode="lines+markers",
            line=dict(color=col, width=2), name=lab,
            showlegend=False), row=1, col=2)
    fig.update_layout(height=280, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(14,17,23,0.8)",
        margin=dict(l=40,r=10,t=40,b=36),
        legend=dict(font=dict(size=9)))
    fig.update_xaxes(title_text="τ (SOC)")
    fig.update_yaxes(title_text="ξ_max", row=1, col=1)
    fig.update_yaxes(title_text="cap (mAh/g)", row=1, col=2)
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="main-title">⚡ ASSB Digital Twin Surrogate</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Physics-Informed FNO + DeepONet · '
    'Taghikhani & Kee, J. Mech. Phys. Solids 198 (2025) 106060 · '
    'JAX + Flax · Streamlit Cloud</div>', unsafe_allow_html=True)

st.markdown(
    '<div style="text-align:center;margin-bottom:8px">'
    '<span class="r2-badge">FNO: vm R²=0.944</span>'
    '<span class="r2-badge">FNO: ξ R²=0.965</span>'
    '<span class="r2-badge">FNO: σ_xx R²=0.935</span>'
    '<span class="r2-badge">DON: V_cell R²=0.9999</span>'
    '<span class="r2-badge">DON: cap R²=0.9998</span>'
    '</div>', unsafe_allow_html=True)

st.markdown(
    '<div style="text-align:center;margin-bottom:12px">'
    '<span class="badge-choice">[CHOICE: 5 circular particles, NOT paper SEM]</span>'
    '<span class="badge-choice">[CHOICE: galvanostatic, NOT full Butler-Volmer]</span>'
    '<span class="badge-choice">[CHOICE: ROM phase-field]</span>'
    '</div>', unsafe_allow_html=True)

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# CONTROLS + RESULTS
# ══════════════════════════════════════════════════════════════════════════════
col_ctrl, col_out = st.columns([1, 3])

with col_ctrl:
    st.markdown("#### ⚙️ Parameters")
    P_mpa  = st.slider("Pressure P (MPa)", 0.0, 45.0, 20.0, 1.0)
    C_rate = st.slider("C-rate", 0.02, 0.5, 0.1, 0.01, format="%.2f")
    tau    = st.slider("State of Charge τ", 0.0, 1.0, 0.5, 0.01)
    st.markdown("---")
    n_mc   = st.slider("MC Dropout samples", 20, 200, 50, 10)
    st.markdown("---")
    run_btn = st.button("▶ Run Simulation", use_container_width=True,
                        type="primary")

    if "results" not in st.session_state:
        st.session_state.results = None

    if run_btn:
        prog = st.progress(0, text="Initializing...")
        time.sleep(0.3); prog.progress(20, text="Running FNO inference...")
        time.sleep(0.3); prog.progress(50, text="MC Dropout sampling...")
        st.session_state.results = run_forward(P_mpa*1e6, C_rate, tau, n_mc)
        prog.progress(90, text="Generating plots...")
        time.sleep(0.2); prog.progress(100, text="Done!")
        time.sleep(0.3); prog.empty()
        st.success("Done!")

with col_out:
    if st.session_state.results is None:
        st.info("👈 Set parameters and press **Run Simulation**")
    else:
        res = st.session_state.results

        # ── Traffic light ──────────────────────────────────────────────────
        xi_max_val = float(res["xi_mean"].max())
        if xi_max_val < 0.3:
            danger_html = '<span class="danger-green">🟢 Safe</span>'
            danger_msg  = "Damage level acceptable"
        elif xi_max_val < 0.6:
            danger_html = '<span class="danger-yellow">🟡 Warning</span>'
            danger_msg  = "Moderate damage — monitor"
        else:
            danger_html = '<span class="danger-red">🔴 Danger</span>'
            danger_msg  = "High damage — reduce C-rate or increase P"

        st.markdown(
            f'<div style="text-align:center;padding:8px;background:#16213e;'
            f'border-radius:8px;margin-bottom:12px">'
            f'{danger_html} &nbsp; <span style="color:#aaa;font-size:13px">'
            f'{danger_msg}</span></div>', unsafe_allow_html=True)

        # ── Scalar metrics ─────────────────────────────────────────────────
        m1, m2, m3, m4 = st.columns(4)
        for col_m, val, std, label in zip(
            [m1,m2,m3,m4],
            [res["V_mean"], res["cap_mean"], xi_max_val, res["vm_mean"].max()],
            [res["V_std"],  res["cap_std"],  res["xi_std"].max(), res["vm_std"].max()],
            ["Terminal Voltage (V)", "Capacity (mAh/g)", "ξ_max (damage)", "vm_max (MPa)"]
        ):
            with col_m:
                st.markdown(
                    f'<div class="metric-card">'
                    f'<div class="metric-value">{val:.3f}</div>'
                    f'<div class="metric-label">{label}</div>'
                    f'<div class="metric-unc">±{std:.3f} (MC)</div>'
                    f'</div>', unsafe_allow_html=True)

        st.markdown("")

        # ── Heatmaps ───────────────────────────────────────────────────────
        st.markdown('<div class="section-title">📊 Field Distributions</div>',
                    unsafe_allow_html=True)
        h1, h2, h3 = st.columns(3)
        with h1:
            st.plotly_chart(make_heatmap(res["xi_mean"],
                "Phase-field ξ (damage)", "ξ (-)",
                colorscale="Jet", vmin=0, vmax=1),
                use_container_width=True)
        with h2:
            st.plotly_chart(make_heatmap(res["vm_mean"],
                "von Mises stress", "MPa", colorscale="Jet"),
                use_container_width=True)
        with h3:
            st.plotly_chart(make_heatmap(res["sxx_mean"],
                "σ_xx stress", "MPa", colorscale="RdBu_r"),
                use_container_width=True)

        # ── V-cap + xi histogram ───────────────────────────────────────────
        st.markdown('<div class="section-title">📈 Electrochemical Response</div>',
                    unsafe_allow_html=True)
        vc1, vc2 = st.columns(2)
        with vc1:
            st.plotly_chart(make_vcap(tau, res["V_mean"], res["V_std"],
                res["cap_mean"], res["cap_std"]), use_container_width=True)
        with vc2:
            # ξ histogram
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=res["xi_mean"], nbinsx=40,
                marker_color="#4a90d9", opacity=0.8, name="ξ distribution"))
            fig_hist.add_vline(x=0.3, line_dash="dash",
                               line_color="#27ae60", annotation_text="Safe limit")
            fig_hist.add_vline(x=0.6, line_dash="dash",
                               line_color="#e74c3c", annotation_text="Danger limit")
            fig_hist.update_layout(
                title=dict(text="ξ Distribution Histogram", font=dict(size=12), x=0.5),
                xaxis_title="ξ (-)", yaxis_title="Element count",
                margin=dict(l=40,r=10,t=32,b=36), height=250,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(14,17,23,0.8)")
            st.plotly_chart(fig_hist, use_container_width=True)

        # ── Numerical results table ────────────────────────────────────────
        st.markdown('<div class="section-title">📋 Numerical Summary</div>',
                    unsafe_allow_html=True)
        crack_pct = (res["xi_mean"] > 0.5).sum() / n_elem * 100
        tbl_data = {
            "Field": ["ξ (damage)", "von Mises (MPa)", "σ_xx (MPa)"],
            "Mean":  [f"{res['xi_mean'].mean():.4f}",
                      f"{res['vm_mean'].mean():.1f}",
                      f"{res['sxx_mean'].mean():.1f}"],
            "Std (MC)": [f"±{res['xi_std'].mean():.4f}",
                         f"±{res['vm_std'].mean():.1f}",
                         f"±{res['sxx_std'].mean():.1f}"],
            "Max":   [f"{res['xi_mean'].max():.4f}",
                      f"{res['vm_mean'].max():.1f}",
                      f"{res['sxx_mean'].max():.1f}"],
        }
        st.table(tbl_data)
        st.caption(f"Crack area (ξ > 0.5): {crack_pct:.1f}% of domain | "
                   f"MC Dropout N={n_mc} samples")

        # ── Download CSV ───────────────────────────────────────────────────
        csv_data = (
            f"P_MPa,C_rate,tau,V_cell_V,cap_mAhg,xi_max,vm_max_MPa,crack_pct\n"
            f"{P_mpa},{C_rate},{tau},{res['V_mean']:.4f},"
            f"{res['cap_mean']:.2f},{xi_max_val:.4f},"
            f"{res['vm_mean'].max():.2f},{crack_pct:.2f}\n"
        )
        st.download_button("⬇ Download Results (CSV)",
                           csv_data, file_name="assb_results.csv",
                           mime="text/csv")

# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS SECTION
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("### 📊 Parameter Analysis")

tab1, tab2, tab3 = st.tabs(["τ Evolution", "C-rate Comparison", "Pressure Effect"])

with tab1:
    st.markdown("Damage and capacity evolution with state of charge τ")
    if st.button("▶ Run τ Analysis", key="tau_btn"):
        with st.spinner("Computing 12 points across τ..."):
            tau_v, xi_v, vm_v, cap_v, V_v, crack_v = make_tau_analysis(
                P_mpa*1e6, C_rate, n_mc=20)
        fig_tau = make_subplots(2, 2, subplot_titles=[
            "ξ_max vs τ", "vm_max vs τ",
            "Capacity vs τ", "Crack area % vs τ"])
        for row,col,ydata,ylab,col_c in [
            (1,1,xi_v,   "ξ_max","#e74c3c"),
            (1,2,vm_v,   "vm_max (MPa)","#f39c12"),
            (2,1,cap_v,  "cap (mAh/g)","#4a90d9"),
            (2,2,crack_v,"Crack area (%)","#9b59b6"),
        ]:
            fig_tau.add_trace(go.Scatter(
                x=tau_v, y=ydata, mode="lines+markers",
                line=dict(color=col_c, width=2),
                marker=dict(size=6), name=ylab,
                showlegend=False), row=row, col=col)
        fig_tau.update_layout(height=420,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(14,17,23,0.8)",
            margin=dict(l=40,r=10,t=40,b=36))
        fig_tau.update_xaxes(title_text="τ (SOC)")
        st.plotly_chart(fig_tau, use_container_width=True)

with tab2:
    st.markdown("Compare slow vs nominal vs fast charging")
    if st.button("▶ Run C-rate Comparison", key="cr_btn"):
        with st.spinner("Computing 3 C-rates × 8 τ points..."):
            fig_cr = make_crate_comparison(P_mpa*1e6, n_mc=20)
        st.plotly_chart(fig_cr, use_container_width=True)

with tab3:
    st.markdown("Effect of stack pressure on damage at current τ")
    if st.button("▶ Run Pressure Analysis", key="p_btn"):
        P_vals = np.linspace(0, 45e6, 10)
        xi_p=[]; vm_p=[]
        prog_p = st.progress(0)
        for i, P_v in enumerate(P_vals):
            r = run_forward(P_v, C_rate, tau, n_mc=20)
            xi_p.append(r["xi_mean"].max())
            vm_p.append(r["vm_mean"].max())
            prog_p.progress((i+1)/len(P_vals))
        prog_p.empty()
        fig_p = make_subplots(1, 2, subplot_titles=[
            "ξ_max vs Pressure", "vm_max vs Pressure"])
        fig_p.add_trace(go.Scatter(x=P_vals/1e6, y=xi_p,
            mode="lines+markers", line=dict(color="#e74c3c",width=2),
            name="ξ_max"), row=1, col=1)
        fig_p.add_trace(go.Scatter(x=P_vals/1e6, y=vm_p,
            mode="lines+markers", line=dict(color="#f39c12",width=2),
            name="vm_max"), row=1, col=2)
        fig_p.update_layout(height=280,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(14,17,23,0.8)",
            margin=dict(l=40,r=10,t=40,b=36))
        fig_p.update_xaxes(title_text="P (MPa)")
        st.plotly_chart(fig_p, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# MODEL INFO + LIMITATIONS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")

col_info, col_lim = st.columns(2)

with col_info:
    with st.expander("ℹ️ Model Information", expanded=False):
        st.markdown("**Architecture:**")
        st.table({
            "Component": ["FNO (fields)", "DeepONet (scalars)"],
            "Output": ["vm, ξ, σ_xx on 6200 elements",
                       "V_cell, capacity"],
            "R² (test)": ["0.944 / 0.965 / 0.935", "0.9999 / 0.9998"],
            "Parameters": ["1,209,347", "50,306"],
        })
        st.markdown("**Training:**")
        st.markdown(
            "- FNO: 1000 epochs, AdamW + CosineDecay, lr=3e-4\n"
            "- DeepONet: 2000 epochs, lr=1e-3\n"
            "- Data: 600 FEM points (10P × 6Cr × 10τ)\n"
            "- UQ: MC Dropout p=0.15")
        st.markdown("**Materials (Table 1 — Taghikhani & Kee 2025):**")
        st.table({
            "Material": ["NMC (particle)", "LPSC (matrix)", "Interface"],
            "E (GPa)": [175.3, 22.1, "—"],
            "ν": [0.282, 0.37, "—"],
            "Gc (J/m²)": [2.5, 2.785, 1.0],
        })

with col_lim:
    with st.expander("⚠️ Documented Limitations", expanded=False):
        st.markdown("""
**[CHOICE] Geometry:**
5 circular NMC particles in 40×40µm domain —
NOT the SEM microstructure from Bielefeld et al. 2022.

**[CHOICE] Electrochemistry:**
Galvanostatic (current-controlled) formulation —
NOT full Butler-Volmer. BV stress coupling term
(β_ij σ_ij / F ≈ 0.065V at 40MPa) is omitted.

**[CHOICE] Phase-field:**
ROM irreversibility H = ψ₀⁺ · tanh(8τ) —
NOT the full history variable from AT2.

**[CHOICE] Scale:**
~25K DOF vs paper's 3.5M DOF COMSOL model.

**Consequence:**
P has weak effect on V_cell/cap in this model.
Fields are physics-based interpolations, not
exact reproductions of paper figures.
        """)
        st.markdown("""
**Reference:**
Taghikhani & Kee, *J. Mech. Phys. Solids* 198 (2025) 106060
        """)

with st.expander("⚛️ Physics Equations", expanded=False):
    st.markdown(r"""
**Mechanical equilibrium:**
$$\nabla \cdot \boldsymbol{\sigma} = 0, \quad \boldsymbol{\sigma} = \mathbb{C} : (\boldsymbol{\varepsilon} - \boldsymbol{\varepsilon}^{eig})$$

**Chemical eigenstrain:**
$$\varepsilon^{eig} = \beta_{eff} \Delta x \cdot \mathbf{I}, \quad \beta_{eff} = \beta_{iso} \cdot c_{Li,max} = 0.008124$$

**AT2 Phase-field (ROM):**
$$\left[\frac{3G_c}{4\ell_0} + 2H\right]\xi - \frac{3G_c \ell_0}{8}\nabla^2\xi = 2H$$

**Irreversibility (ROM):**
$$H = \psi_0^+ \cdot \tanh(8\tau)$$

**Tensile energy:**
$$\psi_0^+ = \frac{1}{2}K\langle I_1 \rangle_+^2 + 2\mu J_2$$
    """)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div style="text-align:center;font-size:10px;color:#444;">'
    'Physics-Informed Reduced-Order Digital Twin for ASSB Cathode Degradation · '
    'Taghikhani & Kee 2025 · FNO + DeepONet + MC Dropout UQ · '
    'JAX + Flax · Streamlit Cloud · '
    '<a href="https://github.com/tsa2000/assb-surrogate" '
    'style="color:#4a90d9">GitHub</a>'
    '</div>', unsafe_allow_html=True)
