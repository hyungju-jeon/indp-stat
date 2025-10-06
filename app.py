# app.py
# Streamlit App: Interactive + Composite Distributions
# -------------------------------------------------------------
# To run:
#   pip install streamlit scipy numpy plotly
#   streamlit run app.py
# -------------------------------------------------------------

import math
from typing import Dict, Any

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
import streamlit as st

st.set_page_config(page_title="INDP 2025 - Distributions", layout="wide")

# Soft color palette inspired by seaborn
SOFT_COLORS = {
    'primary': '#4C72B0',      # Soft blue
    'secondary': '#55A868',     # Soft green  
    'accent': '#C44E52',        # Soft red
    'purple': '#8172B3',        # Soft purple
    'orange': '#CCB974',        # Soft yellow/orange
    'pink': '#DD8452',          # Soft coral
    'teal': '#64B5CD',          # Soft teal
    'background': '#F8F9FA',    # Light gray background
    'text_primary': '#2F3E46',  # Dark gray for primary text
    'text_secondary': '#52796F', # Medium gray for secondary text
    'text_muted': '#84A98C'     # Light gray for muted text
}

# Custom CSS for better typography and colors
st.markdown("""
<style>
    .main-header {
        color: #2F3E46;
        font-size: 2.5rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================
# 1. Base Distribution Metadata
# =============================================================

def _clip_int(a, lo, hi):
    return int(max(lo, min(hi, a)))

# Fixed plotting ranges for each distribution (to prevent axis jumping)
FIXED_RANGES = {
    "Normal (Gaussian)": (-6, 6),
    "Exponential": (0, 8),
    "Gamma (shape–scale)": (0, 15),
    "Beta": (0, 1),
    "Uniform [a,b]": (-12, 12),
    "Bernoulli": (0, 1),
    "Binomial": (0, 50),  # Will be adjusted based on n parameter
    "Poisson": (0, 25),
    "Geometric (trials until 1st success)": (1, 20),
    "Chi-squared": (0, 30),
    "Log-normal": (0, 10),
    "Student's t": (-8, 8),
}

DISTRIBUTIONS: Dict[str, Dict[str, Any]] = {
    "Normal (Gaussian)": {
        "type": "continuous",
        "params": {
            "mu": {"default": 0.0, "min": -5.0, "max": 5.0, "step": 0.1, "help": "Mean (location)"},
            "sigma": {"default": 1.0, "min": 0.05, "max": 5.0, "step": 0.05, "help": "Standard deviation (scale) > 0"},
        },
        "scipy": ("norm", lambda p: dict(loc=p["mu"], scale=p["sigma"])),
        "equation": r"f(x)=\frac{1}{\sqrt{2\pi\,\sigma^2}}\,\exp\!\Big( -\tfrac{(x-\mu)^2}{2\sigma^2} \Big)",
        "numpy_code": "np.random.normal(loc=mu, scale=sigma, size=1000)",
        "description": """**Biology/Neuroscience Applications:**

• **Experimental Measurements:** Modeling measurement noise in laboratory experiments, population studies of heights/weights, IQ score distributions, and blood pressure readings across populations

• **Neuroscience Research:** Neural firing rate distributions, synaptic strength variability, and imaging noise characterization in fMRI/EEG studies

• **Cancer Biology:** Tumor growth rate modeling, drug concentration profiles in blood plasma, and biomarker level distributions in patient cohorts

• **Fundamental Principle:** The Central Limit Theorem makes this distribution ubiquitous for aggregated biological measurements and any process involving the sum of many small, independent effects""",
        "support_fn": lambda dist: dist.ppf([0.001, 0.999]),
        "technical_notes": """
- Most fundamental distribution in statistics due to Central Limit Theorem
- Mean = median = mode (perfectly symmetric)  
- **68-95-99.7 rule:** ~68% within 1σ, ~95% within 2σ, ~99.7% within 3σ
- Stable under linear transformations: `aX + b ~ N(aμ + b, a²σ²)`
- Sum of independent normals is normal
        """
    },
    "Exponential": {
        "type": "continuous",
        "params": {"lambda": {"default": 1.0, "min": 0.05, "max": 5.0, "step": 0.05, "help": "Rate λ > 0 (mean = 1/λ)"}},
        "scipy": ("expon", lambda p: dict(loc=0.0, scale=1.0 / p["lambda"])),
        "equation": r"f(x)=\lambda\,e^{-\lambda x},\; x\ge 0",
        "numpy_code": "np.random.exponential(scale=1/lam, size=1000)",
        "description": """**Biology/Neuroscience Applications:**

• **Neural Timing:** Inter-spike intervals in neural firing, time between action potentials in single-neuron recordings, and synaptic transmission delays

• **Cell Biology:** Time intervals between cell divisions, duration of cell cycle phases, and protein degradation half-lives

• **Cancer Research:** Time to metastasis events, patient survival times (when hazard is constant), and drug elimination kinetics

• **Key Property:** The memoryless characteristic makes it ideal for modeling biological processes where "aging" or history doesn't affect future event probability""",
        "support_fn": lambda dist: np.array([0.0, float(dist.ppf(0.999))]),
        "technical_notes": """
- Only continuous distribution with **memoryless property:** `P(X > s+t | X > s) = P(X > t)`
- Mean = `1/λ`, Variance = `1/λ²`, Mode = 0
- Survival function: `S(x) = e^(-λx)`
- Closely related to Poisson process (time between events)
- NumPy uses scale parameter `(1/λ)` while mathematical definition uses rate `(λ)`
        """
    },
    "Gamma (shape–scale)": {
        "type": "continuous",
        "params": {
            "k (shape)": {"default": 2.0, "min": 0.2, "max": 10.0, "step": 0.1},
            "theta (scale)": {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1},
        },
        "scipy": ("gamma", lambda p: dict(a=p["k (shape)"], scale=p["theta (scale)"])),
        "equation": r"f(x)=\frac{1}{\Gamma(k)\,\theta^k} x^{k-1} e^{-x/\theta},\;x\ge 0",
        "numpy_code": "np.random.gamma(shape=k, scale=theta, size=1000)",
        "description": """**Biology/Neuroscience Applications:**

• **Multi-Event Processes:** Waiting time for multiple sequential events - such as k mutations in cancer progression pathways or cumulative reaction times in biochemical cascades

• **Neural Networks:** Cumulative synaptic delays in neural pathways, burst duration modeling in neural oscillations, and integration time constants

• **Protein Dynamics:** Protein folding times (multi-step process), enzyme reaction cascades, and molecular transport through multiple barriers

• **Statistical Foundation:** Emerges naturally from sums of exponential processes, making it fundamental in Bayesian analysis of biological rate parameters""",
        "support_fn": lambda dist: np.array([0.0, float(dist.ppf(0.999))]),
        "technical_notes": """
- Sum of `k` independent Exponential(1/θ) random variables
- Mean = `kθ`, Variance = `kθ²`, Mode = `(k-1)θ` for k≥1
- When k=1, reduces to Exponential distribution
- When k=ν/2 and θ=2, becomes **Chi-squared** with ν degrees of freedom
- **Conjugate prior** for Poisson and Exponential distributions in Bayesian analysis
        """
    },
    "Beta": {
        "type": "continuous",
        "params": {
            "alpha": {"default": 2.0, "min": 0.1, "max": 10.0, "step": 0.1},
            "beta": {"default": 2.0, "min": 0.1, "max": 10.0, "step": 0.1},
        },
        "scipy": ("beta", lambda p: dict(a=p["alpha"], b=p["beta"])),
        "equation": r"f(x)=\frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\,\Gamma(\beta)} x^{\alpha-1} (1-x)^{\beta-1},\;0\le x\le 1",
        "numpy_code": "np.random.beta(a=alpha, b=beta, size=1000)",
        "description": """**Biology/Neuroscience Applications:**

• **Biological Proportions:** Gene expression levels (as fractions of maximum), tumor purity estimates in tissue samples, and proportion of active neurons in brain regions

• **Population Genetics:** Allele frequencies in populations, genetic diversity measures, and evolutionary fitness proportions

• **Clinical Research:** Bayesian priors for treatment success rates in clinical trials, cell viability percentages in drug screening, and biomarker sensitivity/specificity

• **Natural Bounds:** The [0,1] constraint makes it perfect for any biological measurement that represents a proportion, percentage, or probability""",
        "support_fn": lambda dist: np.array([0.0, 1.0]),
        "technical_notes": """
- Extremely flexible on [0,1] interval with various shapes
- Mean = `α/(α+β)`, Mode = `(α-1)/(α+β-2)` for α,β > 1
- When α=β=1, becomes **Uniform(0,1)**
- Shape behavior: α=1,β>1 (decreasing); α>1,β=1 (increasing); α,β>1 (unimodal)
- **Conjugate prior** for Binomial and Bernoulli distributions
        """
    },
    "Uniform [a,b]": {
        "type": "continuous",
        "params": {
            "a": {"default": -1.0, "min": -10.0, "max": 9.0, "step": 0.1},
            "b": {"default": 1.0, "min": -9.0, "max": 10.0, "step": 0.1},
        },
        "scipy": ("uniform", lambda p: dict(loc=min(p["a"], p["b"]), scale=abs(p["b"] - p["a"]))),
        "equation": r"f(x)=\frac{1}{b-a},\; a\le x\le b",
        "numpy_code": "np.random.uniform(low=a, high=b, size=1000)",
        "description": """**Biology/Neuroscience Applications:**

• **Non-informative Priors:** Baseline models in Bayesian analysis when limited prior knowledge exists about parameter ranges in biological systems

• **Controlled Studies:** Modeling random drug dosing within therapeutic windows, uniform spatial distribution of cells in tissue cultures, and random sampling protocols

• **Null Hypotheses:** Serving as neutral baseline models in comparative studies before incorporating specific biological knowledge or constraints

• **Quality Control:** Random sampling procedures in laboratory protocols and uniform background assumptions in imaging studies""",
        "support_fn": lambda dist: np.array([float(dist.ppf(0.001)), float(dist.ppf(0.999))]),
        "technical_notes": """
- Constant probability density over [a,b] interval
- Mean = `(a+b)/2`, Variance = `(b-a)²/12`
- No unique mode (all values equally likely)
- **Maximum entropy** distribution for given support [a,b]
- Often used as **non-informative prior** in Bayesian analysis
        """
    },
    "Bernoulli": {
        "type": "discrete",
        "params": {
            "p": {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "help": "Success probability"},
        },
        "scipy": ("bernoulli", lambda p: dict(p=p["p"])),
        "equation": r"P(X=x)=p^x (1-p)^{1-x},\; x\in\{0,1\}",
        "numpy_code": "np.random.binomial(n=1, p=p, size=1000)",
        "description": """**Biology/Neuroscience Applications:**

• **Binary Outcomes:** Cell survival assays (live/dead), treatment response classification (responder/non-responder), and gene expression states (on/off)

• **Neural Activity:** Action potential firing (spike/no spike) in single trials, synaptic transmission success/failure, and binary neural decoding tasks

• **Molecular Biology:** Mutation presence/absence, protein binding events, and binary molecular switches in cellular pathways

• **Foundation Model:** Forms the building block for more complex discrete models used throughout biological experimentation and clinical trials""",
        "support_fn": lambda dist: np.array([0, 1]),
        "technical_notes": """
- Simplest discrete distribution with two outcomes
- Mean = `p`, Variance = `p(1-p)`, Mode = 1 if p > 0.5, else 0
- **Special case** of Binomial with n=1
- Maximum variance occurs at p=0.5
- Foundation for more complex discrete distributions
        """
    },
    "Binomial": {
        "type": "discrete",
        "params": {
            "n": {"default": 10, "min": 1, "max": 200, "step": 1, "help": "Number of trials"},
            "p": {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "help": "Success probability"},
        },
        "scipy": ("binom", lambda p: dict(n=int(p["n"]), p=p["p"])) ,
        "equation": r"P(X=k)=\binom{n}{k} p^k (1-p)^{n-k},\;k=0,\dots,n",
        "numpy_code": "np.random.binomial(n=n, p=p, size=1000)",
        "description": """**Biology/Neuroscience Applications:**

• **Cell Biology:** Number of cells responding to treatment out of n total cells, successful transfections in cell culture experiments, and viable cells after drug exposure

• **Neuroscience:** Successful synaptic transmissions out of n attempts, number of neurons responding to stimuli in population recordings, and behavioral response trials

• **Clinical Trials:** Number of patients showing improvement out of total enrolled, adverse events in drug studies, and diagnostic test accuracy studies

• **Genetics:** Number of offspring with specific traits, genetic variants found in population samples, and allele transmission patterns across generations""",
        "support_fn": lambda dist: np.array([0, int(dist.args[0])]),  # 0..n
        "technical_notes": """
- Sum of `n` independent Bernoulli(p) trials
- Mean = `np`, Variance = `np(1-p)`, Mode ≈ `np`
- Approaches **Normal(np, np(1-p))** as n increases (Central Limit Theorem)
- When n is large and p is small, approximates **Poisson(λ=np)**
- Symmetric when p=0.5, skewed otherwise
        """
    },
    "Poisson": {
        "type": "discrete",
        "params": {"lambda": {"default": 3.0, "min": 0.1, "max": 20.0, "step": 0.1, "help": "Rate λ > 0 (mean = λ)"}},
        "scipy": ("poisson", lambda p: dict(mu=p["lambda"])),
        "equation": r"P(X=k)=e^{-\lambda}\,\frac{\lambda^k}{k!},\;k=0,1,2,\dots",
        "numpy_code": "np.random.poisson(lam=lam, size=1000)",
        "description": """**Biology/Neuroscience Applications:**

• **Rare Event Modeling:** Number of mutations per genome sequence, action potentials per recording window, and cancer cases per population cohort

• **Neural Recording:** Spike count analysis in electrophysiology, synaptic vesicle release events, and neural population activity during specific time windows

• **Molecular Events:** Protein-protein interactions per cell, DNA repair events, and enzyme-substrate collision counts in biochemical reactions

• **Epidemiology:** Disease outbreak modeling, adverse drug reaction frequencies, and rare genetic variant discoveries in large-scale studies""",
        "support_fn": lambda dist: np.array([0, int(np.ceil(float(dist.ppf(0.999))))]),
        "technical_notes": """
- Models count of rare events in fixed interval
- Mean = Variance = `λ` (**equidispersion property**)
- Approximates Binomial when n is large and p is small (`λ = np`)
- Sum of independent Poisson(λᵢ) is **Poisson(Σλᵢ)**
- Approaches **Normal(λ, λ)** as λ increases
        """
    },
    "Geometric (trials until 1st success)": {
        "type": "discrete",
        "params": {
            "p": {"default": 0.3, "min": 0.01, "max": 0.99, "step": 0.01, "help": "Success probability (support starts at 1)"},
        },
        "scipy": ("geom", lambda p: dict(p=p["p"])),
        "equation": r"P(X=k)=(1-p)^{k-1}p,\;k=1,2,\dots",
        "numpy_code": "np.random.geometric(p=p, size=1000)",
        "description": """**Biology/Neuroscience Applications:**

• **Laboratory Protocols:** Number of attempts until successful cell culture establishment, trials until achieving target transfection efficiency, and experimental iterations until desired outcome

• **Treatment Response:** Clinical trials measuring time-to-response (in discrete intervals), number of treatment cycles until therapeutic effect, and drug titration protocols

• **Evolutionary Biology:** Number of generations until beneficial mutation fixation, breeding attempts until desired trait expression, and selection pressure modeling

• **Neural Plasticity:** Learning trials until criterion performance, stimulation attempts until synaptic potentiation, and behavioral conditioning protocols""",
        "support_fn": lambda dist: np.array([1, int(np.ceil(float(dist.ppf(0.999))))]),
        "technical_notes": """
- Discrete analogue of exponential distribution
- **Memoryless property:** `P(X > n+k | X > n) = P(X > k)`
- Mean = `1/p`, Variance = `(1-p)/p²`
- Mode = 1 (most likely number of trials)
- **Only discrete distribution** with memoryless property
        """
    },
    # Composite Distributions
    "Chi-squared": {
        "type": "continuous",
        "params": {
            "df": {"default": 3.0, "min": 1.0, "max": 20.0, "step": 1.0, "help": "Degrees of freedom"},
        },
        "scipy": ("chi2", lambda p: dict(df=int(p["df"]))),
        "equation": r"f(x)=\frac{1}{2^{\nu/2}\Gamma(\nu/2)} x^{\nu/2-1} e^{-x/2},\;x\ge 0",
        "numpy_code": "np.random.chisquare(df=df, size=1000)",
        "description": """**Biology/Neuroscience Applications:**

• **Statistical Testing:** Goodness-of-fit tests for genetic models, independence tests in contingency tables, and variance testing in experimental data

• **Signal Processing:** Power spectral analysis in EEG/MEG data, noise characterization in imaging systems, and energy distribution in biological signals

• **Quality Control:** Variability assessment in laboratory measurements, outlier detection in high-throughput experiments, and measurement precision validation

• **Composite Process:** Emerges from sum of squared standardized normal variables, representing combined variability from multiple independent sources""",
        "support_fn": lambda dist: np.array([0.0, float(dist.ppf(0.999))]),
        "technical_notes": """
- Sum of `ν` independent squared standard normal variables
- Mean = `ν`, Variance = `2ν`, Mode = `max(0, ν-2)`
- **Special case** of Gamma(ν/2, 2) distribution
- Used in hypothesis testing (**chi-squared test**, likelihood ratio test)
- Approaches **normal distribution** as ν increases
        """
    },
    "Log-normal": {
        "type": "continuous",
        "params": {
            "mu": {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.1, "help": "Mean of underlying normal"},
            "sigma": {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.1, "help": "Std dev of underlying normal"},
        },
        "scipy": ("lognorm", lambda p: dict(s=p["sigma"], scale=np.exp(p["mu"]))),
        "equation": r"f(x)=\frac{1}{x\sigma\sqrt{2\pi}}\exp\left(-\frac{(\ln x-\mu)^2}{2\sigma^2}\right),\;x>0",
        "numpy_code": "np.random.lognormal(mean=mu, sigma=sigma, size=1000)",
        "description": """**Biology/Neuroscience Applications:**

• **Concentration Data:** Drug concentrations in plasma, protein expression levels, and metabolite concentrations where values must be positive

• **Cell Biology:** Cell sizes, organelle volumes, and molecular counts where multiplicative growth processes dominate

• **Survival Analysis:** Survival times in cancer studies, time to disease progression, and equipment failure times in laboratory settings

• **Economic Biology:** Cost distributions in healthcare, resource allocation in biological systems, and multiplicative environmental factors""",
        "support_fn": lambda dist: np.array([float(dist.ppf(0.001)), float(dist.ppf(0.999))]),
        "technical_notes": """
- If `X ~ LogNormal(μ,σ)`, then `ln(X) ~ Normal(μ,σ)`
- Mean = `exp(μ + σ²/2)`, Median = `exp(μ)`
- Always **positively skewed**, no negative values possible
- Arises from **multiplicative processes** (product of many positive random variables)
- Used when data spans **several orders of magnitude**
        """
    },
    "Student's t": {
        "type": "continuous",
        "params": {
            "df": {"default": 5.0, "min": 1.0, "max": 30.0, "step": 1.0, "help": "Degrees of freedom"},
            "loc": {"default": 0.0, "min": -3.0, "max": 3.0, "step": 0.1, "help": "Location parameter"},
        },
        "scipy": ("t", lambda p: dict(df=p["df"], loc=p["loc"])),
        "equation": r"f(x)=\frac{\Gamma((\nu+1)/2)}{\sqrt{\nu\pi}\,\Gamma(\nu/2)}\left(1+\frac{x^2}{\nu}\right)^{-(\nu+1)/2}",
        "numpy_code": "np.random.standard_t(df=df, size=1000) + loc",
        "description": """**Biology/Neuroscience Applications:**

• **Small Sample Inference:** Statistical testing with small sample sizes in expensive biological experiments, confidence intervals for means with unknown variance

• **Robust Statistics:** Analysis of biological data with outliers, heavy-tailed distributions in genomics, and robust parameter estimation

• **Clinical Trials:** Statistical inference in pilot studies, interim analyses with limited data, and bioequivalence testing

• **Measurement Error:** Modeling measurement uncertainty in precision instruments, accounting for additional variability beyond normal assumptions""",
        "support_fn": lambda dist: np.array([float(dist.ppf(0.001)), float(dist.ppf(0.999))]),
        "technical_notes": """
- Approaches **standard normal** as df → ∞
- **Heavier tails** than normal distribution (more robust to outliers)
- Mean = 0 (for df > 1), Variance = `df/(df-2)` for df > 2
- Used in **t-tests**, confidence intervals for means
- Arises when estimating mean of normal distribution with **unknown variance**
        """
    },
}

# =============================================================
# 2. Sidebar Mode Selection
# =============================================================

mode = st.sidebar.radio("Mode", ["Single Distribution", "Composite Distribution"])

# =============================================================
# 3. Single Distribution Visualizer
# =============================================================
if mode == "Single Distribution":
    st.markdown('<h1 class="main-header">📊 Interactive Distribution Explorer</h1>', unsafe_allow_html=True)
    st.caption("Explore probability distributions with real-time parameter adjustment and biological science applications")
    
    with st.sidebar:
        st.header("Controls")
        dist_name = st.selectbox("Choose a distribution", list(DISTRIBUTIONS.keys()))
        meta = DISTRIBUTIONS[dist_name]

        # Parameter sliders
        st.subheader("Parameters")
        params: Dict[str, Any] = {}
        for pname, spec in meta["params"].items():
            if meta["type"] == "discrete" and isinstance(spec.get("default"), (int, np.integer)):
                params[pname] = st.slider(pname, min_value=int(spec["min"]), max_value=int(spec["max"]), value=int(spec["default"]), step=int(spec.get("step", 1)), help=spec.get("help"))
            else:
                params[pname] = st.slider(pname, min_value=float(spec["min"]), max_value=float(spec["max"]), value=float(spec["default"]), step=float(spec.get("step", 0.1)), help=spec.get("help"))

        st.subheader("Display")
        view = st.radio("Plot", ["PDF/PMF", "CDF"], index=0, horizontal=True)

        show_stats = st.checkbox("Show mean / median / mode", value=True)
        show_sample = st.checkbox("Overlay simulated samples", value=False)
        sample_n = st.slider("# samples", min_value=100, max_value=100_000, value=2000, step=100)

        st.subheader("Plot Options")
        bins = st.slider("Histogram bins (for samples)", min_value=10, max_value=200, value=50, step=5)

    # -------------------------------------------------------------
    # Build scipy.stats frozen distribution
    # -------------------------------------------------------------
    scipy_name, kw_fn = meta["scipy"]
    dist = getattr(stats, scipy_name)(**kw_fn(params))

    # -------------------------------------------------------------
    # Display equation and description before visualization
    # -------------------------------------------------------------
    # Center-aligned equation section
    st.markdown('<div style="text-align: center;">', unsafe_allow_html=True)
    st.subheader("📐 Mathematical Equation")
    st.latex(meta["equation"])  # PDF/PMF equation in LaTeX
    st.markdown('</div>', unsafe_allow_html=True)

    # Single column application section
    st.subheader("🧬 Applications in Biological Sciences")
    st.write(meta["description"].strip())

    st.markdown("---")

    # -------------------------------------------------------------
    # Prepare domain (x or k) and values using fixed ranges
    # -------------------------------------------------------------
    if meta["type"] == "continuous":
        # Use fixed range instead of parameter-dependent range
        lo, hi = FIXED_RANGES[dist_name]
        # Special handling for Uniform distribution to show the actual range
        if dist_name == "Uniform [a,b]":
            a, b = params["a"], params["b"]
            range_span = max(abs(b - a), 2)  # minimum span of 2
            center = (a + b) / 2
            lo = center - 3 * range_span
            hi = center + 3 * range_span
            lo, hi = max(lo, -12), min(hi, 12)  # clamp to reasonable bounds
        
        x = np.linspace(lo, hi, 600)
        ypdf = dist.pdf(x)
        ycdf = dist.cdf(x)
    else:
        # Use fixed range for discrete distributions
        kmin, kmax = FIXED_RANGES[dist_name]
        # Special handling for binomial - adjust max based on n
        if dist_name == "Binomial":
            kmax = min(int(params["n"]) + 5, 100)  # n + buffer, capped at 100
        
        k = np.arange(kmin, kmax + 1)
        ypmf = dist.pmf(k)
        ycdf = dist.cdf(k)

    # -------------------------------------------------------------
    # Compute summary statistics
    # -------------------------------------------------------------
    try:
        mean = float(dist.mean())
    except Exception:
        mean = np.nan
    try:
        median = float(dist.median())
    except Exception:
        # Fallback via 0.5-quantile
        try:
            median = float(dist.ppf(0.5))
        except Exception:
            median = np.nan
    try:
        # scipy returns array for mode sometimes
        mode_arr = dist.mode()
        mode = float(np.atleast_1d(mode_arr)[0])
    except Exception:
        # heuristic fallbacks for common cases
        if scipy_name == "norm":
            mode = mean
        elif scipy_name == "expon":
            mode = 0.0
        elif scipy_name == "uniform":
            mode = np.nan  # no unique mode
        else:
            mode = np.nan

    # -------------------------------------------------------------
    # Plotting
    # -------------------------------------------------------------
    fig = go.Figure()

    if meta["type"] == "continuous":
        if view == "PDF/PMF":
            fig.add_trace(go.Scatter(
                x=x, y=ypdf, mode="lines", name="PDF",
                line=dict(color=SOFT_COLORS['primary'], width=3)
            ))
        else:
            fig.add_trace(go.Scatter(
                x=x, y=ycdf, mode="lines", name="CDF",
                line=dict(color=SOFT_COLORS['secondary'], width=3)
            ))

        if show_sample:
            samples = dist.rvs(size=sample_n, random_state=42)
            # Histogram normalized to density for PDF view; to probability for CDF view overlay still shows density
            histnorm = "probability density" if view == "PDF/PMF" else "probability"
            fig.add_trace(
                go.Histogram(
                    x=samples, nbinsx=bins, name="Samples", opacity=0.4, 
                    histnorm=histnorm,
                    marker_color=SOFT_COLORS['accent']
                )
            )
    else:
        if view == "PDF/PMF":
            fig.add_trace(go.Bar(
                x=k, y=ypmf, name="PMF",
                marker_color=SOFT_COLORS['primary'], opacity=0.8
            ))
        else:
            fig.add_trace(go.Scatter(
                x=k, y=ycdf, mode="lines+markers", name="CDF",
                line=dict(color=SOFT_COLORS['secondary'], width=3),
                marker=dict(size=6, color=SOFT_COLORS['secondary'])
            ))

        if show_sample:
            samples = dist.rvs(size=sample_n, random_state=42)
            # Empirical PMF
            counts = np.bincount(samples - (1 if scipy_name == "geom" else 0)) if scipy_name == "geom" else np.bincount(samples)
            xs = (np.arange(len(counts)) + (1 if scipy_name == "geom" else 0))
            probs = counts / counts.sum() if counts.sum() > 0 else counts
            fig.add_trace(go.Bar(
                x=xs, y=probs, name="Empirical", opacity=0.5,
                marker_color=SOFT_COLORS['accent']
            ))

    # Stats lines with smart positioning to avoid overlap
    if show_stats:
        # Collect all valid stats and their positions
        stats_data = []
        if not np.isnan(mean):
            stats_data.append((mean, "mean", SOFT_COLORS['accent']))
        if not np.isnan(median):
            stats_data.append((median, "median", SOFT_COLORS['secondary']))
        if not np.isnan(mode):
            stats_data.append((mode, "mode", SOFT_COLORS['purple']))
        
        # Sort by x-position
        stats_data.sort(key=lambda x: x[0])
        
        # Add lines with enhanced y-positioning to avoid overlap
        y_positions = ["top", "top right", "top left", "bottom right", "bottom left"]
        if meta["type"] == "continuous":
            x_range = hi - lo
        else:
            x_range = max(k) - min(k) if len(k) > 0 else 1
        min_distance = 0.2 * x_range  # Increased minimum distance threshold
        
        for i, (x_val, label, color) in enumerate(stats_data):
            # Check if this line is too close to previous ones
            y_pos = "top"
            if i > 0:
                prev_x = stats_data[i-1][0]
                if abs(x_val - prev_x) < min_distance:
                    y_pos = y_positions[i % len(y_positions)]
            
            fig.add_vline(
                x=x_val, 
                line=dict(dash="dash", color=color, width=2), 
                annotation_text=f"{label}: {x_val:.3f}",
                annotation_position=y_pos,
                annotation=dict(
                    font=dict(size=11, color=color),
                    bgcolor=None,  # Remove background box
                    bordercolor=None,  # Remove border
                    borderwidth=0
                )
            )

    fig.update_layout(
        height=520,
        margin=dict(l=40, r=20, t=40, b=40),
        xaxis_title="x" if meta["type"] == "continuous" else "k",
        yaxis_title="Density" if view == "PDF/PMF" and meta["type"] == "continuous" else ("Probability" if view == "PDF/PMF" else "CDF"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color=SOFT_COLORS['text_primary'], size=12),
        xaxis=dict(gridcolor='rgba(132, 169, 140, 0.3)', gridwidth=1),
        yaxis=dict(gridcolor='rgba(132, 169, 140, 0.3)', gridwidth=1)
    )

    st.plotly_chart(fig, use_container_width=True)

    # -------------------------------------------------------------
    # Parameter table and summary statistics (single column layout)
    # -------------------------------------------------------------
    # Parameter table
    st.subheader("📊 Current Parameters")
    rows = []
    for pname, spec in meta["params"].items():
        rows.append((pname, params[pname], spec.get("help", "")))
    st.table({"parameter": [r[0] for r in rows], "value": [r[1] for r in rows], "notes": [r[2] for r in rows]})

    # Summary statistics
    st.subheader("📈 Summary Statistics")
    # For some distributions variance or higher moments may be inf; handle gracefully
    try:
        var = float(dist.var())
        sd = math.sqrt(var) if var >= 0 and np.isfinite(var) else np.nan
    except Exception:
        var, sd = np.nan, np.nan

    # Display metrics in a more compact way
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Mean", f"{mean:.6g}" if np.isfinite(mean) else "—")
    with col2:
        st.metric("Median", f"{median:.6g}" if np.isfinite(median) else "—")
    with col3:
        st.metric("Mode", f"{mode:.6g}" if np.isfinite(mode) else "—")
    with col4:
        st.metric("Variance", f"{var:.6g}" if np.isfinite(var) else "—")
    with col5:
        st.metric("Std. dev.", f"{sd:.6g}" if np.isfinite(sd) else "—")

    st.subheader("💻 NumPy Usage & Technical Notes")

    # NumPy code example for current distribution
    st.write(f"**Generate random samples with NumPy:**")
    st.code(meta["numpy_code"], language='python')

    # Distribution-specific technical notes with proper formatting
    st.markdown("**Technical Details:**")
    technical_notes = meta.get("technical_notes", "No specific technical notes available for this distribution.")
    st.markdown(technical_notes)

# =============================================================
# 4. Composite Distribution Visualizer
# =============================================================
else:
    st.title("🧩 Composite Distributions")
    st.caption("Explore how combining distributions creates new mathematical relationships")

    composite = st.selectbox(
        "Choose composite type",
        [
            "Sum of two Gaussians",
            "Product of two Gaussians", 
            "Sum of two Poissons",
            "Linear transform of Gaussian",
            "F-distribution (Ratio of Chi-squared)",
        ],
    )

    # Fixed x-axis ranges for composite distributions
    COMPOSITE_RANGES = {
        "Sum of two Gaussians": (-15, 15),
        "Product of two Gaussians": (-10, 10),
        "Sum of two Poissons": (0, 50),
        "Linear transform of Gaussian": (-20, 20),
        "F-distribution (Ratio of Chi-squared)": (0, 8),
    }

    # Better organized parameter panel
    st.subheader("📊 Distribution Parameters")
    
    # --------------------------------------------------
    # 1. Sum of two Gaussians (analytic)
    # --------------------------------------------------
    if composite == "Sum of two Gaussians":
        # Organized parameter layout
        col1, col2 = st.columns(2)
        with col1:
            st.write("**First Gaussian (X₁)**")
            mu1 = st.slider("μ₁", -5.0, 5.0, 0.0, 0.1)
            sigma1 = st.slider("σ₁", 0.1, 3.0, 1.0, 0.1)
        with col2:
            st.write("**Second Gaussian (X₂)**") 
            mu2 = st.slider("μ₂", -5.0, 5.0, 1.0, 0.1)
            sigma2 = st.slider("σ₂", 0.1, 3.0, 1.5, 0.1)

        # Calculate composite
        mu_z = mu1 + mu2
        sigma_z = np.sqrt(sigma1**2 + sigma2**2)
        
        # Fixed range for plotting
        x_range = COMPOSITE_RANGES[composite]
        x = np.linspace(x_range[0], x_range[1], 400)
        
        # Original distributions
        pdf1 = stats.norm.pdf(x, mu1, sigma1)
        pdf2 = stats.norm.pdf(x, mu2, sigma2)
        
        # Composite distribution
        pdf_composite = stats.norm.pdf(x, mu_z, sigma_z)

        # Display equation and results
        st.subheader("📐 Mathematical Foundation")
        st.latex(r"Z = X_1 + X_2,\quad X_1 \sim \mathcal{N}(\mu_1,\sigma_1^2),\quad X_2 \sim \mathcal{N}(\mu_2,\sigma_2^2)")
        st.latex(r"Z \sim \mathcal{N}(\mu_1 + \mu_2, \sigma_1^2 + \sigma_2^2)")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Composite Mean", f"{mu_z:.3f}", f"μ₁ + μ₂")
        with col2:
            st.metric("Composite Std", f"{sigma_z:.3f}", f"√(σ₁² + σ₂²)")
        with col3:
            st.metric("Variance", f"{sigma_z**2:.3f}", f"σ₁² + σ₂²")

    # --------------------------------------------------
    # 2. Product of two Gaussians (sampling + special cases)
    # --------------------------------------------------
    elif composite == "Product of two Gaussians":
        # Parameter layout
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            st.write("**First Gaussian (X₁)**")
            mu1 = st.slider("μ₁", -2.0, 2.0, 0.0, 0.1)
            sigma1 = st.slider("σ₁", 0.1, 2.0, 1.0, 0.1)
        with col2:
            st.write("**Second Gaussian (X₂)**")
            mu2 = st.slider("μ₂", -2.0, 2.0, 0.0, 0.1)
            sigma2 = st.slider("σ₂", 0.1, 2.0, 1.0, 0.1)
        with col3:
            st.write("**Simulation Settings**")
            n_samples = st.slider("Samples", 10000, 100000, 50000, 10000)
            bins = st.slider("Histogram bins", 100, 300, 150, 25)

        # Generate samples and compute histogram with better range handling
        np.random.seed(42)  # For reproducible results
        x1_samples = np.random.normal(mu1, sigma1, n_samples)
        x2_samples = np.random.normal(mu2, sigma2, n_samples)
        z_samples = x1_samples * x2_samples
        
        # Adaptive range based on sample quantiles for better visualization
        q01, q99 = np.percentile(z_samples, [1, 99])
        x_range_adaptive = (max(q01, -15), min(q99, 15))  # Clamp to reasonable bounds
        
        # Fixed range for plotting (for consistent comparison)
        x_range = COMPOSITE_RANGES[composite]
        x = np.linspace(x_range[0], x_range[1], 400)
        
        # Original distributions
        pdf1 = stats.norm.pdf(x, mu1, sigma1)
        pdf2 = stats.norm.pdf(x, mu2, sigma2)
        
        # Composite distribution from histogram with adaptive range
        hist_counts, hist_bins = np.histogram(z_samples, bins=bins, range=x_range_adaptive, density=True)
        x_composite = (hist_bins[1:] + hist_bins[:-1]) / 2
        pdf_composite = hist_counts

        # Display equation and properties with more mathematical detail
        st.subheader("📐 Mathematical Foundation") 
        st.latex(r"Z = X_1 \times X_2,\quad X_1 \sim \mathcal{N}(\mu_1,\sigma_1^2),\quad X_2 \sim \mathcal{N}(\mu_2,\sigma_2^2)")
        
        # Special case analysis
        if abs(mu1) < 0.01 and abs(mu2) < 0.01:
            st.info("**Special Case:** When μ₁ = μ₂ = 0, the product has known properties but no simple closed form.")
            st.latex(r"\text{When } \mu_1 = \mu_2 = 0: \quad E[Z] = 0, \quad \text{Var}(Z) = \sigma_1^2\sigma_2^2")
        else:
            st.warning("**General Case:** No simple closed-form PDF exists. The distribution exhibits complex behavior with potential multimodality.")
        
        # Theoretical moments
        theoretical_mean = mu1 * mu2 + 0  # Cov(X1,X2) = 0 for independent
        theoretical_var = (mu1**2 * sigma2**2 + mu2**2 * sigma1**2 + sigma1**2 * sigma2**2)
        
        st.markdown("**Mathematical Properties:**")
        st.markdown(f"""
        - **Theoretical Mean:** `E[Z] = μ₁μ₂ = {theoretical_mean:.3f}`
        - **Theoretical Variance:** `Var(Z) = μ₁²σ₂² + μ₂²σ₁² + σ₁²σ₂² = {theoretical_var:.3f}`
        - **Distribution Shape:** Generally heavy-tailed, potentially multimodal, non-Gaussian
        """)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Sample Mean", f"{np.mean(z_samples):.3f}")
        with col2:
            st.metric("Sample Std", f"{np.std(z_samples):.3f}")
        with col3:
            st.metric("Theoretical Mean", f"{theoretical_mean:.3f}")
        
        # Additional statistical information
        sample_skew = stats.skew(z_samples)
        sample_kurt = stats.kurtosis(z_samples)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Sample Skewness", f"{sample_skew:.3f}")
        with col2:
            st.metric("Sample Kurtosis", f"{sample_kurt:.3f}")
        with col3:
            st.metric("Sample Range", f"[{np.min(z_samples):.2f}, {np.max(z_samples):.2f}]")

    # --------------------------------------------------
    # 3. Sum of Poissons (analytic)
    # --------------------------------------------------
    elif composite == "Sum of two Poissons":
        # Parameter layout
        col1, col2 = st.columns(2)
        with col1:
            st.write("**First Poisson (X₁)**")
            lam1 = st.slider("λ₁", 0.5, 15.0, 3.0, 0.5)
        with col2:
            st.write("**Second Poisson (X₂)**")
            lam2 = st.slider("λ₂", 0.5, 15.0, 5.0, 0.5)
        
        # Calculate composite
        lam_z = lam1 + lam2
        
        # Fixed range for plotting
        x_range = COMPOSITE_RANGES[composite]
        k = np.arange(x_range[0], min(x_range[1], int(5 * lam_z)) + 1)
        
        # Original distributions
        pmf1 = stats.poisson.pmf(k, lam1)
        pmf2 = stats.poisson.pmf(k, lam2)
        
        # Composite distribution
        pmf_composite = stats.poisson.pmf(k, lam_z)
        
        x = k  # For discrete case
        pdf1, pdf2, pdf_composite = pmf1, pmf2, pmf_composite

        # Display equation and results
        st.subheader("📐 Mathematical Foundation")
        st.latex(r"Z = X_1 + X_2,\quad X_1 \sim \text{Poisson}(\lambda_1),\quad X_2 \sim \text{Poisson}(\lambda_2)")
        st.latex(r"Z \sim \text{Poisson}(\lambda_1 + \lambda_2)")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Composite Rate", f"{lam_z:.1f}", f"λ₁ + λ₂")
        with col2:
            st.metric("Composite Mean", f"{lam_z:.1f}", f"= λ")
        with col3:
            st.metric("Composite Variance", f"{lam_z:.1f}", f"= λ")

    # --------------------------------------------------
    # 4. Linear transform of Gaussian (analytic)
    # --------------------------------------------------
    elif composite == "Linear transform of Gaussian":
        # Parameter layout
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Original Gaussian (X)**")
            mu = st.slider("μ", -5.0, 5.0, 0.0, 0.1)
            sigma = st.slider("σ", 0.1, 3.0, 1.0, 0.1)
        with col2:
            st.write("**Linear Transformation**")
            a = st.slider("a (scale)", -3.0, 3.0, 2.0, 0.1)
            b = st.slider("b (shift)", -5.0, 5.0, 1.0, 0.1)

        # Calculate composite
        mu_z = a * mu + b
        sigma_z = abs(a) * sigma
        
        # Fixed range for plotting
        x_range = COMPOSITE_RANGES[composite]
        x = np.linspace(x_range[0], x_range[1], 400)
        
        # Original distribution
        pdf1 = stats.norm.pdf(x, mu, sigma)
        pdf2 = np.zeros_like(x)  # No second distribution
        
        # Composite distribution
        pdf_composite = stats.norm.pdf(x, mu_z, sigma_z)

        # Display equation and results
        st.subheader("📐 Mathematical Foundation")
        st.latex(r"Z = aX + b,\quad X \sim \mathcal{N}(\mu,\sigma^2)")
        st.latex(r"Z \sim \mathcal{N}(a\mu + b, a^2\sigma^2)")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Composite Mean", f"{mu_z:.3f}", f"aμ + b")
        with col2:
            st.metric("Composite Std", f"{sigma_z:.3f}", f"|a|σ")
        with col3:
            st.metric("Scale Factor", f"{abs(a):.1f}", f"|a|")

    # --------------------------------------------------
    # 5. F-distribution (Ratio of Chi-squared) - ANALYTIC
    # --------------------------------------------------
    elif composite == "F-distribution (Ratio of Chi-squared)":
        # Parameter layout
        col1, col2 = st.columns(2)
        with col1:
            st.write("**First Chi-squared (X₁)**")
            df1 = st.slider("ν₁ (degrees of freedom)", 1, 30, 5, 1)
        with col2:
            st.write("**Second Chi-squared (X₂)**")
            df2 = st.slider("ν₂ (degrees of freedom)", 1, 30, 10, 1)

        # Fixed range for plotting
        x_range = COMPOSITE_RANGES[composite]
        x = np.linspace(0.01, x_range[1], 400)  # Start slightly above 0
        
        # Original Chi-squared distributions
        chi2_1 = stats.chi2(df=df1)
        chi2_2 = stats.chi2(df=df2)
        pdf1 = chi2_1.pdf(x * df2)  # Scale for visualization
        pdf2 = chi2_2.pdf(x * df1)  # Scale for visualization
        
        # F-distribution (analytic)
        f_dist = stats.f(dfn=df1, dfd=df2)
        pdf_composite = f_dist.pdf(x)

        # Display equation and results
        st.subheader("📐 Mathematical Foundation")
        st.latex(r"F = \frac{X_1/\nu_1}{X_2/\nu_2},\quad X_1 \sim \chi^2(\nu_1),\quad X_2 \sim \chi^2(\nu_2)")
        st.latex(r"F \sim F(\nu_1, \nu_2) \text{ with PDF: } f(x) = \frac{\Gamma\left(\frac{\nu_1+\nu_2}{2}\right)}{\Gamma\left(\frac{\nu_1}{2}\right)\Gamma\left(\frac{\nu_2}{2}\right)} \left(\frac{\nu_1}{\nu_2}\right)^{\nu_1/2} \frac{x^{\nu_1/2-1}}{(1+\frac{\nu_1}{\nu_2}x)^{(\nu_1+\nu_2)/2}}")
        
        # Theoretical properties
        if df2 > 2:
            f_mean = df2 / (df2 - 2)
        else:
            f_mean = np.inf
            
        if df2 > 4:
            f_var = (2 * df2**2 * (df1 + df2 - 2)) / (df1 * (df2 - 2)**2 * (df2 - 4))
        else:
            f_var = np.inf
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Theoretical Mean", f"{f_mean:.3f}" if np.isfinite(f_mean) else "∞", f"ν₂/(ν₂-2)")
        with col2:
            st.metric("Theoretical Variance", f"{f_var:.3f}" if np.isfinite(f_var) else "∞")
        with col3:
            st.metric("Mode", f"{max(0, (df1-2)*df2/(df1*(df2+2))):.3f}" if df1 > 2 else "0")
        
        st.markdown("**Applications in Biology:**")
        st.markdown("""
        - **ANOVA:** Testing equality of variances between groups
        - **Genetics:** Comparing genetic variance components  
        - **Clinical Trials:** Testing treatment effect variability
        - **Quality Control:** Comparing measurement precision between instruments
        """)

    # --------------------------------------------------
    # Plot composite result with original distributions
    # --------------------------------------------------
    st.subheader("📈 Distribution Visualization")
    
    fig = go.Figure()
    
    # Plot original distributions
    if composite != "Linear transform of Gaussian":
        if composite == "Sum of two Poissons":
            # Discrete case
            fig.add_trace(go.Bar(
                x=x, y=pdf1, name="X₁ (Original)", 
                marker_color="#55A868", opacity=0.6, width=0.8
            ))
            fig.add_trace(go.Bar(
                x=x, y=pdf2, name="X₂ (Original)", 
                marker_color="#8172B3", opacity=0.6, width=0.8
            ))
            fig.add_trace(go.Bar(
                x=x, y=pdf_composite, name="Z = X₁ + X₂ (Composite)", 
                marker_color="#C44E52", opacity=0.9, width=0.8
            ))
        else:
            # Continuous case
            fig.add_trace(go.Scatter(
                x=x, y=pdf1, mode="lines", name="X₁ (Original)",
                line=dict(color="#55A868", width=2, dash="dash")
            ))
            fig.add_trace(go.Scatter(
                x=x, y=pdf2, mode="lines", name="X₂ (Original)",
                line=dict(color="#8172B3", width=2, dash="dash")
            ))
            if composite == "Product of two Gaussians":
                fig.add_trace(go.Scatter(
                    x=x_composite, y=pdf_composite, mode="lines", name="Z = X₁ × X₂ (Composite)",
                    line=dict(color="#C44E52", width=3)
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=x, y=pdf_composite, mode="lines", name="Z = X₁ + X₂ (Composite)",
                    line=dict(color="#C44E52", width=3)
                ))
    else:
        if composite == "Linear transform of Gaussian":
            # Linear transform case - show original and transformed
            fig.add_trace(go.Scatter(
                x=x, y=pdf1, mode="lines", name="X (Original)",
                line=dict(color="#55A868", width=2, dash="dash")
            ))
            fig.add_trace(go.Scatter(
                x=x, y=pdf_composite, mode="lines", name="Z = aX + b (Transformed)",
                line=dict(color="#C44E52", width=3)
            ))
        elif composite == "F-distribution (Ratio of Chi-squared)":
            # F-distribution case - show chi-squared components and F result
            fig.add_trace(go.Scatter(
                x=x, y=pdf1/max(pdf1), mode="lines", name="χ²(ν₁) (scaled)",
                line=dict(color="#55A868", width=2, dash="dash")
            ))
            fig.add_trace(go.Scatter(
                x=x, y=pdf2/max(pdf2), mode="lines", name="χ²(ν₂) (scaled)",
                line=dict(color="#8172B3", width=2, dash="dash")
            ))
            fig.add_trace(go.Scatter(
                x=x, y=pdf_composite, mode="lines", name="F(ν₁,ν₂) (Ratio)",
                line=dict(color="#C44E52", width=3)
            ))
    
    fig.update_layout(
        height=500,
        title=f"{composite} - Comparison of Original and Composite Distributions",
        xaxis_title="Value",
        yaxis_title="Density" if composite != "Sum of two Poissons" else "Probability",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12),
        xaxis=dict(gridcolor='rgba(132, 169, 140, 0.3)', gridwidth=1, range=x_range if composite != "Product of two Gaussians" else x_range_adaptive),
        yaxis=dict(gridcolor='rgba(132, 169, 140, 0.3)', gridwidth=1),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

