"""
ConditionedFNO: FNO with language conditioning (FiLM or Spectral Gating).

APEBench interface: model(u_t) -> u_{t+1}

The conditioning vector z is stored inside the model as a frozen constant.
Freezing is enforced by jax.lax.stop_gradient in the forward pass, so no
gradients flow into z during training. Only FNO weights, z_proj, and
conditioning layers are updated.

Architecture:
    u_t
     |
     v
    [FNO lifting layer]
     |
     v  <---- z (frozen, stop_gradient)
    [FNO spectral block 0]      |
     |                  z_cond = z_proj(z)   (trained linear projection)
    [conditioning layer]    <---+
     |
    [FNO spectral block 1]
     |
    [conditioning layer]    <--- same z_cond
     |
    ...
     |
    [FNO projection layer]
     |
     v
    u_{t+1}

Conditioning methods (primary ablation axis):
    "film":
        FiLMLayer — feature-wise linear modulation after each spectral block.
        x -> (1 + gamma) * x + beta, where gamma/beta are linear in z_cond.

    "spectral_gating":
        SpectralGating — learned per-frequency gate applied in Fourier space.
        Computes gate = 1 + alpha * tanh(MLP(z_cond)) for the first num_modes
        frequencies, multiplies the rfft output, then irffts back.
"""

from typing import Callable, List

import equinox as eqx
import jax
import jax.numpy as jnp
import pdequinox as pdeqx
from jaxtyping import Array, PRNGKeyArray


# ── FiLM layer ────────────────────────────────────────────────────────────────

class FiLMLayer(eqx.Module):
    """
    Feature-wise Linear Modulation.

    Given conditioning vector z_cond, produces per-channel scale (gamma) and
    shift (beta), then applies:  x -> (1 + gamma) * x + beta

    Operates on spatial feature maps of shape (channels, *spatial_dims).
    """
    gamma: eqx.nn.Linear
    beta: eqx.nn.Linear
    num_spatial_dims: int

    def __init__(
        self,
        cond_dim: int,
        hidden_channels: int,
        num_spatial_dims: int,
        *,
        key: PRNGKeyArray,
    ):
        k1, k2 = jax.random.split(key)
        # Initialize gamma near 1 and beta near 0 so FiLM starts as identity
        self.gamma = eqx.nn.Linear(cond_dim, hidden_channels, key=k1)
        self.beta = eqx.nn.Linear(cond_dim, hidden_channels, key=k2)
        self.num_spatial_dims = num_spatial_dims

    def __call__(self, x: Array, z_cond: Array) -> Array:
        # x      : (hidden_channels, *spatial_dims)
        # z_cond : (cond_dim,)
        gamma = self.gamma(z_cond)   # (hidden_channels,)
        beta = self.beta(z_cond)     # (hidden_channels,)
        # Reshape for broadcasting over all spatial dims
        shape = (-1,) + (1,) * self.num_spatial_dims
        return (1.0 + gamma.reshape(shape)) * x + beta.reshape(shape)
        # Note: (1 + gamma) instead of gamma so FiLM starts as identity
        # when gamma weights are near zero at init.


# ── Spectral Gating layer ─────────────────────────────────────────────────────

class SpectralGating(eqx.Module):
    """
    Spectral gating: modulates the first num_modes Fourier coefficients of x
    using a frequency-dependent gate derived from z_cond.

    Gate formula:  gate_k = 1 + alpha * tanh(MLP(z_cond)_k)

    The 1 + ... ensures identity at initialization (MLP weights near zero
    -> gate_raw ≈ 0 -> tanh ≈ 0 -> gate ≈ 1). alpha controls how far the
    gate can deviate from identity, keeping training stable.

    Applied after each FNO spectral block (same placement as FiLM).
    """
    gate_mlp: eqx.nn.MLP
    num_modes: int
    alpha: float

    def __init__(
        self,
        cond_dim: int,
        num_modes: int,
        mlp_width: int = 64,
        mlp_depth: int = 2,
        *,
        key: PRNGKeyArray,
    ):
        self.gate_mlp = eqx.nn.MLP(
            in_size=cond_dim,
            out_size=num_modes,
            width_size=mlp_width,
            depth=mlp_depth,
            key=key,
        )
        self.num_modes = num_modes
        self.alpha = 0.1

    def __call__(self, x: Array, z_cond: Array) -> Array:
        # x      : (channels, spatial_dim)
        # z_cond : (cond_dim,)
        n = x.shape[-1]
        x_hat = jnp.fft.rfft(x, axis=-1)                   # (channels, n//2+1)
        gate_raw = self.gate_mlp(z_cond)                     # (num_modes,)
        gate = 1.0 + self.alpha * jnp.tanh(gate_raw)        # (num_modes,)
        # Multiply only the first num_modes frequency bins; broadcast over channels
        x_hat = x_hat.at[:, :self.num_modes].multiply(gate[None, :])
        return jnp.fft.irfft(x_hat, n=n, axis=-1)           # (channels, spatial_dim)


# ── ConditionedFNO ────────────────────────────────────────────────────────────

class ConditionedFNO(eqx.Module):
    """
    FNO with per-block language conditioning.

    Replicates ClassicFNO's forward pass (lifting -> blocks -> projection)
    with a conditioning layer inserted after each spectral block.
    The active conditioning method is selected by the conditioning_method field.

    Both film_layers and spectral_gating_layers are always built so the
    PyTree structure is stable regardless of which method is selected.
    The unused set receives zero gradients and stays at initialization.
    """
    # FNO components (extracted from pdequinox ClassicFNO / Sequential)
    fno_lifting: eqx.Module
    fno_blocks: List[eqx.Module]
    fno_projection: eqx.Module

    # Shared projection: z_embed_dim -> cond_dim  (always trained)
    z_proj: eqx.nn.Linear

    # Conditioning layers (both built; only the active method receives gradients)
    film_layers: List[FiLMLayer]
    spectral_gating_layers: List[SpectralGating]

    # Selects which conditioning path is active (static string, not a JAX array)
    conditioning_method: str

    # If True, stop_gradient is applied to all FNO trunk outputs so gradients
    # only flow through z_proj and the conditioning layers.
    freeze_trunk: bool

    # Frozen language embedding (stop_gradient enforced in __call__)
    z: Array                     # shape (z_embed_dim,)

    def __init__(
        self,
        num_spatial_dims: int,
        num_channels: int,
        num_modes: int,
        hidden_channels: int,
        num_blocks: int,
        activation: Callable,
        z_embed: Array,           # precomputed LLM embedding, shape (z_embed_dim,)
        cond_dim: int,            # projection target dim
        conditioning_method: str = "film",
        sg_mlp_width: int = 64,
        sg_mlp_depth: int = 2,
        freeze_trunk: bool = True,
        *,
        key: PRNGKeyArray,
    ):
        k_fno, k_proj, k_film, k_sg = jax.random.split(key, 4)

        # Build the base FNO, then extract its components so we can intercept
        # the forward pass and insert conditioning layers between blocks.
        fno = pdeqx.arch.ClassicFNO(
            num_spatial_dims=num_spatial_dims,
            in_channels=num_channels,
            out_channels=num_channels,
            num_modes=num_modes,
            hidden_channels=hidden_channels,
            num_blocks=num_blocks,
            activation=activation,
            key=k_fno,
        )
        self.fno_lifting = fno.lifting
        self.fno_blocks = list(fno.blocks)
        self.fno_projection = fno.projection

        # Projection: z_embed_dim -> cond_dim
        z_embed_dim = z_embed.shape[0]
        self.z_proj = eqx.nn.Linear(z_embed_dim, cond_dim, key=k_proj)

        # FiLM layers (one per block)
        film_keys = jax.random.split(k_film, num_blocks)
        self.film_layers = [
            FiLMLayer(
                cond_dim=cond_dim,
                hidden_channels=hidden_channels,
                num_spatial_dims=num_spatial_dims,
                key=film_keys[i],
            )
            for i in range(num_blocks)
        ]

        # Spectral gating layers (one per block)
        sg_keys = jax.random.split(k_sg, num_blocks)
        self.spectral_gating_layers = [
            SpectralGating(
                cond_dim=cond_dim,
                num_modes=num_modes,
                mlp_width=sg_mlp_width,
                mlp_depth=sg_mlp_depth,
                key=sg_keys[i],
            )
            for i in range(num_blocks)
        ]

        self.conditioning_method = conditioning_method
        self.freeze_trunk = freeze_trunk

        # Store embedding as JAX array; frozen via stop_gradient in forward
        self.z = jnp.array(z_embed)

    def __call__(self, u_t: Array) -> Array:
        # ── Freeze the embedding ──────────────────────────────────────────────
        z = jax.lax.stop_gradient(self.z)       # (z_embed_dim,)
        z_cond = self.z_proj(z)                 # (cond_dim,)   [trained]

        # ── FNO forward with conditioning after each block ────────────────────
        # If freeze_trunk, stop_gradient is applied to trunk weights so they
        # are treated as constants during backprop. Gradients still flow back
        # through the trunk's linear transformation to reach all conditioning
        # layers. Only array leaves are stop_gradiented — non-array fields
        # (ints, bools) are left unchanged to avoid errors.
        def _trunk(fn, x):
            if self.freeze_trunk:
                frozen_fn = jax.tree_util.tree_map(
                    lambda leaf: jax.lax.stop_gradient(leaf) if eqx.is_array(leaf) else leaf,
                    fn,
                )
                return frozen_fn(x)
            return fn(x)

        x = _trunk(self.fno_lifting, u_t)

        if self.conditioning_method == "film":
            for block, film in zip(self.fno_blocks, self.film_layers):
                x = _trunk(block, x)
                x = film(x, z_cond)

        elif self.conditioning_method == "spectral_gating":
            for block, sg in zip(self.fno_blocks, self.spectral_gating_layers):
                x = _trunk(block, x)
                x = sg(x, z_cond)

        x = _trunk(self.fno_projection, x)
        return x


# ── APEBench architecture registration ───────────────────────────────────────

def make_cfno_constructor(
    z_embed: Array,
    cond_dim: int,
    conditioning_method: str = "film",
) -> Callable:
    """
    Returns a constructor compatible with APEBench's architecture_dict signature:

        constructor(
            architecture_config,   # "cfno;MODES;HIDDEN;BLOCKS;ACTIVATION"
            num_spatial_dims,
            num_points,            # unused here (FNO is resolution-independent)
            num_channels,
            activation_fn,         # already-resolved Callable
            key,
        ) -> eqx.Module

    Usage before calling an APEBench scenario:
        from apebench.components._architectures import architecture_dict
        architecture_dict["cfno"] = make_cfno_constructor(
            z, cfg.z_cond_dim, conditioning_method="spectral_gating"
        )
    """
    z_array = jnp.array(z_embed)

    def constructor(
        architecture_config: str,
        num_spatial_dims: int,
        num_points: int,
        num_channels: int,
        activation_fn: Callable,
        key: PRNGKeyArray,
    ) -> ConditionedFNO:
        # Parse: "cfno;MODES;HIDDEN;BLOCKS;ACTIVATION"
        args = architecture_config.split(";")
        num_modes = int(args[1])
        hidden_channels = int(args[2])
        num_blocks = int(args[3])
        # args[4] is the activation name — already resolved as activation_fn

        return ConditionedFNO(
            num_spatial_dims=num_spatial_dims,
            num_channels=num_channels,
            num_modes=num_modes,
            hidden_channels=hidden_channels,
            num_blocks=num_blocks,
            activation=activation_fn,
            z_embed=z_array,
            cond_dim=cond_dim,
            conditioning_method=conditioning_method,
            key=key,
        )

    return constructor
