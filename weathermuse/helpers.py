import torch
import omegaconf
from pathlib import Path
import typing as tp
from omegaconf import OmegaConf
from weatherlm import WeatherLM
from audiocraft.models.loaders import _delete_param, load_lm_model_ckpt
from audiocraft.models.builders import get_lm_model, dict_from_config, get_condition_fuser, get_conditioner_provider, get_codebooks_pattern_provider


def get_weather_lm_model(cfg: omegaconf.DictConfig) -> WeatherLM:
    # Instantiate a transformer LM as WeatherLM
    if cfg.lm_model == "transformer_lm":
        # Extract configuration details
        kwargs = dict_from_config(getattr(cfg, "transformer_lm"))
        n_q = kwargs["n_q"]
        q_modeling = kwargs.pop("q_modeling", None)
        codebooks_pattern_cfg = getattr(cfg, "codebooks_pattern")
        attribute_dropout = dict_from_config(getattr(cfg, "attribute_dropout"))
        cls_free_guidance = dict_from_config(getattr(cfg, "classifier_free_guidance"))
        cfg_prob, cfg_coef = (
            cls_free_guidance["training_dropout"],
            cls_free_guidance["inference_coef"],
        )
        fuser = get_condition_fuser(cfg)
        condition_provider = get_conditioner_provider(kwargs["dim"], cfg).to(cfg.device)

        # Enforce cross-attention programmatically if needed
        if len(fuser.fuse2cond["cross"]) > 0:
            kwargs["cross_attention"] = True

        # Handle codebooks pattern configuration
        if codebooks_pattern_cfg.modeling is None:
            assert (
                q_modeling is not None
            ), "LM model should either have a codebook pattern defined or transformer_lm.q_modeling"
            codebooks_pattern_cfg = omegaconf.OmegaConf.create(
                {"modeling": q_modeling, "delay": {"delays": list(range(n_q))}}
            )
        pattern_provider = get_codebooks_pattern_provider(n_q, codebooks_pattern_cfg)

        # Instantiate WeatherLM directly
        return WeatherLM(
            pattern_provider=pattern_provider,
            condition_provider=condition_provider,
            fuser=fuser,
            cfg_dropout=cfg_prob,
            cfg_coef=cfg_coef,
            attribute_dropout=attribute_dropout,
            dtype=getattr(torch, cfg.dtype),
            device=cfg.device,
            **kwargs,
        ).to(cfg.device)
    else:
        raise KeyError(f"Unexpected LM model {cfg.lm_model}")
    
def load_lm_model(file_or_url_or_id: tp.Union[Path, str], device='cpu', cache_dir: tp.Optional[str] = None):
    pkg = load_lm_model_ckpt(file_or_url_or_id, cache_dir=cache_dir)
    cfg = OmegaConf.create(pkg['xp.cfg'])
    cfg.device = str(device)
    if cfg.device == 'cpu':
        cfg.dtype = 'float32'
    else:
        cfg.dtype = 'float16'
    _delete_param(cfg, 'conditioners.self_wav.chroma_stem.cache_path')
    _delete_param(cfg, 'conditioners.args.merge_text_conditions_p')
    _delete_param(cfg, 'conditioners.args.drop_desc_p')

    # Use WeatherLM if transformer_lm
    if cfg.lm_model == "transformer_lm":
        model = get_weather_lm_model(cfg)
    else:
        model = get_lm_model(cfg)

    model.load_state_dict(pkg['best_state'])
    model.eval()
    model.cfg = cfg
    return model