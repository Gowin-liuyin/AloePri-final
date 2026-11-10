"""Llama扩维入口：保留Llama架构、头宽和偏置，复用已校验的离线矩阵融合。"""
from src.aloepri._implementation.export.expanded_qwen_checkpoint import (
    build_expanded_decoder_model,export_expanded_decoder_checkpoint,
)


def build_expanded_llama_model(baseline,tokenizer,**kwargs):
    if baseline.config.model_type!="llama":
        raise ValueError("Expected Llama baseline")
    kwargs.setdefault("attention_profile", "rqk_hqk")
    return build_expanded_decoder_model(baseline,tokenizer,**kwargs)


def export_expanded_llama_checkpoint(export_dir,*,model_dir,**kwargs):
    kwargs.setdefault("attention_profile", "rqk_hqk")
    return export_expanded_decoder_checkpoint(export_dir,model_dir=model_dir,expected_model_type="llama",**kwargs)
