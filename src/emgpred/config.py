"""Leitura dos arquivos de configuração TOML.

Caminhos aceitam variáveis de ambiente (`$EMGPRED_DATA`, `~`), e qualquer chave pode
ser sobrescrita na linha de comando com `--set secao.chave=valor`.
"""
import ast
import os
import tomllib


def _expand(obj):
    if isinstance(obj, dict):
        return {k: _expand(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand(v) for v in obj]
    if isinstance(obj, str):
        return os.path.expanduser(os.path.expandvars(obj))
    return obj


def _parse_value(text):
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def load_config(path, overrides=()):
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    for item in overrides:
        key, _, value = item.partition("=")
        *parents, leaf = key.strip().split(".")
        node = cfg
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = _parse_value(value.strip())
    return _expand(cfg)
