# McLics - Monte Carlo Liquid Crystal Simulator
# Copyright (C) 2026 The McLics authors (see AUTHORS)
# SPDX-License-Identifier: GPL-3.0-or-later
# Como usar:
# python3 -m venv venv
# source venv/bin/activate

"""
analyze.py — Script de análise dos resultados do McLics-backend
================================================================
Uso:
    python3 -m venv venv                      # cria o ambiente virtual
    source venv/bin/activate                  # ativa o ambiente virtual
    python3 analyze.py                        # lê po.dat na pasta atual
    python3 analyze.py resultados/po.dat      # lê arquivo específico
    python3 analyze.py po.dat -o grafico.png  # salva em arquivo específico
    python3 analyze.py po.dat --show          # exibe o gráfico interativo
    python3 analyze.py po.dat --txt           # imprime tabela de estatísticas
    python3 analyze.py po.dat --dark          # tema escuro (apresentações)
    python3 analyze.py po.dat --mode electric # rótulo de eixo p/ simulação de campo elétrico

Saída:
    - Gráfico em três painéis empilhados (eixo único cada): S(x), E(x) e Var(S)
    - Bandas de erro (±1 desvio padrão) e detecção automática da transição de fase
    - Tabela de estatísticas no terminal (com --txt)

Nota:
    po.dat sempre grava o cabeçalho "T S varS E varE", independente do modo de
    evolução (thermal/electric/step/quench) — o backend não distingue os
    modos no arquivo. Por isso o modo não é autodetectável; use --mode para
    rotular o eixo x corretamente (padrão: thermal).
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MaxNLocator

mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["axes.unicode_minus"] = False


# ── Leitura do arquivo ────────────────────────────────────────────────────────

# NOTA: o backend (evolve_thermal.cpp, evolve_electric.cpp, evolve_step.cpp,
# evolve_quench.cpp) sempre grava o mesmo cabeçalho literal "T S varS E varE"
# em po.dat, independente do modo de evolução real (params.evol). Ou seja,
# o modo NÃO é detectável a partir do conteúdo do arquivo — precisa ser
# informado explicitamente via --mode; caso contrário assume-se "thermal".
MODE_INFO = {
    "thermal":  ("Temperatura (T)", "T"),
    "electric": ("Campo Elétrico (E)", "E"),
    "step":     ("Passo (índice)", "step"),
    "quench":   ("Passo de quench (índice)", "step"),
}


def read_po_dat(file_path, forced_mode=None):
    """
    Lê po.dat e retorna um DataFrame com as colunas:
      x, S, varS, E, varE
    onde x é temperatura, campo elétrico ou índice de passo, conforme --mode.
    """
    if not os.path.isfile(file_path):
        print(f"  [ERRO] Arquivo não encontrado: {file_path}")
        sys.exit(1)

    rows = []

    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Linha de cabeçalho ("T S varS E varE[...]") — não carrega
            # informação de modo, apenas é ignorada.
            if not line[0].replace('.', '').replace('-', '').isdigit():
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                rows.append([float(p) for p in parts[:5]])
            except ValueError:
                continue

    if not rows:
        print("  [ERRO] Nenhum dado válido encontrado em po.dat.")
        sys.exit(1)

    mode = forced_mode or "thermal"
    xlabel, xcol = MODE_INFO[mode]

    df = pd.DataFrame(rows, columns=["x", "S", "varS", "E", "varE"])
    df["errS"] = np.sqrt(df["varS"].clip(lower=0))
    df["errE"] = np.sqrt(df["varE"].clip(lower=0))

    if mode == "thermal":
        df = df.sort_values("x").reset_index(drop=True)

    return df, mode, xlabel, xcol


# ── Detecção de transição de fase ────────────────────────────────────────────

def find_transition(df):
    """
    Estima a temperatura (ou campo) de transição de fase como o ponto de
    maior variação absoluta de dS/dx. Retorna None se não for detectável.
    """
    if len(df) < 4:
        return None
    dS = np.abs(np.gradient(df["S"].values, df["x"].values))
    idx = np.argmax(dS)
    # Só reporta se a variação for significativa
    if dS[idx] < 0.01:
        return None
    return df["x"].iloc[idx]


# ── Estatísticas ──────────────────────────────────────────────────────────────

def print_stats(df, xlabel):
    """Imprime uma tabela de estatísticas descritivas no terminal."""
    sep = "─" * 52
    print(f"\n{sep}")
    print(f"  Estatísticas  —  {len(df)} pontos")
    print(sep)
    print(f"  {'':12s}  {'S':>10s}  {'E':>10s}")
    print(f"  {'Mínimo':12s}  {df['S'].min():10.4f}  {df['E'].min():10.4f}")
    print(f"  {'Máximo':12s}  {df['S'].max():10.4f}  {df['E'].max():10.4f}")
    print(f"  {'Média':12s}  {df['S'].mean():10.4f}  {df['E'].mean():10.4f}")
    print(f"  {'Desvio pad.':12s}  {df['S'].std():10.4f}  {df['E'].std():10.4f}")

    Tx = find_transition(df)
    if Tx is not None:
        print(f"\n  Transição de fase estimada em x ≈ {Tx:.4f}")
    print(sep + "\n")


# ── Paleta ────────────────────────────────────────────────────────────────────

def get_palette(dark=False):
    """
    Paleta validada (seis-checks de acessibilidade/CVD) — mesmos tons,
    ajustados de claro para escuro. slot1=azul (S), slot2=laranja (E).
    """
    if dark:
        return dict(
            page="#0d0d0d", surface="#1a1a19", grid="#2c2c2a", axis="#383835",
            ink="#ffffff", ink2="#c3c2b7", muted="#898781",
            blue="#3987e5", orange="#d95926", ref="#c3c2b7",
        )
    return dict(
        page="#f9f9f7", surface="#fcfcfb", grid="#e1e0d9", axis="#c3c2b7",
        ink="#0b0b0b", ink2="#52514e", muted="#898781",
        blue="#2a78d6", orange="#eb6834", ref="#52514e",
    )


# ── Gráfico ───────────────────────────────────────────────────────────────────

def _style_axis(ax, pal):
    """Aplica grade hairline sólida e recessiva, sem cromo pesado."""
    ax.set_facecolor(pal["surface"])
    ax.grid(True, linestyle="-", linewidth=0.8, alpha=1.0, color=pal["grid"], zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(pal["axis"])
        ax.spines[spine].set_linewidth(0.8)
    ax.tick_params(colors=pal["muted"], labelsize=9.5)


def plot(df, mode, xlabel, output_path, show, dark=False):
    """Gera o gráfico de S(x), E(x) e Var(S), em painéis empilhados de eixo único."""

    pal = get_palette(dark)
    x = df["x"].values
    n = len(x)

    fig = plt.figure(figsize=(10, 8.5), constrained_layout=True)
    fig.patch.set_facecolor(pal["page"])

    gs = fig.add_gridspec(3, 1, hspace=0.06, height_ratios=[3, 3, 1.6])
    ax_s   = fig.add_subplot(gs[0])
    ax_e   = fig.add_subplot(gs[1], sharex=ax_s)
    ax_var = fig.add_subplot(gs[2], sharex=ax_s)

    Tx = find_transition(df)

    # Com muitos pontos próximos, o anel do marcador quebra a linha
    # visualmente (parece tracejada) — acima de ~30 pontos, mostra só a linha.
    show_markers = n <= 30
    marker_kw = dict(marker="o", markersize=6, markeredgewidth=1.2,
                      markeredgecolor=pal["surface"]) if show_markers else dict(marker="")

    # ── Painel 1: S(x) ───────────────────────────────────────────────────────
    _style_axis(ax_s, pal)
    ax_s.plot(x, df["S"], color=pal["blue"], linewidth=2,
              markerfacecolor=pal["blue"], zorder=3, **marker_kw)
    if n > 1:
        ax_s.fill_between(x, df["S"] - df["errS"], df["S"] + df["errS"],
                           color=pal["blue"], alpha=0.12, linewidth=0, zorder=2)
    ax_s.set_ylabel("Parâmetro de ordem  S", color=pal["ink2"], fontsize=11)
    ax_s.set_ylim(-0.05, 1.05)
    ax_s.annotate(f"{df['S'].iloc[-1]:.3f}",
                  (x[-1], df["S"].iloc[-1]), xytext=(6, 0),
                  textcoords="offset points", va="center", fontsize=9,
                  color=pal["ink2"])
    ax_s.set_title("Resultados da Simulação de Monte Carlo — McLics",
                    fontsize=13.5, color=pal["ink"], pad=14, loc="left",
                    fontweight="semibold")
    plt.setp(ax_s.get_xticklabels(), visible=False)

    # ── Painel 2: E(x) ───────────────────────────────────────────────────────
    _style_axis(ax_e, pal)
    ax_e.plot(x, df["E"], color=pal["orange"], linewidth=2,
              markerfacecolor=pal["orange"], zorder=3, **marker_kw)
    if n > 1:
        ax_e.fill_between(x, df["E"] - df["errE"], df["E"] + df["errE"],
                           color=pal["orange"], alpha=0.14, linewidth=0, zorder=2)
    ax_e.set_ylabel("Energia média  E", color=pal["ink2"], fontsize=11)
    ax_e.annotate(f"{df['E'].iloc[-1]:.3f}",
                  (x[-1], df["E"].iloc[-1]), xytext=(6, 0),
                  textcoords="offset points", va="center", fontsize=9,
                  color=pal["ink2"])
    plt.setp(ax_e.get_xticklabels(), visible=False)

    # ── Painel 3: Var(S) ─────────────────────────────────────────────────────
    _style_axis(ax_var, pal)
    bar_w = (x[-1] - x[0]) / (n + 1) * 0.85 if n > 1 else 0.01
    ax_var.bar(x, df["varS"], width=bar_w, color=pal["blue"], alpha=0.55,
               linewidth=0, zorder=3)
    ax_var.set_ylabel("Var(S)", fontsize=10, color=pal["ink2"])
    ax_var.set_xlabel(xlabel, fontsize=11.5, color=pal["ink2"])
    ax_var.xaxis.set_minor_locator(AutoMinorLocator())
    ax_var.xaxis.set_major_locator(MaxNLocator(nbins=10, prune=None))

    # ── Linha de transição (anotação, não série) ────────────────────────────
    if Tx is not None:
        for ax in (ax_s, ax_e, ax_var):
            ax.axvline(Tx, color=pal["ref"], linewidth=1.1,
                       linestyle=(0, (3, 3)), zorder=1)
        ax_s.annotate(f"transição ≈ {Tx:.3f}", (Tx, 1.0),
                      xytext=(6, -4), textcoords="offset points",
                      fontsize=9, color=pal["ink2"], rotation=0, va="top")

    for ax in (ax_s, ax_e, ax_var):
        ax.set_facecolor(pal["surface"])

    fig.align_ylabels([ax_s, ax_e, ax_var])

    plt.savefig(output_path, dpi=300, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"  Gráfico salvo em: {output_path}")

    if show:
        plt.show()

    plt.close()


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Analisa e plota os resultados do McLics-backend (po.dat).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "input", nargs="?", default="po.dat",
        help="Caminho para o arquivo po.dat (padrão: ./po.dat)"
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Nome do arquivo de saída do gráfico (padrão: <input>.png)"
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Exibe o gráfico interativo além de salvar"
    )
    parser.add_argument(
        "--txt", action="store_true",
        help="Imprime tabela de estatísticas no terminal"
    )
    parser.add_argument(
        "--dark", action="store_true",
        help="Usa tema escuro no gráfico (bom para apresentações/slides)"
    )
    parser.add_argument(
        "--mode", choices=sorted(MODE_INFO.keys()), default=None,
        help=("Modo de evolução da simulação (thermal | electric | step | quench). "
              "Não é detectável a partir de po.dat — o cabeçalho é sempre o mesmo "
              "independente do modo. Padrão: thermal.")
    )
    return parser.parse_args()


def main():
    args = parse_args()

    df, mode, xlabel, xcol = read_po_dat(args.input, forced_mode=args.mode)
    if args.mode is None:
        print(f"  [aviso] --mode não informado; assumindo 'thermal' "
              f"(po.dat não permite detectar o modo real). Use --mode se necessário.")

    # Nome de saída descritivo se não for especificado pelo usuário
    if args.output:
        output = args.output
    else:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = f"mclics_{mode}_{timestamp}.png"

    print(f"\n  McLics Analyzer")
    print(f"  {'─'*40}")
    print(f"  Entrada : {args.input}")
    print(f"  Saída   : {output}")
    print(f"  Modo    : {mode}")
    print(f"  Pontos  : {len(df)}")
    print(f"  {xcol} de {df['x'].min():.4f} até {df['x'].max():.4f}\n")

    if args.txt:
        print_stats(df, xlabel)

    plot(df, mode, xlabel, output, args.show, dark=args.dark)

    Tx = find_transition(df)
    if Tx is not None:
        print(f"  Transição de fase estimada em {xcol} ≈ {Tx:.4f}")

    print()


if __name__ == "__main__":
    main()